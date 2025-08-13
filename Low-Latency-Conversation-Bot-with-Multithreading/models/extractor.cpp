#include "extractor.h"
#include <algorithm>
#include <iomanip>

// NER Model Implementation
NERModel::NERModel(const std::string& model_path, const std::string& metadata_path) 
    : env{ORT_LOGGING_LEVEL_WARNING, "NERModel"} {
    
    // Load metadata
    std::ifstream metadata_file(metadata_path);
    if (!metadata_file.is_open()) {
        throw std::runtime_error("Cannot open NER metadata file: " + metadata_path);
    }
    
    json metadata;
    metadata_file >> metadata;
    
    word_to_idx = metadata["word_to_idx"].get<std::unordered_map<std::string, int>>();
    label_classes = metadata["label_classes"].get<std::vector<std::string>>();
    vocab_size = metadata["vocab_size"].get<int>();
    max_length = metadata["max_length"].get<int>();
    
    // Load ONNX model
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    
    try {
        session = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to load NER model: " + std::string(e.what()));
    }
}

std::vector<int> NERModel::tokenize(const std::string& text) {
    std::vector<int> tokens;
    std::istringstream iss(text);
    std::string word;
    
    // Convert to lowercase and split by spaces
    std::string lower_text = text;
    std::transform(lower_text.begin(), lower_text.end(), lower_text.begin(), ::tolower);
    std::istringstream lower_iss(lower_text);
    
    while (lower_iss >> word && tokens.size() < max_length) {
        auto it = word_to_idx.find(word);
        if (it != word_to_idx.end()) {
            tokens.push_back(it->second);
        } else {
            tokens.push_back(word_to_idx.at("<UNK>")); // Unknown token
        }
    }
    
    // Pad to max_length
    while (tokens.size() < max_length) {
        tokens.push_back(word_to_idx.at("<PAD>")); // Padding token
    }
    
    return tokens;
}

std::string NERModel::extract(const std::string& text) {
    try {
        // Tokenize input
        std::vector<int> tokens = tokenize(text);
        
        // Convert to int64 for ONNX
        std::vector<int64_t> input_ids(tokens.begin(), tokens.end());
        
        // Create input tensor
        std::vector<int64_t> input_shape = {1, max_length}; // [batch_size, seq_len]
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        
        Ort::Value input_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info, input_ids.data(), input_ids.size(), 
            input_shape.data(), input_shape.size()
        );
        
        // Run inference
        const char* input_names[] = {"input_ids"};
        const char* output_names[] = {"logits"};
        
        auto output_tensors = session->Run(
            Ort::RunOptions{nullptr}, 
            input_names, &input_tensor, 1,
            output_names, 1
        );
        
        // Get predictions
        float* logits = output_tensors[0].GetTensorMutableData<float>();
        auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
        int seq_len = output_shape[1];
        int num_labels = output_shape[2];
        
        // Find predicted labels (argmax)
        std::vector<std::string> words;
        std::istringstream iss(text);
        std::string word;
        while (iss >> word) {
            words.push_back(word);
        }
        
        // Extract entities
        for (int i = 0; i < std::min(seq_len, static_cast<int>(words.size())); i++) {
            int best_label = 0;
            float best_score = logits[i * num_labels];
            
            for (int j = 1; j < num_labels; j++) {
                if (logits[i * num_labels + j] > best_score) {
                    best_score = logits[i * num_labels + j];
                    best_label = j;
                }
            }
            
            if (best_label < label_classes.size()) {
                std::string label = label_classes[best_label];
                if (label.find("B-") == 0) { // Beginning of entity
                    return words[i]; // Return the word
                }
            }
        }
        
        return ""; // No entity found
        
    } catch (const std::exception& e) {
        std::cerr << "NER extraction error: " << e.what() << std::endl;
        return "";
    }
}

// Extraction Crew Implementation
ExtractionCrew::ExtractionCrew(const std::string& ner_models_dir, float threshold) 
    : ner_confidence_threshold(threshold) {
    loadNERModels(ner_models_dir);
}

void ExtractionCrew::loadNERModels(const std::string& models_dir) {
    std::cout << "🔄 Loading NER Extraction Models..." << std::endl;
    
    std::vector<std::string> entity_types = {
        "caller_name", "phone_number", "day_preference", 
        "time_preference", "service_type"
    };
    
    for (const auto& entity : entity_types) {
        std::string model_path = models_dir + "/" + entity + "_ner.onnx";
        std::string metadata_path = models_dir + "/" + entity + "_metadata.json";
        
        try {
            ner_models[entity] = std::make_unique<NERModel>(model_path, metadata_path);
            std::cout << "✅ Loaded NER extractor for " << entity << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "❌ Failed to load NER extractor for " << entity << ": " << e.what() << std::endl;
        }
    }
}

std::future<ExtractionResult> ExtractionCrew::extractEntityAsync(const std::string& sentence, const std::string& entity_type) {
    return std::async(std::launch::async, [this, sentence, entity_type]() {
        ExtractionResult result(entity_type);
        
        try {
            if (ner_models.find(entity_type) != ner_models.end()) {
                std::string extracted = ner_models[entity_type]->extract(sentence);
                
                if (!extracted.empty()) {
                    result.found = true;
                    result.extracted_value = extracted;
                    result.ner_confidence = 1.0f; // Simplified for now
                    result.method_used = "ner";
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "Error extracting " << entity_type << ": " << e.what() << std::endl;
        }
        
        return result;
    });
}

std::vector<ExtractionResult> ExtractionCrew::extractEntities(const std::string& input_sentence, const std::vector<std::string>& target_entities) {
    std::vector<std::future<ExtractionResult>> futures;
    
    // Launch async extraction for target entities only
    for (const auto& entity : target_entities) {
        futures.push_back(extractEntityAsync(input_sentence, entity));
    }
    
    // Collect results
    std::vector<ExtractionResult> results;
    for (auto& future : futures) {
        results.push_back(future.get());
    }
    
    return results;
}

ExtractionResult ExtractionCrew::llmFallback(const std::string& sentence, const std::string& entity_type) {
    ExtractionResult result(entity_type);
    std::cout << "🔄 LLM fallback triggered for extraction of " << entity_type << std::endl;
    
    // TODO: Implement your LLM API call here
    // This is where you would call your existing LLM crew system
    // For now, return empty result
    
    result.method_used = "llm_fallback";
    return result;
}

std::vector<ExtractionResult> ExtractionCrew::extractWithFallback(const std::string& input_sentence, const std::vector<std::string>& target_entities) {
    auto results = extractEntities(input_sentence, target_entities);
    
    // Use LLM fallback for entities not extracted by NER models
    for (auto& result : results) {
        if (!result.found || result.ner_confidence < ner_confidence_threshold) {
            auto llm_result = llmFallback(input_sentence, result.entity_name);
            if (llm_result.found) {
                result = llm_result;
            }
        }
    }
    
    return results;
}

void ExtractionCrew::setNERConfidenceThreshold(float threshold) {
    ner_confidence_threshold = threshold;
}

void ExtractionCrew::printExtractionResults(const std::vector<ExtractionResult>& results) {
    std::cout << "\n🎯 Extraction Results:" << std::endl;
    std::cout << "=====================" << std::endl;
    
    for (const auto& result : results) {
        std::cout << std::setw(15) << result.entity_name << ": ";
        
        if (result.found) {
            std::cout << "✅ \"" << result.extracted_value << "\" ";
            std::cout << "(method: " << result.method_used;
            if (result.method_used == "ner") {
                std::cout << ", confidence: " << std::fixed << std::setprecision(2) << result.ner_confidence;
            }
            std::cout << ")";
        } else {
            std::cout << "❌ Not extracted";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}