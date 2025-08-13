#include "classifier.h"
#include <algorithm>
#include <iomanip>

// SVM Model Implementation
SVMModel::SVMModel(const std::string& model_path) : env{ORT_LOGGING_LEVEL_WARNING, "SVMModel"} {
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    
    try {
        session = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);
        
        // Debug: Print input/output names
        std::cout << "  Input names: ";
        for (size_t i = 0; i < session->GetInputCount(); i++) {
            std::cout << session->GetInputNameAllocated(i, Ort::AllocatorWithDefaultOptions()).get() << " ";
        }
        std::cout << std::endl;
        
        std::cout << "  Output names: ";
        for (size_t i = 0; i < session->GetOutputCount(); i++) {
            std::cout << session->GetOutputNameAllocated(i, Ort::AllocatorWithDefaultOptions()).get() << " ";
        }
        std::cout << std::endl;
        
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to load SVM model: " + std::string(e.what()));
    }
}

float SVMModel::predict(const std::string& text) {
    try {
        // Create string tensor for input
        std::vector<std::string> input_strings = {text};
        
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        
        // Convert to char pointers
        std::vector<const char*> raw_strings;
        for (const auto& str : input_strings) {
            raw_strings.push_back(str.c_str());
        }
        
        // Shape for string tensor
        std::vector<int64_t> input_shape = {static_cast<int64_t>(input_strings.size())};
        
        // Create string tensor
        auto input_tensor = Ort::Value::CreateTensor(
            memory_info,
            const_cast<void*>(static_cast<const void*>(raw_strings.data())),
            raw_strings.size() * sizeof(const char*),
            input_shape.data(),
            input_shape.size(),
            ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING
        );
        
        // Run inference
        const char* input_names[] = {"text_input"};
        const char* output_names[] = {"output_probability"};
        
        auto output_tensors = session->Run(
            Ort::RunOptions{nullptr}, 
            input_names, &input_tensor, 1,
            output_names, 1
        );
        
        // Get probability for positive class
        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        
        auto shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
        
        // Return probability of class 1 (entity present)
        if (shape.size() > 0 && shape[shape.size()-1] > 1) {
            return output_data[1]; // Return probability of class 1
        } else {
            return output_data[0]; // Fallback to first value
        }
        
    } catch (const std::exception& e) {
        std::cerr << "SVM prediction error: " << e.what() << std::endl;
        return 0.0f;
    }
}

// Classification Crew Implementation
ClassificationCrew::ClassificationCrew(const std::string& svm_models_dir, float threshold) 
    : confidence_threshold(threshold) {
    
    entity_types = {
        "caller_name", "phone_number", "day_preference", 
        "time_preference", "service_type"
    };
    
    loadSVMModels(svm_models_dir);
}

void ClassificationCrew::loadSVMModels(const std::string& models_dir) {
    std::cout << "🔄 Loading SVM Classification Models..." << std::endl;
    
    for (const auto& entity : entity_types) {
        std::string model_path = models_dir + "/" + entity + "_svm.onnx";
        
        try {
            svm_models[entity] = std::make_unique<SVMModel>(model_path);
            std::cout << "✅ Loaded SVM classifier for " << entity << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "❌ Failed to load SVM classifier for " << entity << ": " << e.what() << std::endl;
        }
    }
}

std::future<ClassificationResult> ClassificationCrew::classifyEntityAsync(const std::string& sentence, const std::string& entity_type) {
    return std::async(std::launch::async, [this, sentence, entity_type]() {
        ClassificationResult result(entity_type);
        
        try {
            if (svm_models.find(entity_type) != svm_models.end()) {
                float confidence = svm_models[entity_type]->predict(sentence);
                result.confidence = confidence;
                result.detected = (confidence >= confidence_threshold);
            }
        } catch (const std::exception& e) {
            std::cerr << "Error classifying " << entity_type << ": " << e.what() << std::endl;
        }
        
        return result;
    });
}

std::vector<ClassificationResult> ClassificationCrew::classifyAllEntities(const std::string& input_sentence) {
    std::vector<std::future<ClassificationResult>> futures;
    
    // Launch async classification for all entities
    for (const auto& entity : entity_types) {
        futures.push_back(classifyEntityAsync(input_sentence, entity));
    }
    
    // Collect results
    std::vector<ClassificationResult> results;
    for (auto& future : futures) {
        results.push_back(future.get());
    }
    
    return results;
}

std::vector<std::string> ClassificationCrew::getDetectedEntities(const std::string& input_sentence) {
    auto classification_results = classifyAllEntities(input_sentence);
    std::vector<std::string> detected_entities;
    
    for (const auto& result : classification_results) {
        if (result.detected) {
            detected_entities.push_back(result.entity_name);
        }
    }
    
    return detected_entities;
}

void ClassificationCrew::setConfidenceThreshold(float threshold) {
    confidence_threshold = threshold;
}

void ClassificationCrew::printClassificationResults(const std::vector<ClassificationResult>& results) {
    std::cout << "\n🔍 Classification Results:" << std::endl;
    std::cout << "=========================" << std::endl;
    
    for (const auto& result : results) {
        std::cout << std::setw(15) << result.entity_name << ": ";
        
        if (result.detected) {
            std::cout << "✅ DETECTED ";
        } else {
            std::cout << "❌ NOT DETECTED ";
        }
        
        std::cout << "(confidence: " << std::fixed << std::setprecision(3) << result.confidence << ")";
        std::cout << std::endl;
    }
    std::cout << std::endl;
}