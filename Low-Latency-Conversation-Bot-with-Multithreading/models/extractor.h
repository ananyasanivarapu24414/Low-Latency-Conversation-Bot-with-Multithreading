#ifndef EXTRACTOR_H
#define EXTRACTOR_H

#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <future>
#include <fstream>
#include <sstream>

// ONNX Runtime
#include <onnxruntime/onnxruntime_cxx_api.h>

// JSON library 
#include <nlohmann/json.hpp>
using json = nlohmann::json;

// Extraction result structure
struct ExtractionResult {
    std::string entity_name;
    std::string extracted_value;
    float ner_confidence;
    bool found;
    std::string method_used; // "ner", "llm_fallback"
    
    ExtractionResult(const std::string& name) 
        : entity_name(name), extracted_value(""), ner_confidence(0.0f), 
          found(false), method_used("none") {}
};

// NER Model wrapper
class NERModel {
private:
    std::unique_ptr<Ort::Session> session;
    Ort::Env env;
    std::unordered_map<std::string, int> word_to_idx;
    std::vector<std::string> label_classes;
    int vocab_size;
    int max_length;
    
public:
    NERModel(const std::string& model_path, const std::string& metadata_path);
    std::vector<int> tokenize(const std::string& text);
    std::string extract(const std::string& text);
};

// Extraction Crew - handles entity value extraction
class ExtractionCrew {
private:
    std::unordered_map<std::string, std::unique_ptr<NERModel>> ner_models;
    float ner_confidence_threshold;
    
public:
    ExtractionCrew(const std::string& ner_models_dir, float threshold = 0.5f);
    
    // Load all NER models
    void loadNERModels(const std::string& models_dir);
    
    // Extract single entity async
    std::future<ExtractionResult> extractEntityAsync(const std::string& sentence, const std::string& entity_type);
    
    // Extract given entities in parallel
    std::vector<ExtractionResult> extractEntities(const std::string& input_sentence, const std::vector<std::string>& target_entities);
    
    // LLM fallback for low-confidence extractions
    ExtractionResult llmFallback(const std::string& sentence, const std::string& entity_type);
    
    // Extract with LLM fallback
    std::vector<ExtractionResult> extractWithFallback(const std::string& input_sentence, const std::vector<std::string>& target_entities);
    
    void setNERConfidenceThreshold(float threshold);
    void printExtractionResults(const std::vector<ExtractionResult>& results);
};

#endif // EXTRACTOR_H