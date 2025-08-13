#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <future>
#include <fstream>

// ONNX Runtime
#include <onnxruntime/onnxruntime_cxx_api.h>

// Classification result structure
struct ClassificationResult {
    std::string entity_name;
    float confidence;
    bool detected;
    
    ClassificationResult(const std::string& name) 
        : entity_name(name), confidence(0.0f), detected(false) {}
};

// SVM Model wrapper (handles TF-IDF pipelines)
class SVMModel {
private:
    std::unique_ptr<Ort::Session> session;
    Ort::Env env;
    
public:
    SVMModel(const std::string& model_path);
    float predict(const std::string& text);
};

// Classification Crew - handles entity detection
class ClassificationCrew {
private:
    std::unordered_map<std::string, std::unique_ptr<SVMModel>> svm_models;
    float confidence_threshold;
    std::vector<std::string> entity_types;
    
public:
    ClassificationCrew(const std::string& svm_models_dir, float threshold = 0.7f);
    
    // Load all SVM models
    void loadSVMModels(const std::string& models_dir);
    
    // Classify single entity async
    std::future<ClassificationResult> classifyEntityAsync(const std::string& sentence, const std::string& entity_type);
    
    // Classify all entities in parallel
    std::vector<ClassificationResult> classifyAllEntities(const std::string& input_sentence);
    
    // Get detected entities (above threshold)
    std::vector<std::string> getDetectedEntities(const std::string& input_sentence);
    
    void setConfidenceThreshold(float threshold);
    void printClassificationResults(const std::vector<ClassificationResult>& results);
};

#endif // CLASSIFIER_H