#ifndef COMPOSER_H
#define COMPOSER_H

#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <future>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>

// Composition request structure
struct CompositionRequest {
    std::vector<std::string> missing_entities;  // Up to 2 entities to ask about
    std::unordered_map<std::string, std::string> known_entities;  // Already extracted
    std::string conversation_context;  // Previous conversation
    
    CompositionRequest() = default;
    CompositionRequest(const std::vector<std::string>& missing, 
                      const std::unordered_map<std::string, std::string>& known,
                      const std::string& context = "")
        : missing_entities(missing), known_entities(known), conversation_context(context) {}
};

// Composition result structure
struct CompositionResult {
    std::string generated_question;
    std::vector<std::string> targeted_entities;  // Which entities this question targets
    float quality_score;  // LLM quality assessment
    bool is_valid;
    std::string generation_method;  // "llm_primary", "llm_fallback", "template"
    
    CompositionResult() 
        : generated_question(""), quality_score(0.0f), is_valid(false), generation_method("none") {}
};

// LLM API interface (you'll implement the actual API calls)
class LLMInterface {
public:
    virtual ~LLMInterface() = default;
    
    // Generate question for missing entities
    virtual std::string generateQuestion(const CompositionRequest& request) = 0;
    
    // Quality check the generated question
    virtual float assessQuestionQuality(const std::string& question, 
                                       const CompositionRequest& request) = 0;
    
    // Test if LLM is available/working
    virtual bool isAvailable() = 0;
};

// Thread-safe composer with LLM integration
class ComposerCrew {
private:
    std::unique_ptr<LLMInterface> llm_interface;
    
    // Thread pool for composition tasks
    std::vector<std::thread> worker_threads;
    std::queue<std::function<void()>> task_queue;
    std::mutex queue_mutex;
    std::condition_variable queue_condition;
    std::atomic<bool> stop_workers{false};
    
    // Configuration
    float quality_threshold;
    int max_retries;
    int num_worker_threads;
    
    // Template fallbacks
    std::unordered_map<std::string, std::vector<std::string>> entity_templates;
    
public:
    ComposerCrew(std::unique_ptr<LLMInterface> llm, int num_threads = 0);
    ~ComposerCrew();
    
    // Main composition functions
    std::future<CompositionResult> composeQuestionAsync(const CompositionRequest& request);
    CompositionResult composeQuestion(const CompositionRequest& request);
    
    // Batch composition for multiple missing entity groups
    std::vector<std::future<CompositionResult>> composeMultipleQuestionsAsync(
        const std::vector<CompositionRequest>& requests);
    
    // Configuration
    void setQualityThreshold(float threshold);
    void setMaxRetries(int retries);
    
    // Thread management
    void startWorkers();
    void stopWorkers();
    void adjustThreadCount(int new_count);
    
private:
    // Core composition logic
    CompositionResult generateWithLLM(const CompositionRequest& request);
    CompositionResult generateWithTemplate(const CompositionRequest& request);
    CompositionResult validateAndImprove(const CompositionResult& initial_result, 
                                        const CompositionRequest& request);
    
    // Template system
    void initializeTemplates();
    std::string selectTemplate(const std::vector<std::string>& entities);
    
    // Worker thread function
    void workerLoop();
    
    // Utility functions
    std::vector<std::vector<std::string>> groupMissingEntities(
        const std::vector<std::string>& missing_entities);
    bool areEntitiesRelated(const std::string& entity1, const std::string& entity2);
};

// Entity state manager interface
class EntityStateManager {
private:
    std::unordered_map<std::string, std::string> entity_values;
    std::vector<std::string> required_entities;
    mutable std::mutex state_mutex;
    
public:
    EntityStateManager();
    
    // Entity management
    void updateEntity(const std::string& entity_name, const std::string& value);
    std::string getEntity(const std::string& entity_name) const;
    bool hasEntity(const std::string& entity_name) const;
    
    // State queries
    std::vector<std::string> getMissingEntities() const;
    std::unordered_map<std::string, std::string> getKnownEntities() const;
    bool isComplete() const;
    float getCompletionPercentage() const;
    
    // Bulk operations
    void updateMultipleEntities(const std::unordered_map<std::string, std::string>& updates);
    void reset();
    
    // Thread-safe accessors
    std::vector<std::string> getRequiredEntities() const;
    void setRequiredEntities(const std::vector<std::string>& entities);
};

#endif // COMPOSER_H