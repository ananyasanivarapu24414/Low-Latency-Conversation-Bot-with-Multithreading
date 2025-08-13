#include "composer.h"
#include <algorithm>
#include <random>
#include <sstream>
#include <iomanip>

// ComposerCrew Implementation
ComposerCrew::ComposerCrew(std::unique_ptr<LLMInterface> llm, int num_threads) 
    : llm_interface(std::move(llm)), quality_threshold(0.7f), max_retries(2) {
    
    // Determine optimal thread count
    if (num_threads <= 0) {
        num_worker_threads = std::max(1u, std::thread::hardware_concurrency() / 2);
    } else {
        num_worker_threads = num_threads;
    }
    
    std::cout << "🎵 Composer Crew initialized with " << num_worker_threads << " worker threads" << std::endl;
    
    initializeTemplates();
    startWorkers();
}

ComposerCrew::~ComposerCrew() {
    stopWorkers();
}

void ComposerCrew::initializeTemplates() {
    // Template fallbacks for when LLM fails
    entity_templates["caller_name+phone_number"] = {
        "Great! Can you please tell me your name and phone number?",
        "I'd like to get your name and contact number, please.",
        "Could you provide your name and a phone number where I can reach you?"
    };
    
    entity_templates["day_preference+time_preference"] = {
        "What day and time would work best for your appointment?",
        "When would you prefer to schedule this? What day and time?",
        "Could you let me know your preferred day and time?"
    };
    
    entity_templates["service_type+time_preference"] = {
        "What service are you looking for and what time would work for you?",
        "Which service do you need and when would you prefer to come in?",
        "What type of appointment do you need and what time works best?"
    };
    
    // Single entity templates
    entity_templates["caller_name"] = {
        "May I have your name, please?",
        "Could you tell me your name?",
        "What name should I put this appointment under?"
    };
    
    entity_templates["phone_number"] = {
        "What's the best phone number to reach you at?",
        "Could I get a contact number for you?",
        "What phone number should I use for this appointment?"
    };
    
    entity_templates["day_preference"] = {
        "What day would work best for you?",
        "Which day would you prefer for your appointment?",
        "What day are you looking to schedule this?"
    };
    
    entity_templates["time_preference"] = {
        "What time would work best for you?",
        "Do you have a preferred time?",
        "What time would you like to come in?"
    };
    
    entity_templates["service_type"] = {
        "What service are you looking for today?",
        "Which service do you need?",
        "What type of appointment would you like to schedule?"
    };
}

void ComposerCrew::startWorkers() {
    for (int i = 0; i < num_worker_threads; ++i) {
        worker_threads.emplace_back(&ComposerCrew::workerLoop, this);
    }
}

void ComposerCrew::stopWorkers() {
    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        stop_workers = true;
    }
    queue_condition.notify_all();
    
    for (auto& thread : worker_threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    worker_threads.clear();
}

void ComposerCrew::workerLoop() {
    while (true) {
        std::function<void()> task;
        
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            queue_condition.wait(lock, [this] { 
                return !task_queue.empty() || stop_workers; 
            });
            
            if (stop_workers && task_queue.empty()) {
                break;
            }
            
            task = std::move(task_queue.front());
            task_queue.pop();
        }
        
        task();
    }
}

std::future<CompositionResult> ComposerCrew::composeQuestionAsync(const CompositionRequest& request) {
    auto promise = std::make_shared<std::promise<CompositionResult>>();
    auto future = promise->get_future();
    
    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        task_queue.push([this, request, promise]() {
            try {
                auto result = composeQuestion(request);
                promise->set_value(result);
            } catch (const std::exception& e) {
                std::cerr << "Composition task failed: " << e.what() << std::endl;
                CompositionResult error_result;
                error_result.generated_question = "I apologize, but I'm having trouble generating a question right now.";
                error_result.is_valid = false;
                promise->set_value(error_result);
            }
        });
    }
    queue_condition.notify_one();
    
    return future;
}

CompositionResult ComposerCrew::composeQuestion(const CompositionRequest& request) {
    std::cout << "🎵 Composing question for " << request.missing_entities.size() << " missing entities..." << std::endl;
    
    // Limit to 2 entities as requested
    CompositionRequest limited_request = request;
    if (limited_request.missing_entities.size() > 2) {
        limited_request.missing_entities.resize(2);
    }
    
    CompositionResult result;
    
    // Try LLM first
    if (llm_interface && llm_interface->isAvailable()) {
        result = generateWithLLM(limited_request);
        
        // Quality check and potential retry
        if (result.is_valid && result.quality_score < quality_threshold) {
            std::cout << "  📊 Quality score too low (" << result.quality_score << "), retrying..." << std::endl;
            
            for (int retry = 0; retry < max_retries; ++retry) {
                auto retry_result = generateWithLLM(limited_request);
                if (retry_result.quality_score > result.quality_score) {
                    result = retry_result;
                    break;
                }
            }
        }
    }
    
    // Fallback to templates if LLM fails or quality is poor
    if (!result.is_valid || result.quality_score < quality_threshold) {
        std::cout << "  🔄 Using template fallback..." << std::endl;
        result = generateWithTemplate(limited_request);
    }
    
    result.targeted_entities = limited_request.missing_entities;
    
    std::cout << "  ✅ Generated: \"" << result.generated_question << "\"" << std::endl;
    std::cout << "  📊 Quality: " << std::fixed << std::setprecision(2) << result.quality_score 
              << " (" << result.generation_method << ")" << std::endl;
    
    return result;
}

CompositionResult ComposerCrew::generateWithLLM(const CompositionRequest& request) {
    CompositionResult result;
    
    try {
        // Generate question using LLM
        std::string question = llm_interface->generateQuestion(request);
        
        if (!question.empty()) {
            result.generated_question = question;
            result.generation_method = "llm_primary";
            
            // Quality assessment
            result.quality_score = llm_interface->assessQuestionQuality(question, request);
            result.is_valid = true;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "LLM generation failed: " << e.what() << std::endl;
        result.is_valid = false;
    }
    
    return result;
}

CompositionResult ComposerCrew::generateWithTemplate(const CompositionRequest& request) {
    CompositionResult result;
    
    // Create entity combination key
    std::string template_key;
    if (request.missing_entities.size() == 2) {
        template_key = request.missing_entities[0] + "+" + request.missing_entities[1];
    } else if (request.missing_entities.size() == 1) {
        template_key = request.missing_entities[0];
    }
    
    // Find appropriate template
    auto it = entity_templates.find(template_key);
    if (it != entity_templates.end() && !it->second.empty()) {
        // Randomly select a template variant
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, it->second.size() - 1);
        
        result.generated_question = it->second[dis(gen)];
        result.quality_score = 0.8f;  // Templates get decent quality score
        result.is_valid = true;
        result.generation_method = "template";
    } else {
        // Generic fallback
        result.generated_question = "Could you please provide some additional information?";
        result.quality_score = 0.5f;
        result.is_valid = true;
        result.generation_method = "template_fallback";
    }
    
    return result;
}

std::vector<std::future<CompositionResult>> ComposerCrew::composeMultipleQuestionsAsync(
    const std::vector<CompositionRequest>& requests) {
    
    std::vector<std::future<CompositionResult>> futures;
    
    for (const auto& request : requests) {
        futures.push_back(composeQuestionAsync(request));
    }
    
    return futures;
}

std::vector<std::vector<std::string>> ComposerCrew::groupMissingEntities(
    const std::vector<std::string>& missing_entities) {
    
    std::vector<std::vector<std::string>> groups;
    std::vector<std::string> remaining = missing_entities;
    
    while (!remaining.empty()) {
        std::vector<std::string> group;
        group.push_back(remaining[0]);
        remaining.erase(remaining.begin());
        
        // Try to find a related entity to pair with
        for (auto it = remaining.begin(); it != remaining.end(); ++it) {
            if (areEntitiesRelated(group[0], *it)) {
                group.push_back(*it);
                remaining.erase(it);
                break;  // Only pair 2 entities max
            }
        }
        
        groups.push_back(group);
    }
    
    return groups;
}

bool ComposerCrew::areEntitiesRelated(const std::string& entity1, const std::string& entity2) {
    // Define related entity pairs
    std::vector<std::pair<std::string, std::string>> related_pairs = {
        {"caller_name", "phone_number"},
        {"day_preference", "time_preference"},
        {"service_type", "time_preference"},
        {"service_type", "day_preference"}
    };
    
    for (const auto& pair : related_pairs) {
        if ((entity1 == pair.first && entity2 == pair.second) ||
            (entity1 == pair.second && entity2 == pair.first)) {
            return true;
        }
    }
    
    return false;
}

void ComposerCrew::adjustThreadCount(int new_count) {
    if (new_count != num_worker_threads) {
        stopWorkers();
        num_worker_threads = new_count;
        startWorkers();
        std::cout << "🔧 Adjusted Composer thread count to " << new_count << std::endl;
    }
}

void ComposerCrew::setQualityThreshold(float threshold) {
    quality_threshold = threshold;
}

void ComposerCrew::setMaxRetries(int retries) {
    max_retries = retries;
}

// EntityStateManager Implementation
EntityStateManager::EntityStateManager() {
    required_entities = {
        "caller_name", "phone_number", "day_preference", 
        "time_preference", "service_type"
    };
}

void EntityStateManager::updateEntity(const std::string& entity_name, const std::string& value) {
    std::lock_guard<std::mutex> lock(state_mutex);
    entity_values[entity_name] = value;
    std::cout << "📝 Updated " << entity_name << " = \"" << value << "\"" << std::endl;
}

std::string EntityStateManager::getEntity(const std::string& entity_name) const {
    std::lock_guard<std::mutex> lock(state_mutex);
    auto it = entity_values.find(entity_name);
    return (it != entity_values.end()) ? it->second : "";
}

bool EntityStateManager::hasEntity(const std::string& entity_name) const {
    std::lock_guard<std::mutex> lock(state_mutex);
    auto it = entity_values.find(entity_name);
    return (it != entity_values.end() && !it->second.empty());
}

std::vector<std::string> EntityStateManager::getMissingEntities() const {
    std::lock_guard<std::mutex> lock(state_mutex);
    std::vector<std::string> missing;
    
    for (const auto& entity : required_entities) {
        auto it = entity_values.find(entity);
        if (it == entity_values.end() || it->second.empty()) {
            missing.push_back(entity);
        }
    }
    
    return missing;
}

std::unordered_map<std::string, std::string> EntityStateManager::getKnownEntities() const {
    std::lock_guard<std::mutex> lock(state_mutex);
    std::unordered_map<std::string, std::string> known;
    
    for (const auto& pair : entity_values) {
        if (!pair.second.empty()) {
            known[pair.first] = pair.second;
        }
    }
    
    return known;
}

bool EntityStateManager::isComplete() const {
    return getMissingEntities().empty();
}

float EntityStateManager::getCompletionPercentage() const {
    std::lock_guard<std::mutex> lock(state_mutex);
    int filled = 0;
    
    for (const auto& entity : required_entities) {
        auto it = entity_values.find(entity);
        if (it != entity_values.end() && !it->second.empty()) {
            filled++;
        }
    }
    
    return static_cast<float>(filled) / required_entities.size() * 100.0f;
}

void EntityStateManager::updateMultipleEntities(const std::unordered_map<std::string, std::string>& updates) {
    std::lock_guard<std::mutex> lock(state_mutex);
    for (const auto& pair : updates) {
        entity_values[pair.first] = pair.second;
    }
}

void EntityStateManager::reset() {
    std::lock_guard<std::mutex> lock(state_mutex);
    entity_values.clear();
}

std::vector<std::string> EntityStateManager::getRequiredEntities() const {
    std::lock_guard<std::mutex> lock(state_mutex);
    return required_entities;
}

void EntityStateManager::setRequiredEntities(const std::vector<std::string>& entities) {
    std::lock_guard<std::mutex> lock(state_mutex);
    required_entities = entities;
}