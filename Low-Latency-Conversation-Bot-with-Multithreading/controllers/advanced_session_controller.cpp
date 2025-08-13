#include "SessionController.h"
#include "classifier.h"
#include "extractor.h" 
#include "composer.h"
#include "closer.h"
#include <thread>
#include <algorithm>
#include <future>
#include <random>
#include <sstream>
#include <iostream>

// EntityStateManager Implementation - Simple and clean
void EntityStateManager::create_session(const std::string& session_id) {
    std::lock_guard<std::mutex> lock(sessions_mutex_);
    sessions_[session_id] = ConfigModel();  // Empty entities
    active_sessions_[session_id] = true;
}

ConfigModel EntityStateManager::get_session(const std::string& session_id) const {
    std::lock_guard<std::mutex> lock(sessions_mutex_);
    auto it = sessions_.find(session_id);
    if (it != sessions_.end()) {
        return it->second;
    }
    return ConfigModel();  // Return empty if not found
}

void EntityStateManager::update_session(const std::string& session_id, const ConfigModel& entities) {
    std::lock_guard<std::mutex> lock(sessions_mutex_);
    sessions_[session_id] = entities;
}

bool EntityStateManager::is_session_active(const std::string& session_id) const {
    std::lock_guard<std::mutex> lock(sessions_mutex_);
    auto it = active_sessions_.find(session_id);
    return it != active_sessions_.end() && it->second;
}

void EntityStateManager::set_session_active(const std::string& session_id, bool active) {
    std::lock_guard<std::mutex> lock(sessions_mutex_);
    active_sessions_[session_id] = active;
}

void EntityStateManager::end_session(const std::string& session_id) {
    std::lock_guard<std::mutex> lock(sessions_mutex_);
    sessions_.erase(session_id);
    active_sessions_.erase(session_id);
}

size_t EntityStateManager::get_session_count() const {
    std::lock_guard<std::mutex> lock(sessions_mutex_);
    return sessions_.size();
}
SessionController::SessionController() {
    // Simple thread management
    size_t cores = std::thread::hardware_concurrency();
    max_threads_ = cores > 4 ? cores / 2 : 2;
    state_manager_ = std::make_unique<EntityStateManager>();
}

bool SessionController::initialize(const std::string& svm_models_dir, const std::string& ner_models_dir) {
    try {
        // Initialize using your actual constructors from the header files
        classifier_ = std::make_unique<ClassificationCrew>(svm_models_dir, 0.5f);
        extractor_ = std::make_unique<ExtractionCrew>(ner_models_dir, 0.5f);
        composer_ = std::make_unique<ComposerCrew>(nullptr, max_threads_);  // Null LLM for now
        closer_ = std::make_unique<CloserCrew>(nullptr);  // Null LLM for now
        
        std::cout << "SessionController initialized successfully" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Exception during initialization: " << e.what() << std::endl;
        return false;
    }
}

std::vector<std::string> SessionController::group_entities(const std::vector<std::string>& empty_entities) const {
    // Simple grouping - max 2 entities
    std::vector<std::string> grouped;
    
    // Predefined pairs
    if (std::find(empty_entities.begin(), empty_entities.end(), "name") != empty_entities.end() &&
        std::find(empty_entities.begin(), empty_entities.end(), "phone") != empty_entities.end()) {
        grouped = {"name", "phone"};
    } else if (std::find(empty_entities.begin(), empty_entities.end(), "day") != empty_entities.end() &&
               std::find(empty_entities.begin(), empty_entities.end(), "time") != empty_entities.end()) {
        grouped = {"day", "time"};
    } else if (!empty_entities.empty()) {
        grouped.push_back(empty_entities[0]);
        if (empty_entities.size() > 1) {
            grouped.push_back(empty_entities[1]);
        }
    }
    
    return grouped;
}

std::string SessionController::generate_greeting() const {
    std::vector<std::string> greetings = {
        "Hello! I'm here to help you book your hair appointment.",
        "Hi there! I'd love to help you schedule your appointment.",
        "Welcome! Let's book your appointment together."
    };
    
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, greetings.size() - 1);
    
    return greetings[dis(gen)];
}

std::string SessionController::generate_question_for_entities(const std::vector<std::string>& entities) const {
    if (entities.empty()) {
        return "How can I help you today?";
    }
    
    if (entities.size() == 1) {
        const std::string& entity = entities[0];
        if (entity == "name") return "May I have your name, please?";
        if (entity == "phone") return "What's your phone number?";
        if (entity == "service") return "What service would you like?";
        if (entity == "day") return "What day works for you?";
        if (entity == "time") return "What time would you prefer?";
        return "Could you provide your " + entity + "?";
    } else if (entities.size() == 2) {
        return "Could you please provide your " + entities[0] + " and " + entities[1] + "?";
    }
    
    return "Could you provide some information?";
}

EntitiesModel SessionController::create_session(const std::string& session_id) {
    std::lock_guard<std::mutex> lock(controller_mutex_);
    
    EntitiesModel result;
    
    try {
        state_manager_->create_session(session_id);
        state_manager_->set_session_active(session_id, true);
        
        ConfigModel entities; // Empty by default
        auto empty_entities = entities.get_empty_entities();
        auto entities_to_ask = group_entities(empty_entities);
        
        result.response = generate_greeting();
        result.question = generate_question_for_entities(entities_to_ask);
        result.session_active = true;
        result.entities = entities;
        
        state_manager_->update_session(session_id, entities);
        
    } catch (const std::exception& e) {
        result.response = "Error creating session.";
        result.question = "";
        result.session_active = false;
    }
    
    return result;
}

EntitiesModel SessionController::get_session(const std::string& session_id) const {
    std::lock_guard<std::mutex> lock(controller_mutex_);
    
    EntitiesModel result;
    
    try {
        bool is_active = state_manager_->is_session_active(session_id);
        
        if (!is_active) {
            result.response = "Session not active";
            result.session_active = false;
            return result;
        }
        
        ConfigModel entities = state_manager_->get_session(session_id);
        result.entities = entities;
        result.session_active = is_active;
        
        auto empty_entities = entities.get_empty_entities();
        if (!empty_entities.empty()) {
            result.response = "Here's your current information:";
            auto entities_to_ask = group_entities(empty_entities);
            result.question = generate_question_for_entities(entities_to_ask);
        } else {
            result.response = "Your information is complete!";
            result.question = "All done!";
        }
        
    } catch (const std::exception& e) {
        result.response = "Error getting session.";
        result.session_active = false;
    }
    
    return result;
}

EntitiesModel SessionController::end_session(const std::string& session_id) {
    std::lock_guard<std::mutex> lock(controller_mutex_);
    
    EntitiesModel result;
    
    try {
        ConfigModel final_entities = state_manager_->get_session(session_id);
        bool was_active = state_manager_->is_session_active(session_id);
        
        if (was_active) {
            result.response = "Session ended successfully.";
        } else {
            result.response = "Session was already inactive.";
        }
        
        state_manager_->end_session(session_id);
        
        result.entities = final_entities;
        result.session_active = false;
        result.question = "";
        
    } catch (const std::exception& e) {
        result.response = "Error ending session.";
        result.session_active = false;
    }
    
    return result;
}

EntitiesModel SessionController::update_session(const std::string& session_id, const std::string& user_input) {
    std::lock_guard<std::mutex> lock(controller_mutex_);
    
    EntitiesModel result;
    
    try {
        if (!state_manager_->is_session_active(session_id)) {
            result.response = "Session not active.";
            result.session_active = false;
            return result;
        }
        
        ConfigModel current_entities = state_manager_->get_session(session_id);
        auto missing_entities = current_entities.get_empty_entities();
        
        // Use ONLY the methods that exist in your actual classes
        
        // Classification using your actual method
        auto detected_entities = classifier_->getDetectedEntities(user_input);
        
        // Find entities to extract
        std::vector<std::string> entities_to_extract;
        for (const auto& entity : missing_entities) {
            // Convert to your C++ entity names
            std::string cpp_entity = entity;
            if (entity == "name") cpp_entity = "caller_name";
            else if (entity == "phone") cpp_entity = "phone_number";
            else if (entity == "day") cpp_entity = "day_preference";
            else if (entity == "time") cpp_entity = "time_preference";
            else if (entity == "service") cpp_entity = "service_type";
            
            if (std::find(detected_entities.begin(), detected_entities.end(), cpp_entity) != detected_entities.end()) {
                entities_to_extract.push_back(cpp_entity);
            }
        }
        
        // Extraction using your actual method
        if (!entities_to_extract.empty()) {
            auto extraction_results = extractor_->extractEntities(user_input, entities_to_extract);
            
            // Update entities with results
            for (const auto& ext_result : extraction_results) {
                if (ext_result.found && !ext_result.extracted_value.empty()) {
                    // Convert back to simple names
                    std::string entity_name = ext_result.entity_name;
                    if (entity_name == "caller_name") entity_name = "name";
                    else if (entity_name == "phone_number") entity_name = "phone";
                    else if (entity_name == "day_preference") entity_name = "day";
                    else if (entity_name == "time_preference") entity_name = "time";
                    else if (entity_name == "service_type") entity_name = "service";
                    
                    current_entities.set_entity(entity_name, ext_result.extracted_value);
                }
            }
        }
        
        // Check if complete
        auto remaining_missing = current_entities.get_empty_entities();
        
        if (remaining_missing.empty()) {
            result.response = "Perfect! I have all your information.";
            result.question = "Your appointment is ready!";
        } else {
            result.response = "Thank you for that information.";
            auto next_entities = group_entities(remaining_missing);
            result.question = generate_question_for_entities(next_entities);
        }
        
        state_manager_->update_session(session_id, current_entities);
        
        result.entities = current_entities;
        result.session_active = true;
        
    } catch (const std::exception& e) {
        result.response = "Error processing input.";
        result.session_active = true;
        std::cerr << "Error: " << e.what() << std::endl;
    }
    
    return result;
}
