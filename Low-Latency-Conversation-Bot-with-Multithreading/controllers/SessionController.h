#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include <mutex>
#include <chrono>

// Forward declarations for your actual wrapper classes
class ClassificationCrew;
class ExtractionCrew;
class ComposerCrew;
class CloserCrew;

// C++ equivalent of Python's ConfigModel
struct ConfigModel {
    std::string name;
    std::string phone;
    std::string email;
    std::string service;
    std::string day;
    std::string time;
    std::string stylist;
    std::string notes;
    
    // Default constructor initializes all to empty
    ConfigModel() = default;
    
    // Check if entity is empty
    bool is_empty(const std::string& field) const {
        return field.empty();
    }
    
    // Get all empty entities
    std::vector<std::string> get_empty_entities() const {
        std::vector<std::string> empty;
        if (name.empty()) empty.push_back("name");
        if (phone.empty()) empty.push_back("phone");
        if (email.empty()) empty.push_back("email");
        if (service.empty()) empty.push_back("service");
        if (day.empty()) empty.push_back("day");
        if (time.empty()) empty.push_back("time");
        if (stylist.empty()) empty.push_back("stylist");
        if (notes.empty()) empty.push_back("notes");
        return empty;
    }
    
    // Set entity value
    void set_entity(const std::string& entity_name, const std::string& value) {
        if (entity_name == "name") name = value;
        else if (entity_name == "phone") phone = value;
        else if (entity_name == "email") email = value;
        else if (entity_name == "service") service = value;
        else if (entity_name == "day") day = value;
        else if (entity_name == "time") time = value;
        else if (entity_name == "stylist") stylist = value;
        else if (entity_name == "notes") notes = value;
    }
    
    // Get entity value
    std::string get_entity(const std::string& entity_name) const {
        if (entity_name == "name") return name;
        else if (entity_name == "phone") return phone;
        else if (entity_name == "email") return email;
        else if (entity_name == "service") return service;
        else if (entity_name == "day") return day;
        else if (entity_name == "time") return time;
        else if (entity_name == "stylist") return stylist;
        else if (entity_name == "notes") return notes;
        return "";
    }
};

// C++ equivalent of Python's EntitiesModel  
struct EntitiesModel {
    std::string response;
    std::string question;
    bool session_active;
    ConfigModel entities;
    
    EntitiesModel() : session_active(false) {}
};

// Simple EntityStateManager for sessions
class EntityStateManager {
private:
    std::unordered_map<std::string, ConfigModel> sessions_;
    std::unordered_map<std::string, bool> active_sessions_;
    mutable std::mutex sessions_mutex_;
    
public:
    // Basic session management
    void create_session(const std::string& session_id);
    ConfigModel get_session(const std::string& session_id) const;
    void update_session(const std::string& session_id, const ConfigModel& entities);
    bool is_session_active(const std::string& session_id) const;
    void set_session_active(const std::string& session_id, bool active);
    void end_session(const std::string& session_id);
    // Removed get_session_count - not needed
};

// Main SessionController class
class SessionController {
private:
    // Your actual wrapper classes
    std::unique_ptr<ClassificationCrew> classifier_;
    std::unique_ptr<ExtractionCrew> extractor_;
    std::unique_ptr<ComposerCrew> composer_;
    std::unique_ptr<CloserCrew> closer_;
    
    // Simple state manager
    std::unique_ptr<EntityStateManager> state_manager_;
    
    // Threading
    size_t max_threads_;
    mutable std::mutex controller_mutex_;
    
    // Helper methods
    std::vector<std::string> group_entities(const std::vector<std::string>& empty_entities) const;
    std::string generate_greeting() const;
    std::string generate_question_for_entities(const std::vector<std::string>& entities) const;
    
public:
    // Constructor/Destructor
    SessionController();
    ~SessionController() = default;
    
    // Initialize with actual model paths
    bool initialize(const std::string& svm_models_dir, const std::string& ner_models_dir);
    
    // Main session methods
    EntitiesModel create_session(const std::string& session_id);
    EntitiesModel update_session(const std::string& session_id, const std::string& user_input);
    EntitiesModel get_session(const std::string& session_id) const;
    EntitiesModel end_session(const std::string& session_id);
    
    // That's it - no extra utility methods needed
};