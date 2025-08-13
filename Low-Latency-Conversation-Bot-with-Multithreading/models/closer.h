#ifndef CLOSER_H
#define CLOSER_H

#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <future>
#include <thread>
#include <atomic>

// Forward declaration
class LLMInterface;

// Closing request structure
struct ClosingRequest {
    std::unordered_map<std::string, std::string> complete_entities;  // All filled entities
    std::string conversation_summary;  // Summary of the conversation
    std::string business_context;  // Business/appointment context
    
    ClosingRequest() = default;
    ClosingRequest(const std::unordered_map<std::string, std::string>& entities,
                  const std::string& summary = "",
                  const std::string& context = "")
        : complete_entities(entities), conversation_summary(summary), business_context(context) {}
};

// Closing result structure
struct ClosingResult {
    std::string closing_message;  // Final message to user
    std::string appointment_summary;  // Summary for internal use
    std::string confirmation_details;  // Confirmation information
    bool needs_followup;  // Whether followup is required
    std::vector<std::string> next_steps;  // Any next steps for user/business
    float confidence_score;  // LLM confidence in the closing
    bool is_valid;
    std::string generation_method;  // "llm_primary", "llm_fallback", "template"
    
    ClosingResult()
        : needs_followup(false), confidence_score(0.0f), is_valid(false), generation_method("none") {}
};

// Appointment summary structure for internal processing
struct AppointmentSummary {
    std::string customer_name;
    std::string customer_phone;
    std::string preferred_day;
    std::string preferred_time;
    std::string service_requested;
    std::string booking_timestamp;
    std::string status;  // "confirmed", "pending", "needs_followup"
    
    // Convert to formatted string
    std::string toString() const;
    
    // Convert to JSON-like format
    std::string toJSON() const;
};

// Thread-safe closer with LLM integration
class CloserCrew {
private:
    std::unique_ptr<LLMInterface> llm_interface;
    
    // Configuration
    float confidence_threshold;
    int max_retries;
    std::atomic<int> active_tasks{0};
    
    // Template fallbacks
    std::unordered_map<std::string, std::vector<std::string>> closing_templates;
    std::unordered_map<std::string, std::vector<std::string>> confirmation_templates;
    
public:
    CloserCrew(std::unique_ptr<LLMInterface> llm);
    ~CloserCrew() = default;
    
    // Main closing functions
    std::future<ClosingResult> generateClosingAsync(const ClosingRequest& request);
    ClosingResult generateClosing(const ClosingRequest& request);
    
    // Appointment processing
    AppointmentSummary createAppointmentSummary(const ClosingRequest& request);
    std::future<AppointmentSummary> createAppointmentSummaryAsync(const ClosingRequest& request);
    
    // Validation and confirmation
    bool validateAppointmentData(const ClosingRequest& request);
    std::string generateConfirmationNumber();
    
    // Configuration
    void setConfidenceThreshold(float threshold);
    void setMaxRetries(int retries);
    
    // Status monitoring
    int getActiveTaskCount() const;
    bool isBusy() const;
    
private:
    // Core closing logic
    ClosingResult generateWithLLM(const ClosingRequest& request);
    ClosingResult generateWithTemplate(const ClosingRequest& request);
    ClosingResult validateAndImprove(const ClosingResult& initial_result, 
                                    const ClosingRequest& request);
    
    // Template system
    void initializeTemplates();
    std::string selectClosingTemplate(const ClosingRequest& request);
    std::string selectConfirmationTemplate(const ClosingRequest& request);
    
    // Utility functions
    std::string formatAppointmentDetails(const ClosingRequest& request);
    std::vector<std::string> generateNextSteps(const ClosingRequest& request);
    bool needsFollowup(const ClosingRequest& request);
    
    // Data validation
    bool isValidPhoneNumber(const std::string& phone);
    bool isValidName(const std::string& name);
    bool isValidTimeSlot(const std::string& day, const std::string& time);
};

// Business logic for appointment management
class AppointmentManager {
private:
    std::vector<AppointmentSummary> confirmed_appointments;
    mutable std::mutex appointments_mutex;
    
public:
    // Appointment storage
    bool storeAppointment(const AppointmentSummary& appointment);
    std::vector<AppointmentSummary> getAppointments() const;
    std::vector<AppointmentSummary> getAppointmentsByDay(const std::string& day) const;
    
    // Conflict checking
    bool hasTimeConflict(const std::string& day, const std::string& time) const;
    std::vector<std::string> getSuggestedAlternatives(const std::string& day, const std::string& time) const;
    
    // Statistics
    int getTotalAppointments() const;
    std::unordered_map<std::string, int> getServiceCounts() const;
    
    // Cleanup
    void clearOldAppointments();
    void reset();
};

#endif // CLOSER_H