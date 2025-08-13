#include "closer.h"
#include "composer.h"  // For LLMInterface
#include <algorithm>
#include <random>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <regex>

// AppointmentSummary Implementation
std::string AppointmentSummary::toString() const {
    std::stringstream ss;
    ss << "Appointment Summary:\n"
       << "  Customer: " << customer_name << "\n"
       << "  Phone: " << customer_phone << "\n"
       << "  Service: " << service_requested << "\n"
       << "  Preferred Day: " << preferred_day << "\n"
       << "  Preferred Time: " << preferred_time << "\n"
       << "  Status: " << status << "\n"
       << "  Booked: " << booking_timestamp;
    return ss.str();
}

std::string AppointmentSummary::toJSON() const {
    std::stringstream ss;
    ss << "{\n"
       << "  \"customer_name\": \"" << customer_name << "\",\n"
       << "  \"customer_phone\": \"" << customer_phone << "\",\n"
       << "  \"service_requested\": \"" << service_requested << "\",\n"
       << "  \"preferred_day\": \"" << preferred_day << "\",\n"
       << "  \"preferred_time\": \"" << preferred_time << "\",\n"
       << "  \"status\": \"" << status << "\",\n"
       << "  \"booking_timestamp\": \"" << booking_timestamp << "\"\n"
       << "}";
    return ss.str();
}

// CloserCrew Implementation
CloserCrew::CloserCrew(std::unique_ptr<LLMInterface> llm) 
    : llm_interface(std::move(llm)), confidence_threshold(0.8f), max_retries(2) {
    
    std::cout << "🎯 Closer Crew initialized" << std::endl;
    initializeTemplates();
}

void CloserCrew::initializeTemplates() {
    // Closing message templates
    closing_templates["standard"] = {
        "Perfect! I have all the information I need. Let me confirm your appointment:",
        "Excellent! I've got everything we need to schedule your appointment:",
        "Great! Here's a summary of your appointment request:"
    };
    
    closing_templates["needs_confirmation"] = {
        "I have all the details for your appointment. Let me just confirm everything with you:",
        "Perfect! Before we finalize, let me read back your appointment details:",
        "Excellent! Here's what I have scheduled for you - please confirm:"
    };
    
    // Confirmation templates
    confirmation_templates["standard"] = {
        "Your appointment has been confirmed! You'll receive a confirmation text shortly.",
        "All set! We've confirmed your appointment and will send you a reminder.",
        "Perfect! Your appointment is confirmed. You should receive a confirmation message soon."
    };
    
    confirmation_templates["with_followup"] = {
        "Your appointment request has been received! We'll call you back within 24 hours to confirm the exact time.",
        "Thank you! We have your request and will contact you shortly to finalize the details.",
        "Got it! We'll reach out to you soon to confirm your preferred time slot."
    };
    
    confirmation_templates["needs_callback"] = {
        "Thanks for the information! Someone from our team will call you back to confirm availability.",
        "We have your details! Our scheduler will contact you to confirm your appointment time.",
        "Perfect! We'll have someone call you back to verify the appointment details."
    };
}

std::future<ClosingResult> CloserCrew::generateClosingAsync(const ClosingRequest& request) {
    return std::async(std::launch::async, [this, request]() {
        active_tasks++;
        try {
            auto result = generateClosing(request);
            active_tasks--;
            return result;
        } catch (const std::exception& e) {
            active_tasks--;
            std::cerr << "Closing task failed: " << e.what() << std::endl;
            ClosingResult error_result;
            error_result.closing_message = "Thank you for your interest! We'll be in touch soon.";
            error_result.is_valid = false;
            return error_result;
        }
    });
}

ClosingResult CloserCrew::generateClosing(const ClosingRequest& request) {
    std::cout << "🎯 Generating closing for complete appointment..." << std::endl;
    
    // Validate appointment data first
    if (!validateAppointmentData(request)) {
        std::cout << "  ⚠️ Invalid appointment data, using fallback closing" << std::endl;
        return generateWithTemplate(request);
    }
    
    ClosingResult result;
    
    // Try LLM first
    if (llm_interface && llm_interface->isAvailable()) {
        result = generateWithLLM(request);
        
        // Quality check and potential retry
        if (result.is_valid && result.confidence_score < confidence_threshold) {
            std::cout << "  📊 Confidence too low (" << result.confidence_score << "), retrying..." << std::endl;
            
            for (int retry = 0; retry < max_retries; ++retry) {
                auto retry_result = generateWithLLM(request);
                if (retry_result.confidence_score > result.confidence_score) {
                    result = retry_result;
                    break;
                }
            }
        }
    }
    
    // Fallback to templates if LLM fails or confidence is poor
    if (!result.is_valid || result.confidence_score < confidence_threshold) {
        std::cout << "  🔄 Using template fallback..." << std::endl;
        result = generateWithTemplate(request);
    }
    
    // Add appointment details and next steps
    result.appointment_summary = formatAppointmentDetails(request);
    result.confirmation_details = generateConfirmationNumber();
    result.next_steps = generateNextSteps(request);
    result.needs_followup = needsFollowup(request);
    
    std::cout << "  ✅ Generated closing: \"" << result.closing_message << "\"" << std::endl;
    std::cout << "  📊 Confidence: " << std::fixed << std::setprecision(2) << result.confidence_score 
              << " (" << result.generation_method << ")" << std::endl;
    
    return result;
}

ClosingResult CloserCrew::generateWithLLM(const ClosingRequest& request) {
    ClosingResult result;
    
    try {
        // Generate closing using LLM
        std::string closing = llm_interface->generateQuestion(CompositionRequest({}, request.complete_entities));  // Convert to CompositionRequest
        
        if (!closing.empty()) {
            result.closing_message = closing;
            result.generation_method = "llm_primary";
            
            // Assess quality (reusing quality assessment)
            result.confidence_score = llm_interface->assessQuestionQuality(closing, 
                CompositionRequest({}, request.complete_entities));  // Convert to composition request
            result.is_valid = true;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "LLM closing generation failed: " << e.what() << std::endl;
        result.is_valid = false;
    }
    
    return result;
}

ClosingResult CloserCrew::generateWithTemplate(const ClosingRequest& request) {
    ClosingResult result;
    
    try {
        // Select appropriate template
        std::string template_key = needsFollowup(request) ? "needs_confirmation" : "standard";
        std::string closing_template = selectClosingTemplate(request);
        std::string confirmation_template = selectConfirmationTemplate(request);
        
        // Build closing message
        std::stringstream closing_msg;
        closing_msg << closing_template << "\n\n";
        closing_msg << formatAppointmentDetails(request) << "\n\n";
        closing_msg << confirmation_template;
        
        result.closing_message = closing_msg.str();
        result.confidence_score = 0.85f;  // Templates get good confidence
        result.is_valid = true;
        result.generation_method = "template";
        
    } catch (const std::exception& e) {
        std::cerr << "Template closing generation failed: " << e.what() << std::endl;
        result.closing_message = "Thank you for booking with us! We'll be in touch soon.";
        result.confidence_score = 0.6f;
        result.is_valid = true;
        result.generation_method = "template_fallback";
    }
    
    return result;
}

AppointmentSummary CloserCrew::createAppointmentSummary(const ClosingRequest& request) {
    AppointmentSummary summary;
    
    // Extract entity values
    auto entities = request.complete_entities;
    summary.customer_name = entities.count("caller_name") ? entities["caller_name"] : "Unknown";
    summary.customer_phone = entities.count("phone_number") ? entities["phone_number"] : "Unknown";
    summary.preferred_day = entities.count("day_preference") ? entities["day_preference"] : "Unknown";
    summary.preferred_time = entities.count("time_preference") ? entities["time_preference"] : "Unknown";
    summary.service_requested = entities.count("service_type") ? entities["service_type"] : "Unknown";
    
    // Add metadata
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
    summary.booking_timestamp = ss.str();
    
    summary.status = needsFollowup(request) ? "pending" : "confirmed";
    
    return summary;
}

std::future<AppointmentSummary> CloserCrew::createAppointmentSummaryAsync(const ClosingRequest& request) {
    return std::async(std::launch::async, [this, request]() {
        return createAppointmentSummary(request);
    });
}

bool CloserCrew::validateAppointmentData(const ClosingRequest& request) {
    auto entities = request.complete_entities;
    
    // Check required fields
    std::vector<std::string> required = {"caller_name", "phone_number", "day_preference", 
                                        "time_preference", "service_type"};
    
    for (const auto& field : required) {
        if (entities.count(field) == 0 || entities[field].empty()) {
            std::cout << "  ❌ Missing required field: " << field << std::endl;
            return false;
        }
    }
    
    // Validate specific fields
    if (!isValidName(entities["caller_name"])) {
        std::cout << "  ❌ Invalid name format" << std::endl;
        return false;
    }
    
    if (!isValidPhoneNumber(entities["phone_number"])) {
        std::cout << "  ❌ Invalid phone number format" << std::endl;
        return false;
    }
    
    if (!isValidTimeSlot(entities["day_preference"], entities["time_preference"])) {
        std::cout << "  ❌ Invalid time slot" << std::endl;
        return false;
    }
    
    return true;
}

std::string CloserCrew::generateConfirmationNumber() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(100000, 999999);
    
    return "APT" + std::to_string(dis(gen));
}

std::string CloserCrew::selectClosingTemplate(const ClosingRequest& request) {
    std::string template_key = needsFollowup(request) ? "needs_confirmation" : "standard";
    
    auto it = closing_templates.find(template_key);
    if (it != closing_templates.end() && !it->second.empty()) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, it->second.size() - 1);
        return it->second[dis(gen)];
    }
    
    return "Thank you for scheduling your appointment!";
}

std::string CloserCrew::selectConfirmationTemplate(const ClosingRequest& request) {
    std::string template_key;
    if (needsFollowup(request)) {
        template_key = "with_followup";
    } else {
        template_key = "standard";
    }
    
    auto it = confirmation_templates.find(template_key);
    if (it != confirmation_templates.end() && !it->second.empty()) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, it->second.size() - 1);
        return it->second[dis(gen)];
    }
    
    return "We'll be in touch soon!";
}

std::string CloserCrew::formatAppointmentDetails(const ClosingRequest& request) {
    std::stringstream ss;
    auto entities = request.complete_entities;
    
    ss << "📋 Appointment Details:\n";
    ss << "   Name: " << entities["caller_name"] << "\n";
    ss << "   Phone: " << entities["phone_number"] << "\n";
    ss << "   Service: " << entities["service_type"] << "\n";
    ss << "   Day: " << entities["day_preference"] << "\n";
    ss << "   Time: " << entities["time_preference"];
    
    return ss.str();
}

std::vector<std::string> CloserCrew::generateNextSteps(const ClosingRequest& request) {
    std::vector<std::string> steps;
    
    if (needsFollowup(request)) {
        steps.push_back("Wait for confirmation call within 24 hours");
        steps.push_back("Keep your phone available for our call");
        steps.push_back("Prepare any questions about the service");
    } else {
        steps.push_back("Watch for confirmation text message");
        steps.push_back("Arrive 10 minutes early for your appointment");
        steps.push_back("Bring valid ID if this is your first visit");
    }
    
    return steps;
}

bool CloserCrew::needsFollowup(const ClosingRequest& request) {
    // Simple logic: if time is too vague or conflicts might exist
    auto entities = request.complete_entities;
    
    if (entities.count("time_preference") == 0) return true;
    
    std::string time = entities["time_preference"];
    // If time is vague (morning, afternoon, etc.) rather than specific
    if (time.find("morning") != std::string::npos || 
        time.find("afternoon") != std::string::npos ||
        time.find("evening") != std::string::npos) {
        return true;
    }
    
    return false;
}

bool CloserCrew::isValidPhoneNumber(const std::string& phone) {
    // Simple regex for phone validation
    std::regex phone_regex(R"(\d{3}-\d{3}-\d{4}|\(\d{3}\)\s*\d{3}-\d{4}|\d{10})");
    return std::regex_match(phone, phone_regex);
}

bool CloserCrew::isValidName(const std::string& name) {
    return !name.empty() && name.length() >= 2 && name.length() <= 50;
}

bool CloserCrew::isValidTimeSlot(const std::string& day, const std::string& time) {
    // Basic validation
    std::vector<std::string> valid_days = {"Monday", "Tuesday", "Wednesday", "Thursday", 
                                          "Friday", "Saturday", "Sunday"};
    
    bool valid_day = std::find(valid_days.begin(), valid_days.end(), day) != valid_days.end();
    bool valid_time = !time.empty();
    
    return valid_day && valid_time;
}

void CloserCrew::setConfidenceThreshold(float threshold) {
    confidence_threshold = threshold;
}

void CloserCrew::setMaxRetries(int retries) {
    max_retries = retries;
}

int CloserCrew::getActiveTaskCount() const {
    return active_tasks.load();
}

bool CloserCrew::isBusy() const {
    return active_tasks.load() > 0;
}

// AppointmentManager Implementation
bool AppointmentManager::storeAppointment(const AppointmentSummary& appointment) {
    std::lock_guard<std::mutex> lock(appointments_mutex);
    
    // Check for conflicts
    if (hasTimeConflict(appointment.preferred_day, appointment.preferred_time)) {
        std::cout << "  ⚠️ Time conflict detected for " << appointment.preferred_day 
                  << " at " << appointment.preferred_time << std::endl;
        return false;
    }
    
    confirmed_appointments.push_back(appointment);
    std::cout << "  ✅ Stored appointment for " << appointment.customer_name << std::endl;
    return true;
}

std::vector<AppointmentSummary> AppointmentManager::getAppointments() const {
    std::lock_guard<std::mutex> lock(appointments_mutex);
    return confirmed_appointments;
}

std::vector<AppointmentSummary> AppointmentManager::getAppointmentsByDay(const std::string& day) const {
    std::lock_guard<std::mutex> lock(appointments_mutex);
    std::vector<AppointmentSummary> day_appointments;
    
    for (const auto& apt : confirmed_appointments) {
        if (apt.preferred_day == day) {
            day_appointments.push_back(apt);
        }
    }
    
    return day_appointments;
}

bool AppointmentManager::hasTimeConflict(const std::string& day, const std::string& time) const {
    std::lock_guard<std::mutex> lock(appointments_mutex);
    
    for (const auto& apt : confirmed_appointments) {
        if (apt.preferred_day == day && apt.preferred_time == time) {
            return true;
        }
    }
    
    return false;
}

std::vector<std::string> AppointmentManager::getSuggestedAlternatives(const std::string& day, const std::string& time) const {
    // Simple alternative suggestions
    std::vector<std::string> alternatives;
    alternatives.push_back("Earlier time on " + day);
    alternatives.push_back("Later time on " + day);
    alternatives.push_back("Same time on different day");
    
    return alternatives;
}

int AppointmentManager::getTotalAppointments() const {
    std::lock_guard<std::mutex> lock(appointments_mutex);
    return confirmed_appointments.size();
}

std::unordered_map<std::string, int> AppointmentManager::getServiceCounts() const {
    std::lock_guard<std::mutex> lock(appointments_mutex);
    std::unordered_map<std::string, int> counts;
    
    for (const auto& apt : confirmed_appointments) {
        counts[apt.service_requested]++;
    }
    
    return counts;
}

void AppointmentManager::clearOldAppointments() {
    std::lock_guard<std::mutex> lock(appointments_mutex);
    // For now, just clear all - in real implementation, check dates
    confirmed_appointments.clear();
}

void AppointmentManager::reset() {
    std::lock_guard<std::mutex> lock(appointments_mutex);
    confirmed_appointments.clear();
}