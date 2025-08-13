
#include <httplib.h>

httplib::Server server;

// So we're importing the class server from httlib.h. 


void server::setup_routes() {
    // Enable CORS (equivalent to FastAPI CORS middleware)
    server_.set_pre_routing_handler([](const httplib::Request& req, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type, X-Session-ID");
        return httplib::Server::HandlerResponse::Unhandled;
    });

     // Handle OPTIONS requests (CORS preflight)
    server_.Options(".*", [](const httplib::Request&, httplib::Response& res) {
        return; // Headers already set in pre_routing_handler
    });

    // Route handlers - direct FastAPI equivalents
    server_.Post("/create_session", [this](const httplib::Request& req, httplib::Response& res) {
        handle_create_session(req, res);
    });

    server_.Post(R"(/update_session/(.+))", [this](const httplib::Request& req, httplib::Response& res) {
        handle_update_session(req, res);
    });

    server_.Post(R"(/end_session/(.+))", [this](const httplib::Request& req, httplib::Response& res) {
        handle_end_session(req, res);
    });

    server_.Get(R"(/get_session/(.+))", [this](const httplib::Request& req, httplib::Response& res) {
        handle_get_session(req, res);
    });

    server_.Get("/health", [this](const httplib::Request& req, httplib::Response& res) {
        handle_health_check(req, res);
    });
}


void HTTPServer::handle_create_session(const httplib::Request& req, httplib::Response& res) {
    try {
        // Extract session_id from header (like Python: session_id: str = Header(...))
        std::string session_id;
        if (req.has_header("X-Session-ID")) {
            session_id = req.get_header_value("X-Session-ID");
        }

        // Validation - exact same as Python
        if (session_id.empty()) {
            send_error(res, 400, "Session_id is missing");
            return;
        }

        // Check if session_id is valid
        if (session_id.find_first_not_of(" \t\n\r") == std::string::npos) {
            send_error(res, 400, "Session_id must not be an empty string");
            return;
        }

        {
            std::lock_guard<std::mutex> lock(sessions_mutex_);
            
            // If the session_id already exists, return an error
            if (active_sessions_.find(session_id) != active_sessions_.end()) {
                send_error(res, 409, "Session with ID " + session_id + " already exists");
                return;
            }

            // Create new controller (like Python: new_controller = SessionController(session_id))
            auto new_controller = std::make_unique<SessionController>();
            
            // Initialize with model paths
            if (!new_controller->initialize(svm_models_dir_, ner_models_dir_)) {
                send_error(res, 500, "Failed to initialize SessionController");
                return;
            }

            // Call create_session (like Python: result = await new_controller.create_session())
            auto result = new_controller->create_session(session_id);

            // Store in active_sessions (like Python: active_sessions[session_id] = new_controller)
            active_sessions_[session_id] = std::move(new_controller);

            // Return result (FastAPI auto-converts to JSON, we do it manually)
            send_json(res, entities_model_to_json(result));
        }

    } catch (const std::exception& e) {
        send_error(res, 500, "Internal server error: " + std::string(e.what()));
    }
}

void HTTPServer::handle_update_session(const httplib::Request& req, httplib::Response& res) {
    try {
        // Extract session_id from path parameters
        std::string session_id = req.matches[1];
        
        std::cout << "Accessed the update session API endpoint for session: " << session_id << std::endl;

        {
            std::lock_guard<std::mutex> lock(sessions_mutex_);
            
            // Check if session exists (like Python: if session_id not in active_sessions)
            auto it = active_sessions_.find(session_id);
            if (it == active_sessions_.end()) {
                std::cerr << "Attempt to update non-existent session: " << session_id << std::endl;
                send_error(res, 404, "Session not found");
                return;
            }

            // Parse dialogue_input from JSON body (like Python: dialogue_input: DialogueInput)
            if (req.body.empty()) {
                send_error(res, 400, "Request body is empty");
                return;
            }

            json request_json = json::parse(req.body);
            DialogueInput dialogue_input = DialogueInput::from_json(request_json);

            // Get controller and call update (like Python: controller = active_sessions[session_id])
            auto& controller = it->second;
            auto result = controller->update_session(session_id, dialogue_input.sentence);

            // Return result
            send_json(res, entities_model_to_json(result));
        }

    } catch (const json::exception& e) {
        send_error(res, 400, "Invalid JSON: " + std::string(e.what()));
    } catch (const std::exception& e) {
        send_error(res, 500, "Internal server error: " + std::string(e.what()));
    }
}

void HTTPServer::handle_end_session(const httplib::Request& req, httplib::Response& res) {
    try {
        // Extract session_id from path parameters
        std::string session_id = req.matches[1];

        {
            std::lock_guard<std::mutex> lock(sessions_mutex_);
            
            // Check if session exists (like Python validation)
            auto it = active_sessions_.find(session_id);
            if (it == active_sessions_.end()) {
                std::cerr << "Attempt to end non-existent session: " << session_id << std::endl;
                send_error(res, 404, "Session not found");
                return;
            }

            // Get controller and call end_session
            auto& controller = it->second;
            auto result = controller->end_session(session_id);

            // Remove from active_sessions (like Python: del active_sessions[session_id])
            active_sessions_.erase(it);

            // Return result
            send_json(res, entities_model_to_json(result));
        }

    } catch (const std::exception& e) {
        send_error(res, 500, "Internal server error: " + std::string(e.what()));
    }
}

void HTTPServer::handle_get_session(const httplib::Request& req, httplib::Response& res) {
    try {
        // Extract session_id from path parameters
        std::string session_id = req.matches[1];

        {
            std::lock_guard<std::mutex> lock(sessions_mutex_);
            
            // Check if session exists
            auto it = active_sessions_.find(session_id);
            if (it == active_sessions_.end()) {
                std::cerr << "Attempt to get non-existent session: " << session_id << std::endl;
                send_error(res, 404, "Session not found");
                return;
            }

            // Get controller and call get_session
            auto& controller = it->second;
            auto result = controller->get_session(session_id);

            // Return result
            send_json(res, entities_model_to_json(result));
        }

    } catch (const std::exception& e) {
        send_error(res, 500, "Internal server error: " + std::string(e.what()));
    }
}

void HTTPServer::handle_health_check(const httplib::Request& req, httplib::Response& res) {
    try {
        // Exact same as Python health check
        json health_response = {
            {"status", "Healthy"},
            {"message", "Multi AI Agent System is operational"},
            {"active_sessions", active_sessions_.size()}
        };

        send_json(res, health_response);

    } catch (const std::exception& e) {
        send_error(res, 500, "Internal server error: " + std::string(e.what()));
    }
}

json HTTPServer::entities_model_to_json(const EntitiesModel& model) const {
    json entities_json = {
        {"name", model.entities.name},
        {"phone", model.entities.phone},
        {"email", model.entities.email},
        {"service", model.entities.service},
        {"day", model.entities.day},
        {"time", model.entities.time},
        {"stylist", model.entities.stylist},
        {"notes", model.entities.notes}
    };

    return json{
        {"response", model.response},
        {"question", model.question},
        {"session_active", model.session_active},
        {"entities", entities_json}
    };
}

void HTTPServer::send_error(httplib::Response& res, int status_code, const std::string& message) const {
    json error_response = {
        {"detail", message}
    };
    res.status = status_code;
    res.set_content(error_response.dump(), "application/json");
}

void HTTPServer::send_json(httplib::Response& res, const json& data) const {
    res.set_content(data.dump(), "application/json");
}

bool HTTPServer::start(const std::string& host, int port) {
    std::cout << "Starting HTTP server on " << host << ":" << port << std::endl;
    std::cout << "Available endpoints:" << std::endl;
    std::cout << "  POST /create_session" << std::endl;
    std::cout << "  POST /update_session/{session_id}" << std::endl;
    std::cout << "  POST /end_session/{session_id}" << std::endl;
    std::cout << "  GET  /get_session/{session_id}" << std::endl;
    std::cout << "  GET  /health" << std::endl;

    return server_.listen(host.c_str(), port);
}

void HTTPServer::stop() {
    server_.stop();
}






