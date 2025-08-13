// In your main.cpp or wherever you use it:
SessionController controller;

// Initialize with your actual model paths
if (controller.initialize("./models/svm", "./models/ner")) {
    // Create a session
    auto result = controller.create_session("session123");
    std::cout << result.response << std::endl;
    std::cout << result.question << std::endl;
    
    // Update with user input
    auto update_result = controller.update_session("session123", "Hi, I'm John");
    std::cout << update_result.response << std::endl;
    
    // End session
    auto end_result = controller.end_session("session123");
}