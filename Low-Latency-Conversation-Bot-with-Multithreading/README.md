
Low Latency Conversation Bot with Multithreading. 

A multithreaded C++ system for intelligent appointment booking using machine learning models for entity detection, extraction, and conversation management.

## Overview

The Session Controller is an advanced appointment booking system that uses SVM classifiers and NER (Named Entity Recognition) models to intelligently process customer conversations and extract relevant information such as names, phone numbers, preferred dates, times, and services.

## Features

- **Multithreaded Processing**: Parallel execution of classification, extraction, and composition tasks
- **Machine Learning Integration**: SVM models for entity detection and ONNX NER models for value extraction
- **Intelligent Conversation Flow**: Template-based and LLM-powered question generation
- **Thread-Safe Session Management**: Concurrent handling of multiple customer sessions
- **Fallback Systems**: Multiple layers of redundancy for robust operation
- **Performance Monitoring**: Real-time metrics and system optimization

## Architecture

### Core Components

1. **SessionController**: Main orchestrator managing session lifecycle
2. **ClassificationCrew**: SVM-based entity detection in customer input
3. **ExtractionCrew**: NER-based entity value extraction
4. **ComposerCrew**: Intelligent question generation for missing information
5. **CloserCrew**: Appointment confirmation and closing message generation
6. **EntityStateManager**: Thread-safe entity state tracking

### Data Flow

1. Customer input is classified to detect present entities
2. Detected entities are extracted using NER models
3. Missing entities trigger intelligent question composition
4. Complete entity sets trigger appointment confirmation
5. All operations are parallelized for optimal performance

## Dependencies

### Required Libraries

- **ONNX Runtime**: Machine learning model inference
- **nlohmann/json**: JSON parsing for model metadata
- **C++17 Standard Library**: Threading, futures, and STL containers

### Model Requirements

- SVM models in ONNX format for entity classification
- NER models in ONNX format with corresponding metadata JSON files
- Models should be organized in separate directories

## Installation

### Prerequisites

```bash
# Install ONNX Runtime (macOS with Homebrew)
brew install onnxruntime

# Install nlohmann/json
brew install nlohmann-json

# Ensure C++17 compiler
g++ --version  # GCC 7+ or Clang 5+
```

### Build Instructions

```bash
# Compile the main application
g++ -std=c++17 client.cpp SessionController.cpp classifier.cpp extractor.cpp composer.cpp closer.cpp \
    -I/opt/homebrew/Cellar/onnxruntime/1.22.1/include \
    -L/opt/homebrew/Cellar/onnxruntime/1.22.1/lib \
    -lonnxruntime \
    -pthread \
    -o session_controller

# For advanced multithreaded version
g++ -std=c++17 advanced_session_controller.cpp classifier.cpp extractor.cpp composer.cpp closer.cpp \
    -I/opt/homebrew/Cellar/onnxruntime/1.22.1/include \
    -L/opt/homebrew/Cellar/onnxruntime/1.22.1/lib \
    -lonnxruntime \
    -pthread \
    -o advanced_controller
```

## Usage

### Basic Session Management

```cpp
#include "SessionController.h"

int main() {
    SessionController controller;
    
    // Initialize with model paths
    if (controller.initialize("./models/svm", "./models/ner")) {
        // Create a new session
        auto result = controller.create_session("session123");
        std::cout << result.response << std::endl;
        std::cout << result.question << std::endl;
        
        // Process customer input
        auto update_result = controller.update_session("session123", "Hi, I'm John");
        std::cout << update_result.response << std::endl;
        
        // End session when complete
        auto end_result = controller.end_session("session123");
    }
    
    return 0;
}
```

### Advanced Multithreaded Processing

```cpp
#include "advanced_session_controller.cpp"

int main() {
    // Create LLM interface
    auto llm_interface = std::make_unique<ConcreteLLMInterface>();
    
    // Initialize advanced controller
    AdvancedSessionController controller(
        "./models/onnx_svm",    // SVM models directory
        "./models/onnx_ner",    // NER models directory  
        std::move(llm_interface),
        0.7f,  // Classification threshold
        0.5f   // Extraction threshold
    );
    
    // Process input with full multithreading
    auto result = controller.processInput("Hi I'm John, my number is 555-123-4567");
    controller.printProcessingResults(result);
    
    return 0;
}
```

## Configuration

### Model Directory Structure

```
models/
├── svm/
│   ├── caller_name_svm.onnx
│   ├── phone_number_svm.onnx
│   ├── day_preference_svm.onnx
│   ├── time_preference_svm.onnx
│   └── service_type_svm.onnx
└── ner/
    ├── caller_name_ner.onnx
    ├── caller_name_metadata.json
    ├── phone_number_ner.onnx
    ├── phone_number_metadata.json
    └── ... (similar for other entities)
```

### Entity Configuration

The system tracks five core entities:

- `caller_name`: Customer's name
- `phone_number`: Contact phone number
- `day_preference`: Preferred appointment day
- `time_preference`: Preferred appointment time
- `service_type`: Requested service

### Thread Configuration

Thread allocation is automatically optimized based on CPU cores:

- **8+ cores**: Aggressive parallelization (2 threads per component)
- **4-7 cores**: Balanced approach (1-2 threads per component)
- **<4 cores**: Conservative threading (1 thread per component)

## API Reference

### SessionController Class

#### Methods

- `bool initialize(const std::string& svm_models_dir, const std::string& ner_models_dir)`
- `EntitiesModel create_session(const std::string& session_id)`
- `EntitiesModel update_session(const std::string& session_id, const std::string& user_input)`
- `EntitiesModel get_session(const std::string& session_id)`
- `EntitiesModel end_session(const std::string& session_id)`

### EntitiesModel Structure

```cpp
struct EntitiesModel {
    std::string response;      // System response to user
    std::string question;      // Next question to ask
    bool session_active;       // Session status
    ConfigModel entities;      // Current entity values
};
```

### ConfigModel Structure

```cpp
struct ConfigModel {
    std::string name;
    std::string phone;
    std::string email;
    std::string service;
    std::string day;
    std::string time;
    std::string stylist;
    std::string notes;
};
```

## Performance Optimization

### Multithreading Strategy

1. **Classification Phase**: All SVM models run in parallel
2. **Extraction + Composition Phase**: Concurrent entity extraction and question generation
3. **Closing Phase**: Asynchronous appointment confirmation and storage

### Performance Monitoring

The system provides real-time metrics:

- Classification time
- Extraction time
- Composition time
- Total processing time
- CPU core utilization
- Concurrent task count

### Memory Management

- RAII principles throughout
- Smart pointer usage for automatic cleanup
- Thread-safe data structures
- Minimal memory allocation in hot paths

## Error Handling

### Robust Fallback Systems

1. **Model Loading**: Graceful degradation if models fail to load
2. **Inference Errors**: Fallback to template-based responses
3. **LLM Failures**: Template-based question generation
4. **Threading Issues**: Automatic thread count adjustment

### Logging and Debugging

- Comprehensive error messages
- Performance metric logging
- Thread safety validation
- Model inference debugging

## Testing

### Unit Testing

Test individual components:

```bash
# Test classification crew
./test_classifier "Hi I'm John" 

# Test extraction crew  
./test_extractor "My number is 555-123-4567"

# Test session controller
./test_session_controller
```

### Integration Testing

```bash
# Full system test
./session_controller

# Advanced system test
./advanced_controller
```

## Troubleshooting

### Common Issues

1. **Model Loading Failures**
   - Verify model file paths
   - Check ONNX Runtime installation
   - Validate model file permissions

2. **Threading Issues**
   - Check system thread limits
   - Monitor CPU usage
   - Adjust thread counts if needed

3. **Memory Issues**
   - Monitor memory usage during processing
   - Check for memory leaks in long-running sessions
   - Validate model size compatibility

### Debug Mode

Enable debug output by setting logging level:

```cpp
// In main.cpp
std::cout.setf(std::ios::boolalpha);  // Enable verbose output
```

## Contributing

### Code Style

- Follow C++17 standards
- Use RAII principles
- Prefer smart pointers over raw pointers
- Thread-safe by default
- Comprehensive error handling

### Testing Requirements

- Unit tests for all components
- Integration tests for full workflows
- Performance benchmarks
- Memory leak detection

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Support

For issues and questions:

1. Check troubleshooting section
2. Review error logs
3. Validate model files and paths
4. Test with provided examples

## Version History

- **v1.0**: Basic session management with SVM classification
- **v2.0**: Added NER extraction and multithreading
- **v3.0**: Advanced multithreaded processing with LLM integration
- **v3.1**: Performance optimizations and enhanced error handling

A multithreaded C++ system for intelligent appointment booking using machine learning (SVM, NER) for entity detection, extraction, and conversation management.

## Overview

The Session Controller is an advanced appointment booking system that uses SVM classifiers and NER (Named Entity Recognition) models to intelligently process customer conversations and extract relevant information such as names, phone numbers, preferred dates, times, and services.

## Features

- **Multithreaded Processing**: Parallel execution of classification, extraction, and composition tasks
- **Machine Learning Integration**: SVM models for entity detection and ONNX NER models for value extraction
- **Intelligent Conversation Flow**: Template-based and LLM-powered question generation
- **Thread-Safe Session Management**: Concurrent handling of multiple customer sessions
- **Fallback Systems**: Multiple layers of redundancy for robust operation
- **Performance Monitoring**: Real-time metrics and system optimization

## Architecture

### Core Components

1. **SessionController**: Main orchestrator managing session lifecycle
2. **ClassificationCrew**: SVM-based entity detection in customer input
3. **ExtractionCrew**: NER-based entity value extraction
4. **ComposerCrew**: Intelligent question generation for missing information
5. **CloserCrew**: Appointment confirmation and closing message generation
6. **EntityStateManager**: Thread-safe entity state tracking

### Data Flow

1. Customer input is classified to detect present entities
2. Detected entities are extracted using NER models
3. Missing entities trigger intelligent question composition
4. Complete entity sets trigger appointment confirmation
5. All operations are parallelized for optimal performance

## Dependencies

### Required Libraries

- **ONNX Runtime**: Machine learning model inference
- **nlohmann/json**: JSON parsing for model metadata
- **C++17 Standard Library**: Threading, futures, and STL containers

### Model Requirements

- SVM models in ONNX format for entity classification
- NER models in ONNX format with corresponding metadata JSON files
- Models should be organized in separate directories

## Installation

### Prerequisites

```bash
# Install ONNX Runtime (macOS with Homebrew)
brew install onnxruntime

# Install nlohmann/json
brew install nlohmann-json

# Ensure C++17 compiler
g++ --version  # GCC 7+ or Clang 5+
```

### Build Instructions

```bash
# Compile the main application
g++ -std=c++17 client.cpp SessionController.cpp classifier.cpp extractor.cpp composer.cpp closer.cpp \
    -I/opt/homebrew/Cellar/onnxruntime/1.22.1/include \
    -L/opt/homebrew/Cellar/onnxruntime/1.22.1/lib \
    -lonnxruntime \
    -pthread \
    -o session_controller

# For advanced multithreaded version
g++ -std=c++17 advanced_session_controller.cpp classifier.cpp extractor.cpp composer.cpp closer.cpp \
    -I/opt/homebrew/Cellar/onnxruntime/1.22.1/include \
    -L/opt/homebrew/Cellar/onnxruntime/1.22.1/lib \
    -lonnxruntime \
    -pthread \
    -o advanced_controller
```

## Usage

### Basic Session Management

```cpp
#include "SessionController.h"

int main() {
    SessionController controller;
    
    // Initialize with model paths
    if (controller.initialize("./models/svm", "./models/ner")) {
        // Create a new session
        auto result = controller.create_session("session123");
        std::cout << result.response << std::endl;
        std::cout << result.question << std::endl;
        
        // Process customer input
        auto update_result = controller.update_session("session123", "Hi, I'm John");
        std::cout << update_result.response << std::endl;
        
        // End session when complete
        auto end_result = controller.end_session("session123");
    }
    
    return 0;
}
```

### Advanced Multithreaded Processing

```cpp
#include "advanced_session_controller.cpp"

int main() {
    // Create LLM interface
    auto llm_interface = std::make_unique<ConcreteLLMInterface>();
    
    // Initialize advanced controller
    AdvancedSessionController controller(
        "./models/onnx_svm",    // SVM models directory
        "./models/onnx_ner",    // NER models directory  
        std::move(llm_interface),
        0.7f,  // Classification threshold
        0.5f   // Extraction threshold
    );
    
    // Process input with full multithreading
    auto result = controller.processInput("Hi I'm John, my number is 555-123-4567");
    controller.printProcessingResults(result);
    
    return 0;
}
```

## Configuration

### Model Directory Structure

```
models/
├── svm/
│   ├── caller_name_svm.onnx
│   ├── phone_number_svm.onnx
│   ├── day_preference_svm.onnx
│   ├── time_preference_svm.onnx
│   └── service_type_svm.onnx
└── ner/
    ├── caller_name_ner.onnx
    ├── caller_name_metadata.json
    ├── phone_number_ner.onnx
    ├── phone_number_metadata.json
    └── ... (similar for other entities)
```

### Entity Configuration

The system tracks five core entities:

- `caller_name`: Customer's name
- `phone_number`: Contact phone number
- `day_preference`: Preferred appointment day
- `time_preference`: Preferred appointment time
- `service_type`: Requested service

### Thread Configuration

Thread allocation is automatically optimized based on CPU cores:

- **8+ cores**: Aggressive parallelization (2 threads per component)
- **4-7 cores**: Balanced approach (1-2 threads per component)
- **<4 cores**: Conservative threading (1 thread per component)

## API Reference

### SessionController Class

#### Methods

- `bool initialize(const std::string& svm_models_dir, const std::string& ner_models_dir)`
- `EntitiesModel create_session(const std::string& session_id)`
- `EntitiesModel update_session(const std::string& session_id, const std::string& user_input)`
- `EntitiesModel get_session(const std::string& session_id)`
- `EntitiesModel end_session(const std::string& session_id)`

### EntitiesModel Structure

```cpp
struct EntitiesModel {
    std::string response;      // System response to user
    std::string question;      // Next question to ask
    bool session_active;       // Session status
    ConfigModel entities;      // Current entity values
};
```

### ConfigModel Structure

```cpp
struct ConfigModel {
    std::string name;
    std::string phone;
    std::string email;
    std::string service;
    std::string day;
    std::string time;
    std::string stylist;
    std::string notes;
};
```

## Performance Optimization

### Multithreading Strategy

1. **Classification Phase**: All SVM models run in parallel
2. **Extraction + Composition Phase**: Concurrent entity extraction and question generation
3. **Closing Phase**: Asynchronous appointment confirmation and storage

### Performance Monitoring

The system provides real-time metrics:

- Classification time
- Extraction time
- Composition time
- Total processing time
- CPU core utilization
- Concurrent task count

### Memory Management

- RAII principles throughout
- Smart pointer usage for automatic cleanup
- Thread-safe data structures
- Minimal memory allocation in hot paths

## Error Handling

### Robust Fallback Systems

1. **Model Loading**: Graceful degradation if models fail to load
2. **Inference Errors**: Fallback to template-based responses
3. **LLM Failures**: Template-based question generation
4. **Threading Issues**: Automatic thread count adjustment

### Logging and Debugging

- Comprehensive error messages
- Performance metric logging
- Thread safety validation
- Model inference debugging

## Testing

### Unit Testing

Test individual components:

```bash
# Test classification crew
./test_classifier "Hi I'm John" 

# Test extraction crew  
./test_extractor "My number is 555-123-4567"

# Test session controller
./test_session_controller
```

### Integration Testing

```bash
# Full system test
./session_controller

# Advanced system test
./advanced_controller
```

## Troubleshooting

### Common Issues

1. **Model Loading Failures**
   - Verify model file paths
   - Check ONNX Runtime installation
   - Validate model file permissions

2. **Threading Issues**
   - Check system thread limits
   - Monitor CPU usage
   - Adjust thread counts if needed

3. **Memory Issues**
   - Monitor memory usage during processing
   - Check for memory leaks in long-running sessions
   - Validate model size compatibility

### Debug Mode

Enable debug output by setting logging level:

```cpp
// In main.cpp
std::cout.setf(std::ios::boolalpha);  // Enable verbose output
```

## Contributing

### Code Style

- Follow C++17 standards
- Use RAII principles
- Prefer smart pointers over raw pointers
- Thread-safe by default
- Comprehensive error handling

### Testing Requirements

- Unit tests for all components
- Integration tests for full workflows
- Performance benchmarks
- Memory leak detection

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Support

For issues and questions:

1. Check troubleshooting section
2. Review error logs
3. Validate model files and paths
4. Test with provided examples

## Version History

- **v1.0**: Basic session management with SVM classification
- **v2.0**: Added NER extraction and multithreading
- **v3.0**: Advanced multithreaded processing with LLM integration
- **v3.1**: Performance optimizations and enhanced error handling