#include "ai_bmt_gui_caller.h"
#include "ai_bmt_interface.h"
#include <thread>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <filesystem>

using namespace std;
using namespace cv;
using namespace Ort;

// class ImageClassification_Interface_Implementation : public AI_BMT_Interface
// {
// private:
//     Env env;
//     shared_ptr<Session> session;
//     array<const char*, 1> inputNames;
//     array<const char*, 1> outputNames;
//     MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
//     RunOptions runOptions;
    
//     // Pre-allocated buffers for single-stream optimization
//     static constexpr int64_t imageSize = 3 * 224 * 224;
//     vector<float> singleInputBuffer;
//     vector<float> singleOutputBuffer;
    
//     // Pre-allocated buffers for batch processing optimization
//     static constexpr int64_t maxBatchSize = 32; //성능 : 512 < 256 < 128 < 64 = 32 (어떤건 64, 어떤건 32가 더 나음, 대체로 32가 안정적)
//     vector<float> batchInputBuffer;
//     vector<float> batchOutputBuffer;
//     vector<int> validIndices;

// public:
//     virtual InterfaceType getInterfaceType() override
//     {
// 		return InterfaceType::ImageClassification;
//     }


//     virtual void initialize(string modelPath) override
//     {
//         //session initializer
//         SessionOptions sessionOptions;
//         sessionOptions.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
//         sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        
//         // CUDA provider with device_id = 1
//         OrtCUDAProviderOptions cuda_options;
//         cuda_options.device_id = 0;//0 for single stream, 1 for offline
//         cuda_options.arena_extend_strategy = 0;
//         cuda_options.gpu_mem_limit = SIZE_MAX;
//         cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
//         cuda_options.do_copy_in_default_stream = 1;
//         sessionOptions.AppendExecutionProvider_CUDA(cuda_options);
        
//         session = make_shared<Session>(env, modelPath.c_str(), sessionOptions);

//         // Get input and output names
//         AllocatorWithDefaultOptions allocator;
//         AllocatedStringPtr inputName = session->GetInputNameAllocated(0, allocator);
//         AllocatedStringPtr outputName = session->GetOutputNameAllocated(0, allocator);
//         inputNames = { inputName.get() };
//         outputNames = { outputName.get() };
//         inputName.release();
//         outputName.release();
        
//         // Pre-allocate buffers for single-stream optimization
//         singleInputBuffer.resize(imageSize);
//         singleOutputBuffer.resize(1000);
        
//         // Pre-allocate buffers for batch processing optimization
//         batchInputBuffer.resize(maxBatchSize * imageSize);
//         batchOutputBuffer.resize(maxBatchSize * 1000);
//         validIndices.reserve(maxBatchSize);
//     }

//     virtual Optional_Data getOptionalData() override
//     {
//         Optional_Data data;
//         data.cpu_type = "INTEL(R) XEON(R) GOLD 6544Y"; // e.g., Intel i7-9750HF
//         data.accelerator_type = "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition"; // e.g., DeepX M1(NPU)
//         data.submitter = "CUDA 13.0 + cuDNN 9.1.0 + ONNX Runtime 1.19.2 + maxBatchSize (32) + RAMChunk (996)";
//         data.cpu_core_count = "64"; // e.g., 16
//         data.cpu_ram_capacity = "188GB"; // e.g., 32GB
//         data.operating_system = "Ubuntu 22.04.5 LTS"; 
//         return data;
//     }

//     virtual VariantType preprocessVisionData(const string& imagePath) override
//     {
//         Mat image = imread(imagePath);
//         if (image.empty()) {
//             throw runtime_error("Failed to load image: " + imagePath);
//         }

//         // convert BGR to RGB before reshaping
//         cvtColor(image, image, cv::COLOR_BGR2RGB);

//         // reshape (3D -> 1D)
//         image = image.reshape(1, 1);

//         // uint_8, [0, 255] -> float, [0 and 1] => Normalize number to between 0 and 1, Convert to vector<float> from cv::Mat.
//         vector<float> vec;
//         image.convertTo(vec, CV_32FC1, 1. / 255);

//         // Mean and Std deviation values
//         const vector<float> means = { 0.485, 0.456, 0.406 };
//         const vector<float> stds = { 0.229, 0.224, 0.225 };

//         // Transpose (Height, Width, Channel)(224,224,3) to (Chanel, Height, Width)(3,224,224)
//         vector<float> output;
//         for (size_t ch = 0; ch < 3; ++ch)
//         {
//             for (size_t i = ch; i < vec.size(); i += 3)
//             {
//                 float normalized = (vec[i] - means[ch]) / stds[ch];
//                 output.emplace_back(normalized);
//             }
//         }
//         return output;
//     }

//     virtual vector<BMTVisionResult> inferVision(const vector<VariantType>& data) override
//     {
//         const int querySize = data.size();
//         vector<BMTVisionResult> results;
//         results.reserve(querySize);

//         // Fast path for single-stream (most common in BMT)
//         if (querySize == 1) {
//             try {
//                 const vector<float>& imageVec = get<vector<float>>(data[0]);
//                 if (imageVec.size() != imageSize) {
//                     cerr << "Error: Invalid image size" << endl;
//                     return results;
//                 }
                
//                 // Use pre-allocated buffer
//                 copy(imageVec.begin(), imageVec.end(), singleInputBuffer.begin());
                
//                 const array<int64_t, 4> inputShape = { 1, 3, 224, 224 };
//                 const array<int64_t, 2> outputShape = { 1, 1000 };
                
//                 auto inputTensor = Ort::Value::CreateTensor<float>(memory_info, singleInputBuffer.data(), imageSize, inputShape.data(), inputShape.size());
//                 auto outputTensor = Ort::Value::CreateTensor<float>(memory_info, singleOutputBuffer.data(), 1000, outputShape.data(), outputShape.size());
                
//                 session->Run(runOptions, inputNames.data(), &inputTensor, 1, outputNames.data(), &outputTensor, 1);
                
//                 BMTVisionResult result;
//                 result.classProbabilities = singleOutputBuffer;
//                 results.push_back(result);
//                 return results;
//             }
//             catch (const std::bad_variant_access& e) {
//                 cerr << "Error: bad_variant_access. Reason: " << e.what() << endl;
//                 return results;
//             }
//         }

//         // Batch processing with pre-allocated buffers
//         for (int startIdx = 0; startIdx < querySize; startIdx += maxBatchSize) {
//             const int64_t currentBatchSize = min((int64_t)(querySize - startIdx), maxBatchSize);
            
//             validIndices.clear();  // Reuse pre-allocated vector
            
//             // Collect valid samples for this batch (reuse pre-allocated buffer)
//             for (int i = 0; i < currentBatchSize; ++i) {
//                 try {
//                     const vector<float>& imageVec = get<vector<float>>(data[startIdx + i]);
//                     if (imageVec.size() != imageSize) {
//                         cerr << "Error: Invalid image size at index " << (startIdx + i) << endl;
//                         continue;
//                     }
//                     copy(imageVec.begin(), imageVec.end(), batchInputBuffer.begin() + validIndices.size() * imageSize);
//                     validIndices.push_back(startIdx + i);
//                 }
//                 catch (const std::bad_variant_access& e) {
//                     cerr << "Error: bad_variant_access at index " << (startIdx + i) << ". Reason: " << e.what() << endl;
//                 }
//             }
            
//             if (validIndices.empty()) {
//                 continue;
//             }
            
//             const int64_t validCount = validIndices.size();
//             const array<int64_t, 4> inputShape = { validCount, 3, 224, 224 };
//             const array<int64_t, 2> outputShape = { validCount, 1000 };
            
//             auto inputTensor = Ort::Value::CreateTensor<float>(memory_info, batchInputBuffer.data(), validCount * imageSize, inputShape.data(), inputShape.size());
//             auto outputTensor = Ort::Value::CreateTensor<float>(memory_info, batchOutputBuffer.data(), validCount * 1000, outputShape.data(), outputShape.size());
            
//             session->Run(runOptions, inputNames.data(), &inputTensor, 1, outputNames.data(), &outputTensor, 1);
            
//             // Parse results from pre-allocated buffer (minimize copying)
//             for (int i = 0; i < validCount; ++i) {
//                 BMTVisionResult result;
//                 result.classProbabilities.assign(
//                     batchOutputBuffer.begin() + i * 1000,
//                     batchOutputBuffer.begin() + (i + 1) * 1000
//                 );
//                 results.push_back(move(result));  // Use move semantics
//             }
//         }

//         return results;
//     }
// };

class ImageClassification_Interface_Implementation_InputResolution : public AI_BMT_Interface
{
private:
    Env env;
    shared_ptr<Session> session;
    array<const char*, 1> inputNames;
    array<const char*, 1> outputNames;
    MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    RunOptions runOptions;
    
    // Dynamic image size based on input resolution
    int inputResolution;
    int64_t imageSize;
    
    // Pre-allocated buffers for single-stream optimization
    vector<float> singleInputBuffer;
    vector<float> singleOutputBuffer;
    
    // Pre-allocated buffers for batch processing optimization
    static constexpr int64_t maxBatchSize = 32; //성능 : 512 < 256 < 128 < 64 = 32 (어떤건 64, 어떤건 32가 더 나음, 대체로 32가 안정적)
    vector<float> batchInputBuffer;
    vector<float> batchOutputBuffer;
    vector<int> validIndices;

public:
    ImageClassification_Interface_Implementation_InputResolution(int inputResolution)
    : inputResolution(inputResolution), imageSize(3 * inputResolution * inputResolution)
    {
        cout << "inputResolution : " << this->inputResolution << "x" << this->inputResolution << endl; 

    }

    virtual InterfaceType getInterfaceType() override
    {
		return InterfaceType::ImageClassification;
    }


    virtual void initialize(string modelPath) override
    {
        //session initializer
        SessionOptions sessionOptions;
        sessionOptions.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        
        // CUDA provider with device_id = 1
        OrtCUDAProviderOptions cuda_options;
        cuda_options.device_id = 0;//0 for single stream, 1 for offline
        cuda_options.arena_extend_strategy = 0;
        cuda_options.gpu_mem_limit = SIZE_MAX;
        cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
        cuda_options.do_copy_in_default_stream = 1;
        sessionOptions.AppendExecutionProvider_CUDA(cuda_options);
        
        session = make_shared<Session>(env, modelPath.c_str(), sessionOptions);

        // Get input and output names
        AllocatorWithDefaultOptions allocator;
        AllocatedStringPtr inputName = session->GetInputNameAllocated(0, allocator);
        AllocatedStringPtr outputName = session->GetOutputNameAllocated(0, allocator);
        inputNames = { inputName.get() };
        outputNames = { outputName.get() };
        inputName.release();
        outputName.release();
        
        // Pre-allocate buffers for single-stream optimization
        singleInputBuffer.resize(imageSize);
        singleOutputBuffer.resize(1000);
        
        // Pre-allocate buffers for batch processing optimization
        batchInputBuffer.resize(maxBatchSize * imageSize);
        batchOutputBuffer.resize(maxBatchSize * 1000);
        validIndices.reserve(maxBatchSize);
    }

    virtual Optional_Data getOptionalData() override
    {
        Optional_Data data;
        data.cpu_type = "INTEL(R) XEON(R) GOLD 6544Y"; // e.g., Intel i7-9750HF
        data.accelerator_type = "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition"; // e.g., DeepX M1(NPU)
        data.submitter = "CUDA 13.0 + cuDNN 9.1.0 + ONNX Runtime 1.19.2 + maxBatchSize (32) + RAMChunk (996) + inputResolution (" + to_string(inputResolution) + ")";
        data.cpu_core_count = "64"; // e.g., 16
        data.cpu_ram_capacity = "188GB"; // e.g., 32GB
        data.operating_system = "Ubuntu 22.04.5 LTS"; 
        return data;
    }

    virtual VariantType preprocessVisionData(const string& imagePath) override
    {
        Mat image = imread(imagePath);
        if (image.empty()) {
            throw runtime_error("Failed to load image: " + imagePath);
        }

        // convert BGR to RGB
        cvtColor(image, image, cv::COLOR_BGR2RGB);
        
        // resize to input resolution
        resize(image, image, cv::Size(inputResolution, inputResolution));

        // reshape (3D -> 1D)
        image = image.reshape(1, 1);

        // uint_8, [0, 255] -> float, [0 and 1] => Normalize number to between 0 and 1, Convert to vector<float> from cv::Mat.
        vector<float> vec;
        image.convertTo(vec, CV_32FC1, 1. / 255);

        // Mean and Std deviation values
        const vector<float> means = { 0.485, 0.456, 0.406 };
        const vector<float> stds = { 0.229, 0.224, 0.225 };

        // Transpose (Height, Width, Channel)(224,224,3) to (Chanel, Height, Width)(3,224,224)
        vector<float> output;
        for (size_t ch = 0; ch < 3; ++ch)
        {
            for (size_t i = ch; i < vec.size(); i += 3)
            {
                float normalized = (vec[i] - means[ch]) / stds[ch];
                output.emplace_back(normalized);
            }
        }
        return output;
    }

    virtual vector<BMTVisionResult> inferVision(const vector<VariantType>& data) override
    {
        const int querySize = data.size();
        vector<BMTVisionResult> results;
        results.reserve(querySize);

        // Fast path for single-stream (most common in BMT)
        if (querySize == 1) {
            try {
                const vector<float>& imageVec = get<vector<float>>(data[0]);
                if (imageVec.size() != imageSize) {
                    cerr << "Error: Invalid image size" << endl;
                    return results;
                }
                
                // Use pre-allocated buffer
                copy(imageVec.begin(), imageVec.end(), singleInputBuffer.begin());
                
                const array<int64_t, 4> inputShape = { 1, 3, inputResolution, inputResolution };
                const array<int64_t, 2> outputShape = { 1, 1000 };
                
                auto inputTensor = Ort::Value::CreateTensor<float>(memory_info, singleInputBuffer.data(), imageSize, inputShape.data(), inputShape.size());
                auto outputTensor = Ort::Value::CreateTensor<float>(memory_info, singleOutputBuffer.data(), 1000, outputShape.data(), outputShape.size());
                
                session->Run(runOptions, inputNames.data(), &inputTensor, 1, outputNames.data(), &outputTensor, 1);
                
                BMTVisionResult result;
                result.classProbabilities = singleOutputBuffer;
                results.push_back(result);
                return results;
            }
            catch (const std::bad_variant_access& e) {
                cerr << "Error: bad_variant_access. Reason: " << e.what() << endl;
                return results;
            }
        }

        // Batch processing with pre-allocated buffers
        for (int startIdx = 0; startIdx < querySize; startIdx += maxBatchSize) {
            const int64_t currentBatchSize = min((int64_t)(querySize - startIdx), maxBatchSize);
            
            validIndices.clear();  // Reuse pre-allocated vector
            
            // Collect valid samples for this batch (reuse pre-allocated buffer)
            for (int i = 0; i < currentBatchSize; ++i) {
                try {
                    const vector<float>& imageVec = get<vector<float>>(data[startIdx + i]);
                    if (imageVec.size() != imageSize) {
                        cerr << "Error: Invalid image size at index " << (startIdx + i) << endl;
                        continue;
                    }
                    copy(imageVec.begin(), imageVec.end(), batchInputBuffer.begin() + validIndices.size() * imageSize);
                    validIndices.push_back(startIdx + i);
                }
                catch (const std::bad_variant_access& e) {
                    cerr << "Error: bad_variant_access at index " << (startIdx + i) << ". Reason: " << e.what() << endl;
                }
            }
            
            if (validIndices.empty()) {
                continue;
            }
            
            const int64_t validCount = validIndices.size();
            const array<int64_t, 4> inputShape = { validCount, 3, inputResolution, inputResolution };
            const array<int64_t, 2> outputShape = { validCount, 1000 };
            
            auto inputTensor = Ort::Value::CreateTensor<float>(memory_info, batchInputBuffer.data(), validCount * imageSize, inputShape.data(), inputShape.size());
            auto outputTensor = Ort::Value::CreateTensor<float>(memory_info, batchOutputBuffer.data(), validCount * 1000, outputShape.data(), outputShape.size());
            
            session->Run(runOptions, inputNames.data(), &inputTensor, 1, outputNames.data(), &outputTensor, 1);
            
            // Parse results from pre-allocated buffer (minimize copying)
            for (int i = 0; i < validCount; ++i) {
                BMTVisionResult result;
                result.classProbabilities.assign(
                    batchOutputBuffer.begin() + i * 1000,
                    batchOutputBuffer.begin() + (i + 1) * 1000
                );
                results.push_back(move(result));  // Use move semantics
            }
        }

        return results;
    }
};

int main(int argc, char* argv[])
{
/*
[Single-Stream]
aibmtExample@gmail.com

rm -rf CMakeCache.txt CMakeFiles .ninja* build.ninja rules.ninja \
cmake_install.cmake compile_commands.json qtcsettings.cmake .qtc AI_BMT_GUI_Submitter
cmake -G "Ninja" ..
export LD_LIBRARY_PATH=$(pwd)/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib
cmake --build .
./AI_BMT_GUI_Submitter

export LD_LIBRARY_PATH=$(pwd)/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib
./AI_BMT_GUI_Submitter
*/

    try
    {
        // Check available providers
        cout << "=== ONNX Runtime Available Providers ===" << endl;
        vector<string> availableProviders = Ort::GetAvailableProviders();
        bool cudaAvailable = false;
        for (const auto& provider : availableProviders) {
            cout << "  - " << provider << endl;
            if (provider == "CUDAExecutionProvider") {
                cudaAvailable = true;
            }
        }
        cout << "=========================================" << endl;
        
        if (cudaAvailable) {
            cout << "✓ CUDA Provider is available!" << endl;
        } else {
            cout << "✗ CUDA Provider is NOT available!" << endl;
            cerr << "Warning: CUDA provider not found. Will fall back to CPU." << endl;
        }
        cout << endl;
        
        //shared_ptr<AI_BMT_Interface> interface = make_shared<ImageClassification_Interface_Implementation>(); 
        int inputResolution = 448; // or other resolution as needed
        shared_ptr<AI_BMT_Interface> interface = make_shared<ImageClassification_Interface_Implementation_InputResolution>(inputResolution);
        return AI_BMT_GUI_CALLER::call_BMT_GUI_For_Single_Task(argc, argv, interface);
    }
    catch (const exception& ex)
    {
        cout << ex.what() << endl;
    }
}