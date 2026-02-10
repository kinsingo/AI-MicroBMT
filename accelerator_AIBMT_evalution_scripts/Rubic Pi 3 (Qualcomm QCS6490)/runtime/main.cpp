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
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;
using namespace cv;
using namespace Ort;

class ImageClassification_Interface_Implementation : public AI_BMT_Interface
{
private:
    Ort::Env env;
    Ort::RunOptions runOptions;
    std::shared_ptr<Ort::Session> session;

    std::string inputNameStr;
    std::string outputNameStr;
    std::array<const char *, 1> inputNames{};
    std::array<const char *, 1> outputNames{};

    Ort::MemoryInfo memory_info;
    bool isUseQNNEP;
    bool isUseOpenMP;
    const std::array<int64_t, 4> inputShape = {1, 3, 224, 224};
    const std::array<int64_t, 2> outputShape = {1, 1000};
  

public:
    explicit ImageClassification_Interface_Implementation(bool useQNNEP, bool useOpenMP)
        : env(ORT_LOGGING_LEVEL_WARNING, "AI_BMT_Rubik"),
          memory_info(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU)),
          isUseQNNEP(useQNNEP),
          isUseOpenMP(useOpenMP)
    {
        cout << "useQNNEP:" << isUseQNNEP << endl;
        cout << "useOpenMP:" << isUseOpenMP << endl;
        std::vector<std::string> providers = Ort::GetAvailableProviders();
        std::cout << "Available Execution Providers:\n";
        for (const auto &p : providers)
            std::cout << " - " << p << "\n";
    }

    virtual InterfaceType getInterfaceType() override
    {
        return InterfaceType::ImageClassification;
    }

    virtual void initialize(std::string modelPath) override
    {
        Ort::SessionOptions sessionOptions;
        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        if (isUseQNNEP)
        {
            auto providers = Ort::GetAvailableProviders();
            bool hasQNN = std::find(providers.begin(), providers.end(),
                                    "QNNExecutionProvider") != providers.end();

            if (!hasQNN)
            {
                throw runtime_error("[WARN] QNNExecutionProvider not found. Check you linked onnxruntime_qnn build.");
            }
            else
            {
                std::unordered_map<std::string, std::string> qnn_opts;
                qnn_opts["backend_type"] = "htp"; // Rubik 문서 옵션 
                // qnn_opts["profiling_level"] = "detailed"; // 필요 시

                // 처음엔 fallback 허용 권장 (0 : 허용, 1 : 비허용), 안정화 후 1로 바꾸기
                sessionOptions.AddConfigEntry("session.disable_cpu_ep_fallback", "0");

                sessionOptions.AppendExecutionProvider("QNN", qnn_opts); // C++ 예시 패턴 :contentReference[oaicite:11]{index=11}
                std::cout << "[INFO] QNN EP appended (backend_type=htp)\n";
            }
        }

        session = std::make_shared<Ort::Session>(env, modelPath.c_str(), sessionOptions);

        Ort::AllocatorWithDefaultOptions allocator;
        auto in = session->GetInputNameAllocated(0, allocator);
        auto out = session->GetOutputNameAllocated(0, allocator);

        inputNameStr = in.get();
        outputNameStr = out.get();
        inputNames = {inputNameStr.c_str()};
        outputNames = {outputNameStr.c_str()};
    }

    virtual Optional_Data getOptionalData() override
    {
        Optional_Data data;
        data.cpu_type = "Qualcomm Kryo 670 (Rubik Pi3)";
        data.accelerator_type = isUseQNNEP ? "Qualcomm QCS6490" : "Qualcomm CPU";
        data.submitter = isUseOpenMP ? "useOpenMP" : "";
        data.cpu_core_count = "8";
        data.cpu_ram_capacity = "8GB";
        data.cooling = "Passive";
        data.cooling_option = "Passive";
        data.cpu_accelerator_interconnect_interface = isUseQNNEP ? "On-chip / Shared Memory" : "";
        data.benchmark_model = "";
        data.operating_system = "Ubuntu 24.04";

        return data;
    }

    virtual VariantType preprocessVisionData(const string &imagePath) override
    {
        Mat image = imread(imagePath);
        if (image.empty())
        {
            throw runtime_error("Failed to load image: " + imagePath);
        }

        // convert BGR to RGB
        cvtColor(image, image, cv::COLOR_BGR2RGB);

        // float [0,1]
        image.convertTo(image, CV_32F, 1.0f / 255.0f);

        // ImageNet mean/std
        const vector<float> means = {0.485f, 0.456f, 0.406f};
        const vector<float> stds = {0.229f, 0.224f, 0.225f};

        vector<Mat> channels;
        split(image, channels);

        vector<float> output;
        output.reserve(3 * 224 * 224);

        for (int c = 0; c < 3; ++c)
        {
            channels[c] = (channels[c] - means[c]) / stds[c];
            float *data = reinterpret_cast<float *>(channels[c].data);
            for (int i = 0; i < 224 * 224; ++i)
            {
                output.push_back(data[i]);
            }
        }

        return output;
    }

    virtual vector<BMTVisionResult> inferVision(const vector<VariantType> &data) override
    {
        vector<BMTVisionResult> results(data.size());

        // OpenMP 병렬화 활성화
        // 주의: QNN EP(NPU) 사용 시 병렬 추론이 큐잉되어 순차 처리될 수 있으며,
        // 오버헤드로 인해 성능 향상이 제한적일 수 있음. 실험을 통해 최적값 확인 필요.
        // 파이프라이닝이나 전처리/후처리 병렬화로 일부 이득을 볼 수 있음.
        
        #pragma omp parallel for if(isUseOpenMP)
        for (size_t i = 0; i < data.size(); ++i)
        {
            const vector<float> &imageVec = get<vector<float>>(data[i]);

            vector<float> outputData(1000);

            auto inputTensor = Ort::Value::CreateTensor<float>(
                memory_info,
                const_cast<float *>(imageVec.data()),
                imageVec.size(),
                inputShape.data(),
                inputShape.size());

            auto outputTensor = Ort::Value::CreateTensor<float>(
                memory_info,
                outputData.data(),
                outputData.size(),
                outputShape.data(),
                outputShape.size());

            // 각 스레드마다 독립적인 RunOptions 사용 (thread-safety 보장)
            Ort::RunOptions threadRunOptions;
            session->Run(threadRunOptions,
                         inputNames.data(), &inputTensor, 1,
                         outputNames.data(), &outputTensor, 1);

            BMTVisionResult result;
            result.classProbabilities = std::move(outputData);
            results[i] = std::move(result);
        }

        return results;
    }
};

class ImageClassification_Interface_Implementation_InputResSweep : public AI_BMT_Interface
{
private:
    Ort::Env env;
    Ort::RunOptions runOptions;
    std::shared_ptr<Ort::Session> session;

    std::string inputNameStr;
    std::string outputNameStr;
    std::array<const char *, 1> inputNames{};
    std::array<const char *, 1> outputNames{};

    Ort::MemoryInfo memory_info;
    bool isUseQNNEP;
    bool isUseOpenMP;
    int inputResolution;
    std::array<int64_t, 4> inputShape;
    const std::array<int64_t, 2> outputShape = {1, 1000};

public:
    explicit ImageClassification_Interface_Implementation_InputResSweep(bool useQNNEP, bool useOpenMP, int inputResolution)
        : env(ORT_LOGGING_LEVEL_WARNING, "AI_BMT_Rubik"),
          memory_info(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU)),
          isUseQNNEP(useQNNEP), isUseOpenMP(useOpenMP), inputResolution(inputResolution)
    {
        cout << "useQNNEP:" << useQNNEP << endl;
        cout << "useOpenMP:" << isUseOpenMP << endl;
        inputShape = {1, 3, inputResolution, inputResolution};
        cout << "Input Res : " << this->inputResolution << "x" << this->inputResolution << endl;
        std::vector<std::string> providers = Ort::GetAvailableProviders();
        std::cout << "Available Execution Providers:\n";
        for (const auto &p : providers)
            std::cout << " - " << p << "\n";
    }

    virtual InterfaceType getInterfaceType() override
    {
        return InterfaceType::ImageClassification;
    }

    virtual void initialize(std::string modelPath) override
    {
        cout << "[Initialize Model] Input Res : " << this->inputResolution << "x" << this->inputResolution << endl;
        Ort::SessionOptions sessionOptions;
        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        if (isUseQNNEP)
        {
            auto providers = Ort::GetAvailableProviders();
            bool hasQNN = std::find(providers.begin(), providers.end(),
                                    "QNNExecutionProvider") != providers.end();

            if (!hasQNN)
            {
                throw runtime_error("[WARN] QNNExecutionProvider not found. Check you linked onnxruntime_qnn build.");
            }
            else
            {
                std::unordered_map<std::string, std::string> qnn_opts;
                qnn_opts["backend_type"] = "htp"; // Rubik 문서 옵션 :contentReference[oaicite:9]{index=9}
                // qnn_opts["profiling_level"] = "detailed"; // 필요 시 :contentReference[oaicite:10]{index=10}

                // 처음엔 fallback 허용 권장 (0 : 허용, 1 : 비허용), 안정화 후 1로 바꾸기
                sessionOptions.AddConfigEntry("session.disable_cpu_ep_fallback", "0");

                sessionOptions.AppendExecutionProvider("QNN", qnn_opts); // C++ 예시 패턴 :contentReference[oaicite:11]{index=11}
                std::cout << "[INFO] QNN EP appended (backend_type=htp)\n";
            }
        }

        session = std::make_shared<Ort::Session>(env, modelPath.c_str(), sessionOptions);

        Ort::AllocatorWithDefaultOptions allocator;
        auto in = session->GetInputNameAllocated(0, allocator);
        auto out = session->GetOutputNameAllocated(0, allocator);

        inputNameStr = in.get();
        outputNameStr = out.get();
        inputNames = {inputNameStr.c_str()};
        outputNames = {outputNameStr.c_str()};
    }

    virtual Optional_Data getOptionalData() override
    {
        Optional_Data data;
        data.cpu_type = "Qualcomm Kryo 670 (Rubik Pi3)";
        data.accelerator_type = isUseQNNEP ? "Qualcomm QCS6490" : "Qualcomm CPU";
        data.submitter = isUseOpenMP ? "useOpenMP" : "";
        data.cpu_core_count = "8";
        data.cpu_ram_capacity = "8GB";
        data.cooling = "inputResolution:" + to_string(inputResolution);
        data.cpu_accelerator_interconnect_interface = isUseQNNEP ? "On-chip / Shared Memory" : "";
        data.benchmark_model = "";
        data.operating_system = "Ubuntu 24.04";

        return data;
    }

    virtual VariantType preprocessVisionData(const string &imagePath) override
    {
        Mat image = imread(imagePath);
        cv::resize(image, image, cv::Size(inputResolution, inputResolution));

        // convert BGR to RGB
        cvtColor(image, image, cv::COLOR_BGR2RGB);

        // float [0,1]
        image.convertTo(image, CV_32F, 1.0f / 255.0f);

        // ImageNet mean/std
        const vector<float> means = {0.485f, 0.456f, 0.406f};
        const vector<float> stds = {0.229f, 0.224f, 0.225f};

        vector<Mat> channels;
        split(image, channels);

        vector<float> output;
        output.reserve(3 * inputResolution * inputResolution);

        for (int c = 0; c < 3; ++c)
        {
            channels[c] = (channels[c] - means[c]) / stds[c];
            float *data = reinterpret_cast<float *>(channels[c].data);
            for (int i = 0; i < inputResolution * inputResolution; ++i)
            {
                output.push_back(data[i]);
            }
        }

        return output;
    }

    virtual vector<BMTVisionResult> inferVision(const vector<VariantType> &data) override
    {
        vector<BMTVisionResult> results(data.size());

        // OpenMP 병렬화 활성화
        // 주의: QNN EP(NPU) 사용 시 병렬 추론이 큐잉되어 순차 처리될 수 있으며,
        // 오버헤드로 인해 성능 향상이 제한적일 수 있음. 실험을 통해 최적값 확인 필요.
        // 파이프라이닝이나 전처리/후처리 병렬화로 일부 이득을 볼 수 있음.
        
        #pragma omp parallel for if(isUseOpenMP)
        for (size_t i = 0; i < data.size(); ++i)
        {
            const vector<float> &imageVec = get<vector<float>>(data[i]);

            vector<float> outputData(1000);

            auto inputTensor = Ort::Value::CreateTensor<float>(
                memory_info,
                const_cast<float *>(imageVec.data()),
                imageVec.size(),
                inputShape.data(),
                inputShape.size());

            auto outputTensor = Ort::Value::CreateTensor<float>(
                memory_info,
                outputData.data(),
                outputData.size(),
                outputShape.data(),
                outputShape.size());

            // 각 스레드마다 독립적인 RunOptions 사용 (thread-safety 보장)
            Ort::RunOptions threadRunOptions;
            session->Run(threadRunOptions,
                         inputNames.data(), &inputTensor, 1,
                         outputNames.data(), &outputTensor, 1);

            BMTVisionResult result;
            result.classProbabilities = std::move(outputData);
            results[i] = std::move(result);
        }

        return results;
    }
};

int main(int argc, char *argv[])
{
    #ifdef _OPENMP
    cout << "OpenMP is supported. Max threads: " << omp_get_max_threads() << endl;
    #else
    cout << "OpenMP is not supported." << endl;
    #endif
    /*
    aibmtExample@gmail.com
    !jonghyun04

    rm -rf CMakeCache.txt CMakeFiles .ninja* build.ninja rules.ninja \
    cmake_install.cmake compile_commands.json qtcsettings.cmake .qtc AI_BMT_GUI_Submitter
    cmake -G "Ninja" ..
    export LD_LIBRARY_PATH=$(pwd)/lib
    cmake --build .
    ./AI_BMT_GUI_Submitter
    */

    // Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    // std::vector<std::string> providers = Ort::GetAvailableProviders();
    // std::cout << "Available Execution Providers:\n";
    // for (const auto &p : providers)
    //     std::cout << " - " << p << "\n";

    try
    {
        bool useQNNEP = true;
        bool useOpenMP = true;
        // shared_ptr<AI_BMT_Interface> interface =
        //     make_shared<ImageClassification_Interface_Implementation>(useQNNEP, useOpenMP);

        int inputRes = 448;
        shared_ptr<AI_BMT_Interface> interface =
            make_shared<ImageClassification_Interface_Implementation_InputResSweep>(useQNNEP,useOpenMP, inputRes);
        cout << "Starting Image Classification BMT with "
             << (useQNNEP ? "QNNExecutionProvider (NPU)" : "CPU")
             << " backend" << endl;

        return AI_BMT_GUI_CALLER::call_BMT_GUI_For_Single_Task(argc, argv, interface);
    }
    catch (const exception &ex)
    {
        cout << ex.what() << endl;
    }
}
