#include "ai_bmt_gui_caller.h"
#include "ai_bmt_interface.h"
#include <thread>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <filesystem>
#include <stdexcept>
#include "qbruntime/qbruntime.h"
#include <opencv2/opencv.hpp>

using namespace mobilint;
using namespace std;
using namespace cv;

enum class ExecuteMode
{
    Single,
    Global,
    Multi,
};

class Classification_Implementation : public AI_BMT_Interface
{
private:
    ExecuteMode executeMode_;
    StatusCode sc{};
    ModelConfig mc{};
    std::unique_ptr<Accelerator> acc;
    std::unique_ptr<Model> model;
    bool session_initialized = false;

    // Multi 모드에서 한 번에 몇 개 입력을 병렬로 넣을지(선택한 클러스터 수와 동일하게 설정)
    const size_t maxMultiThreads = 64; //FPS 4->3477, 8->5682, 12->5944, 16->6876, 32->7973, 64->8342

public:
    explicit Classification_Implementation(ExecuteMode mode = ExecuteMode::Single)
        : executeMode_(mode)
    {
        acc = Accelerator::create(sc);

        if (executeMode_ == ExecuteMode::Global)
        {
            // 단일 입력의 처리 성능 향상을 위해 모든 코어를 하나의 글로벌 파이프로 사용
            const vector<Cluster> clusters = {Cluster::Cluster0, Cluster::Cluster1};
            mc.setGlobalCoreMode(clusters);
        }
        else if (executeMode_ == ExecuteMode::Multi)
        {
            mc.setSingleCoreMode({{Cluster::Cluster0, Core::Core0}, 
                {Cluster::Cluster0, Core::Core1}, 
                {Cluster::Cluster0, Core::Core2}, 
                {Cluster::Cluster0, Core::Core3}, 
                {Cluster::Cluster1, Core::Core0},
                {Cluster::Cluster1, Core::Core1},
                {Cluster::Cluster1, Core::Core2}, 
                {Cluster::Cluster1, Core::Core3}});
        }
        else
        { // ExecuteMode::Single
            mc.setSingleCoreMode({{Cluster::Cluster0, Core::Core0}});
        }
    }

    virtual InterfaceType getInterfaceType() override
    {
        return InterfaceType::ImageClassification;
    }

    void initialize(std::string modelPath) override
    {
        if(executeMode_ == ExecuteMode::Multi   )
            cout<<"(Multi Mode) maxMultiThreads:"<<maxMultiThreads<<endl;
        if(executeMode_ == ExecuteMode::Single)
            cout<<"(Single Mode)"<<endl;
        try
        {
            if (!std::filesystem::exists(modelPath))
            {
                throw std::runtime_error("Model file not found: " + modelPath);
            }
            if (session_initialized && model)
            {
                model->dispose();
            }
            model = Model::create(modelPath, mc, sc);
            model->launch(*acc);
            session_initialized = true;
        }
        catch (const std::exception &ex)
        {
            std::cerr << "Failed to initialize maccel model: " << ex.what() << "\n";
            session_initialized = false;
        }
    }

    virtual Optional_Data getOptionalData() override
    {
        Optional_Data data;
        data.accelerator_type = "Mobilint-ARIES";
        switch (executeMode_)
        {
        case ExecuteMode::Global:
            data.submitter = "mobilint(cpp) global";
            break;
        case ExecuteMode::Multi:
            data.submitter = "mobilint(cpp) multi, maxMultiThreads=" + std::to_string(maxMultiThreads);
            break;
        default:
            data.submitter = "mobilint(cpp) single";
            break;
        }
        data.submitter += ", RT version=" + mobilint::getQbRuntimeVersion();
        data.operating_system = "Ubuntu 24.04.5 LTS";
        return data;
    }

    // RGB HWC interleaved, float*, size = H*W*3  (delete[] in inferVision)
    virtual VariantType preprocessVisionData(const std::string &imagePath) override
    {
        cv::Mat img = cv::imread(imagePath, cv::IMREAD_COLOR);
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

        const int H = img.rows;
        const int W = img.cols;
        const int C = img.channels(); // expect 3
        std::vector<float> buffer(static_cast<size_t>(H) * W * C);

        const float means[3] = {0.485f, 0.456f, 0.406f};
        const float stds[3] = {0.229f, 0.224f, 0.225f};

        size_t idx = 0;
        for (int y = 0; y < H; ++y)
        {
            const cv::Vec3b *row = img.ptr<cv::Vec3b>(y);
            for (int x = 0; x < W; ++x)
            {
                const cv::Vec3b &p = row[x]; // now RGB
                for (int c = 0; c < 3; ++c)
                {
                    float v = static_cast<float>(p[c]) / 255.0f;
                    buffer[idx++] = (v - means[c]) / stds[c];
                }
            }
        }

        float *dataPtr = new float[buffer.size()];
        std::memcpy(dataPtr, buffer.data(), buffer.size() * sizeof(float));
        return dataPtr;
    }

    vector<BMTVisionResult> inferVisionMultiThreads(const vector<VariantType> &data)
    {
        const size_t total = data.size();
        vector<BMTVisionResult> results(total);
        vector<thread> threads;
        mutex result_mutex;//it boost the performance (why..?)

        auto threadFunc = [&](size_t idx)
        {
            StatusCode local_sc{};  // 각 스레드마다 별도의 StatusCode 사용
            float *inputPtr = std::get<float *>(data[idx]);
            std::vector<std::vector<float>> output = model->infer({inputPtr}, local_sc);

            if (!local_sc)
            {
                cerr << "Inference failed at index " << idx << endl;
                delete[] inputPtr;
                return;
            }

            BMTVisionResult r;
            r.classProbabilities = std::move(output[0]);

            {
                lock_guard<mutex> lock(result_mutex);//it boost the performance (why..?)
                results[idx] = std::move(r);
            }

            delete[] inputPtr;
        };

        // Limit threads to 8 or total input count
        const size_t max_threads = std::min<size_t>(maxMultiThreads, total);
        size_t i = 0;
        while (i < total)
        {
            threads.clear();
            size_t batch = std::min(max_threads, total - i);

            for (size_t j = 0; j < batch; ++j)
            {
                threads.emplace_back(threadFunc, i + j);
            }
            for (auto &t : threads)
                t.join();

            i += batch;
        }

        return results;
    }

    virtual vector<BMTVisionResult> inferVision(const vector<VariantType> &data) override
    {
        vector<BMTVisionResult> results(data.size());
        if (executeMode_ == ExecuteMode::Multi)
        {
            return inferVisionMultiThreads(data);
        }
        else
        {
            for (size_t i = 0; i < data.size(); ++i)
            {
                float *inputPtr = std::get<float *>(data[i]);
                std::vector<std::vector<float>> outputs = model->infer({inputPtr}, sc);
                BMTVisionResult r;
                r.classProbabilities = std::move(outputs[0]);
                results[i] = std::move(r);
                delete[] inputPtr;
            }
        }

        return results;
    }
};

// class Classification_Implementation_InputResolution : public AI_BMT_Interface
// {
// private:
//     ExecuteMode executeMode_;
//     StatusCode sc{};
//     ModelConfig mc{};
//     std::unique_ptr<Accelerator> acc;
//     std::unique_ptr<Model> model;
//     bool session_initialized = false;

//     // Multi 모드에서 한 번에 몇 개 입력을 병렬로 넣을지(선택한 클러스터 수와 동일하게 설정)
//     const size_t maxMultiThreads = 64; //FPS 4->3477, 8->5682, 12->5944, 16->6876, 32->7973, 64->8342
//     int inputResolution;

// public:
//     explicit Classification_Implementation_InputResolution(int inputResolution, ExecuteMode mode = ExecuteMode::Single)
//         : executeMode_(mode),inputResolution(inputResolution)
//     {
//         acc = Accelerator::create(sc);
//          cout << "inputResolution : " << this->inputResolution <<"x"<< this->inputResolution << endl;

//         if (executeMode_ == ExecuteMode::Global)
//         {
//             // 단일 입력의 처리 성능 향상을 위해 모든 코어를 하나의 글로벌 파이프로 사용
//             const vector<Cluster> clusters = {Cluster::Cluster0, Cluster::Cluster1};
//             mc.setGlobalCoreMode(clusters);
//         }
//         else if (executeMode_ == ExecuteMode::Multi)
//         {
//             // 여러 입력을 동시에 처리(입력 개수 == 선택한 클러스터 수 권장)
//             // const vector<Cluster> clusters = {Cluster::Cluster0, Cluster::Cluster1};
//             // mc.setMultiCoreMode(clusters);
//             //mc.setMultiCoreMode({Cluster::Cluster0, Cluster::Cluster1});
//             mc.setSingleCoreMode({{Cluster::Cluster0, Core::Core0}, {Cluster::Cluster0, Core::Core1}, {Cluster::Cluster0, Core::Core2}, {Cluster::Cluster0, Core::Core3}, {Cluster::Cluster1, Core::Core0}, {Cluster::Cluster1, Core::Core1}, {Cluster::Cluster1, Core::Core2}, {Cluster::Cluster1, Core::Core3}});
//         }
//         else
//         { // ExecuteMode::Single
//             mc.setSingleCoreMode({{Cluster::Cluster0, Core::Core0}});
//         }
//     }

//     virtual InterfaceType getInterfaceType() override
//     {
//         return InterfaceType::ImageClassification;
//     }

//     void initialize(std::string modelPath) override
//     {
//         if(executeMode_ == ExecuteMode::Multi   )
//             cout<<"(Multi Mode) maxMultiThreads:"<<maxMultiThreads<<endl;
//         if(executeMode_ == ExecuteMode::Single)
//             cout<<"(Single Mode)"<<endl;
//         try
//         {
//             if (!std::filesystem::exists(modelPath))
//             {
//                 throw std::runtime_error("Model file not found: " + modelPath);
//             }
//             if (session_initialized && model)
//             {
//                 model->dispose();
//             }
//             model = Model::create(modelPath, mc, sc);
//             model->launch(*acc);
//             session_initialized = true;
//         }
//         catch (const std::exception &ex)
//         {
//             std::cerr << "Failed to initialize maccel model: " << ex.what() << "\n";
//             session_initialized = false;
//         }
//     }

//     virtual Optional_Data getOptionalData() override
//     {
//         Optional_Data data;
//         data.accelerator_type = "Mobilint-ARIES";
//         switch (executeMode_)
//         {
//         case ExecuteMode::Global:
//             data.submitter = "mobilint(cpp) global";
//             break;
//         case ExecuteMode::Multi:
//             data.submitter = "mobilint(cpp) multi, maxMultiThreads=" + std::to_string(maxMultiThreads);
//             break;
//         default:
//             data.submitter = "mobilint(cpp) single";
//             break;
//         }
//         data.submitter += ", RT version=" + mobilint::getQbRuntimeVersion();
//         data.cooling = "inputResolution:" + to_string(inputResolution);
//         data.operating_system = "Ubuntu 24.04.5 LTS";
//         return data;
//     }

//     // RGB HWC interleaved, float*, size = H*W*3  (delete[] in inferVision)
//     virtual VariantType preprocessVisionData(const std::string &imagePath) override
//     {
//         cv::Mat img = cv::imread(imagePath, cv::IMREAD_COLOR);
//         cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
//         cv::resize(img, img, cv::Size(inputResolution, inputResolution));

//         const int H = img.rows;
//         const int W = img.cols;
//         const int C = img.channels(); // expect 3
//         std::vector<float> buffer(static_cast<size_t>(H) * W * C);

//         const float means[3] = {0.485f, 0.456f, 0.406f};
//         const float stds[3] = {0.229f, 0.224f, 0.225f};

//         size_t idx = 0;
//         for (int y = 0; y < H; ++y)
//         {
//             const cv::Vec3b *row = img.ptr<cv::Vec3b>(y);
//             for (int x = 0; x < W; ++x)
//             {
//                 const cv::Vec3b &p = row[x]; // now RGB
//                 for (int c = 0; c < 3; ++c)
//                 {
//                     float v = static_cast<float>(p[c]) / 255.0f;
//                     buffer[idx++] = (v - means[c]) / stds[c];
//                 }
//             }
//         }

//         float *dataPtr = new float[buffer.size()];
//         std::memcpy(dataPtr, buffer.data(), buffer.size() * sizeof(float));
//         return dataPtr;
//     }

//     vector<BMTVisionResult> inferVisionMultiThreads(const vector<VariantType> &data)
//     {
//         const size_t total = data.size();
//         vector<BMTVisionResult> results(total);
//         vector<thread> threads;
//         mutex result_mutex;//it boost the performance (why..?)

//         auto threadFunc = [&](size_t idx)
//         {
//             StatusCode local_sc{};  // 각 스레드마다 별도의 StatusCode 사용
//             float *inputPtr = std::get<float *>(data[idx]);
//             std::vector<std::vector<float>> output = model->infer({inputPtr}, local_sc);

//             if (!local_sc)
//             {
//                 cerr << "Inference failed at index " << idx << endl;
//                 delete[] inputPtr;
//                 return;
//             }

//             BMTVisionResult r;
//             r.classProbabilities = std::move(output[0]);

//             {
//                 lock_guard<mutex> lock(result_mutex);//it boost the performance (why..?)
//                 results[idx] = std::move(r);
//             }

//             delete[] inputPtr;
//         };

//         // Limit threads to 8 or total input count
//         const size_t max_threads = std::min<size_t>(maxMultiThreads, total);
//         size_t i = 0;
//         while (i < total)
//         {
//             threads.clear();
//             size_t batch = std::min(max_threads, total - i);

//             for (size_t j = 0; j < batch; ++j)
//             {
//                 threads.emplace_back(threadFunc, i + j);
//             }
//             for (auto &t : threads)
//                 t.join();

//             i += batch;
//         }

//         return results;
//     }

//     virtual vector<BMTVisionResult> inferVision(const vector<VariantType> &data) override
//     {
//         vector<BMTVisionResult> results(data.size());
//         if (executeMode_ == ExecuteMode::Multi)
//         {
//             return inferVisionMultiThreads(data);
//         }
//         else
//         {
//             for (size_t i = 0; i < data.size(); ++i)
//             {
//                 float *inputPtr = std::get<float *>(data[i]);
//                 std::vector<std::vector<float>> outputs = model->infer({inputPtr}, sc);
//                 BMTVisionResult r;
//                 r.classProbabilities = std::move(outputs[0]);
//                 results[i] = std::move(r);
//                 delete[] inputPtr;
//             }
//         }

//         return results;
//     }
// };

int main(int argc, char *argv[])
{
    /*
    jonghyun_shin@naver.com

    rm -rf CMakeCache.txt CMakeFiles .ninja* build.ninja rules.ninja \
    cmake_install.cmake compile_commands.json qtcsettings.cmake .qtc AI_BMT_GUI_Submitter
    cmake -G "Ninja" ..
    export LD_LIBRARY_PATH=$(pwd)/lib
    cmake --build .
    ./AI_BMT_GUI_Submitter
    */
    try
    {
        cout <<"Multi mode (local sc) 260205"<<endl;
        //This RT version should be v1.0.0
        std::cout << "Runtime Version : " << mobilint::getQbRuntimeVersion() << "\n";
        // int inputResolution = 448;
        // auto interface = std::make_shared<Classification_Implementation_InputResolution>(inputResolution, ExecuteMode::Multi);
        auto interface = std::make_shared<Classification_Implementation>(ExecuteMode::Multi);
        return AI_BMT_GUI_CALLER::call_BMT_GUI_For_Single_Task(argc, argv, interface);
    }
    catch (const std::exception &ex)
    {
        std::cerr << ex.what() << std::endl;
        return -1;
    }
}