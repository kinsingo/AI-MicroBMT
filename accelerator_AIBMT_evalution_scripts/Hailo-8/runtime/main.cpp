#include "hailo/hailort.hpp"
#include "ai_bmt_interface.h"
#include "ai_bmt_gui_caller.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <memory>
#include <string>
#include <map>

using namespace hailort;
using namespace std;

// // class HailoBMT : public AI_BMT_Interface
// // {
// // private:
// //     std::unique_ptr<VDevice> m_vdevice;
// //     std::shared_ptr<ConfiguredNetworkGroup> m_network_group;
// //     std::unique_ptr<InferVStreams> m_pipeline;
// //     std::string m_input_name;
// //     std::string m_output_name;
// //     bool isCustomDataset;

// //     // 🔧 이전 리소스를 정리
// //     void cleanup()
// //     {
// //         // 1) VStreams 파이프라인 파괴
// //         if (m_pipeline)
// //         {
// //             m_pipeline.reset(); // dtor가 스트림 닫음
// //         }

// //         // 2) NetworkGroup 종료 (스케줄러/레벨에서 스트림/컨텍스트 닫기)
// //         if (m_network_group)
// //         {
// //             (void)m_network_group->shutdown(); // 상태 반환 무시 (이미 종료 중일 수도 있음)
// //             m_network_group.reset();
// //         }

// //         // 3) VDevice 해제 (물리 디바이스 핸들 반납)
// //         if (m_vdevice)
// //         {
// //             m_vdevice.reset();
// //         }
// //     }

// // public:
// //     HailoBMT(bool isCustomDataset)
// //     {
// //         this->isCustomDataset = isCustomDataset;
// //     }

// //     InterfaceType getInterfaceType() override
// //     {
// //         if (isCustomDataset)
// //             return InterfaceType::ImageClassification_CustomDataset;
// //         else
// //             return InterfaceType::ImageClassification;
// //     }

// //     Optional_Data getOptionalData() override
// //     {
// //         Optional_Data data;
// //         data.cpu_type = "Broadcom BCM2712 quad-core Arm Cortex A76 @ 2.4GHz";
// //         data.accelerator_type = "Hailo-8";
// //         data.submitter = "Hailo(Compiler : 3.31v, Runtime : v4.23, SingleMode)"; 
// //         data.cpu_core_count = "4";
// //         data.cpu_ram_capacity = "8GB";
// //         data.cooling = "Air";
// //         data.cooling_option = "Active";
// //         data.cpu_accelerator_interconnect_interface = "PCIe 3.0 x4";
// //         data.benchmark_model = "mobilenet_v2_opset10";
// //         data.operating_system = "Ubuntu 24.04.2 LTS";
// //         return data;
// //     }

// //     void initialize(string modelPath) override
// //     {
// //         cleanup();

// //         m_vdevice = VDevice::create().release();

// //         auto hef = Hef::create(modelPath).value();
// //         auto config_params = m_vdevice->create_configure_params(hef).value();
// //         auto network_groups = m_vdevice->configure(hef, config_params).value();
// //         m_network_group = network_groups[0];

// //         auto input_params = m_network_group->make_input_vstream_params(
// //                                                false, HAILO_FORMAT_TYPE_UINT8,
// //                                                HAILO_DEFAULT_VSTREAM_TIMEOUT_MS,
// //                                                HAILO_DEFAULT_VSTREAM_QUEUE_SIZE,
// //                                                "")
// //                                 .value();

// //         auto output_params = m_network_group->make_output_vstream_params(
// //                                                 false, HAILO_FORMAT_TYPE_FLOAT32,
// //                                                 HAILO_DEFAULT_VSTREAM_TIMEOUT_MS,
// //                                                 HAILO_DEFAULT_VSTREAM_QUEUE_SIZE,
// //                                                 "")
// //                                  .value();

// //         auto infer_expected = InferVStreams::create(*m_network_group, input_params, output_params);
// //         if (!infer_expected)
// //         {
// //             throw std::runtime_error("InferVStreams::create failed");
// //         }
// //         m_pipeline = std::make_unique<InferVStreams>(std::move(*infer_expected));

// //         m_input_name = input_params.begin()->first;
// //         m_output_name = output_params.begin()->first;
// //     }

// //     VariantType preprocessVisionData(const string &imagePath) override
// //     {
// //         cv::Mat img = cv::imread(imagePath, cv::IMREAD_COLOR);
// //         cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

// //         if (isCustomDataset)
// //         {
// //             const int target_short = 232;
// //             const int crop = 224;

// //             int h = img.rows;
// //             int w = img.cols;

// //             // 1) 짧은 변을 232로 맞추는 비율 (종횡비 유지)
// //             double scale = static_cast<double>(target_short) / std::min(h, w);
// //             int new_w = static_cast<int>(std::round(w * scale));
// //             int new_h = static_cast<int>(std::round(h * scale));

// //             // Downscale면 INTER_AREA, Upscale면 INTER_LINEAR 권장
// //             int interp = (scale < 1.0) ? cv::INTER_AREA : cv::INTER_LINEAR;

// //             cv::Mat resized;
// //             cv::resize(img, resized, cv::Size(new_w, new_h), 0, 0, interp);

// //             // 2) 중심 224x224 크롭
// //             int x = (resized.cols - crop) / 2;
// //             int y = (resized.rows - crop) / 2;
// //             cv::Rect roi(x, y, crop, crop);
// //             img = resized(roi).clone();
// //         }

// //         std::vector<uint8_t> input_buf(img.total() * img.channels());
// //         std::memcpy(input_buf.data(), img.data, input_buf.size());
// //         return input_buf;
// //     }

// //     std::vector<BMTVisionResult> inferVision(const std::vector<VariantType> &data) override
// //     {
// //         std::vector<BMTVisionResult> results;
// //         results.reserve(data.size());

// //         for (const auto &item : data)
// //         {
// //             const auto &input_buf = std::get<std::vector<uint8_t>>(item);
// //             std::map<std::string, MemoryView> input_data;
// //             std::map<std::string, MemoryView> output_data;
// //             input_data[m_input_name] = MemoryView(const_cast<uint8_t *>(input_buf.data()), input_buf.size());
// //             std::vector<float> output_buf(1000);
// //             output_data[m_output_name] = MemoryView(output_buf.data(), output_buf.size() * sizeof(float));
// //             m_pipeline->infer(input_data, output_data, /*batch_size*/ 1);

// //             BMTVisionResult r;
// //             r.classProbabilities = std::move(output_buf);
// //             results.push_back(std::move(r));
// //         }
// //         return results;
// //     }
// // };


// class HailoBMT_Input_Resolution : public AI_BMT_Interface
// {
// private:
//     std::unique_ptr<VDevice> m_vdevice;
//     std::shared_ptr<ConfiguredNetworkGroup> m_network_group;
//     std::unique_ptr<InferVStreams> m_pipeline;
//     std::string m_input_name;
//     std::string m_output_name;
//     int inputResolution;

//     // 🔧 이전 리소스를 정리
//     void cleanup()
//     {
//         // 1) VStreams 파이프라인 파괴
//         if (m_pipeline)
//         {
//             m_pipeline.reset(); // dtor가 스트림 닫음
//         }

//         // 2) NetworkGroup 종료 (스케줄러/레벨에서 스트림/컨텍스트 닫기)
//         if (m_network_group)
//         {
//             (void)m_network_group->shutdown(); // 상태 반환 무시 (이미 종료 중일 수도 있음)
//             m_network_group.reset();
//         }

//         // 3) VDevice 해제 (물리 디바이스 핸들 반납)
//         if (m_vdevice)
//         {
//             m_vdevice.reset();
//         }
//     }

// public:
//     HailoBMT_Input_Resolution(int inputResolution)
//     {
//         this->inputResolution = inputResolution;
//         cout << "inputResolution : " <<this->inputResolution<<"x"<<this->inputResolution<<endl;
//     }

//     InterfaceType getInterfaceType() override
//     {
//         return InterfaceType::ImageClassification;
//     }

//     Optional_Data getOptionalData() override
//     {
//         Optional_Data data;
//         data.cpu_type = "Broadcom BCM2712 quad-core Arm Cortex A76 @ 2.4GHz";
//         data.accelerator_type = "Hailo-8";
//         data.submitter = "Hailo(Compiler : 3.31v, Runtime : v4.23, SingleMode)"; 
//         data.cooling = "inputResolution:" + to_string(inputResolution);
//         data.operating_system = "Ubuntu 24.04.2 LTS";
//         return data;
//     }

//     void initialize(string modelPath) override
//     {
//         cout<<"initialize"<<endl;
//         cleanup();

//         m_vdevice = VDevice::create().release();

//         auto hef = Hef::create(modelPath).value();
//         auto config_params = m_vdevice->create_configure_params(hef).value();
//         auto network_groups = m_vdevice->configure(hef, config_params).value();
//         m_network_group = network_groups[0];

//         auto input_params = m_network_group->make_input_vstream_params(
//                                                false, HAILO_FORMAT_TYPE_UINT8,
//                                                HAILO_DEFAULT_VSTREAM_TIMEOUT_MS,
//                                                HAILO_DEFAULT_VSTREAM_QUEUE_SIZE,
//                                                "")
//                                 .value();

//         auto output_params = m_network_group->make_output_vstream_params(
//                                                 false, HAILO_FORMAT_TYPE_FLOAT32,
//                                                 HAILO_DEFAULT_VSTREAM_TIMEOUT_MS,
//                                                 HAILO_DEFAULT_VSTREAM_QUEUE_SIZE,
//                                                 "")
//                                  .value();

//         auto infer_expected = InferVStreams::create(*m_network_group, input_params, output_params);
//         if (!infer_expected)
//         {
//             throw std::runtime_error("InferVStreams::create failed");
//         }
//         m_pipeline = std::make_unique<InferVStreams>(std::move(*infer_expected));

//         m_input_name = input_params.begin()->first;
//         m_output_name = output_params.begin()->first;
//     }

//     VariantType preprocessVisionData(const string &imagePath) override
//     {
//         cv::Mat img = cv::imread(imagePath, cv::IMREAD_COLOR);
//         cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
//         cv::resize(img, img, cv::Size(inputResolution, inputResolution));
//         std::vector<uint8_t> input_buf(img.total() * img.channels());
//         std::memcpy(input_buf.data(), img.data, input_buf.size());
//         return input_buf;
//     }

//     std::vector<BMTVisionResult> inferVision(const std::vector<VariantType> &data) override
//     {
//         std::vector<BMTVisionResult> results;
//         results.reserve(data.size());

//         for (const auto &item : data)
//         {
//             const auto &input_buf = std::get<std::vector<uint8_t>>(item);
//             std::map<std::string, MemoryView> input_data;
//             std::map<std::string, MemoryView> output_data;
//             input_data[m_input_name] = MemoryView(const_cast<uint8_t *>(input_buf.data()), input_buf.size());
//             std::vector<float> output_buf(1000);
//             output_data[m_output_name] = MemoryView(output_buf.data(), output_buf.size() * sizeof(float));
//             m_pipeline->infer(input_data, output_data, /*batch_size*/ 1);

//             BMTVisionResult r;
//             r.classProbabilities = std::move(output_buf);
//             results.push_back(std::move(r));
//         }
//         return results;
//     }
// };

// int main(int argc, char *argv[])
// {   
//     /*
//     aibmtmodelsubmission@gmail.com

//     rm -rf CMakeCache.txt CMakeFiles .ninja* build.ninja rules.ninja \
//     cmake_install.cmake compile_commands.json qtcsettings.cmake .qtc AI_BMT_GUI_Submitter
//     cmake -G "Ninja" ..
//     export LD_LIBRARY_PATH=$(pwd)/lib
//     cmake --build .
//     ./AI_BMT_GUI_Submitter
//     */
//     try
//     {
//         int inputResolution = 448;
//         std::shared_ptr<AI_BMT_Interface> interface = std::make_shared<HailoBMT_Input_Resolution>(inputResolution);
//         // bool isCustomDataset = false;
//         // std::shared_ptr<AI_BMT_Interface> interface = std::make_shared<HailoBMT>(isCustomDataset);
//         return AI_BMT_GUI_CALLER::call_BMT_GUI_For_Single_Task(argc, argv, interface);
//     }
//     catch (const std::exception &ex)
//     {
//         std::cout << ex.what() << std::endl;
//         return -1;
//     }
// }


#include "hailo/hailort.hpp"
#include "ai_bmt_interface.h"
#include "ai_bmt_gui_caller.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <vector>
#include <filesystem>
#include <memory>
#include <string>
#include <algorithm>
#include <thread>
#include <variant>
#include <stdexcept>
#include <mutex>
#include <future>
#include "utils/async_inference.hpp"
#include "utils/utils.hpp"
using namespace hailort;
using namespace std;
#if defined(__unix__)
#include <sys/mman.h>
#endif

constexpr int WIDTH = 224;
constexpr int HEIGHT = 224;

using BMTDataType = vector<float>;
/////////// Constants ///////////
constexpr size_t MAX_QUEUE_SIZE = 9960;
/////////////////////////////////

int argmax(const std::vector<float> &vec)
{
    return static_cast<int>(std::distance(vec.begin(), std::max_element(vec.begin(), vec.end())));
}

hailo_status run_preprocess(
    std::shared_ptr<BoundedTSQueue<PreprocessedFrameItem>> preprocessed_queue, 
    const vector<VariantType> &data, 
    size_t start, size_t end)
{
    for (int i = start; i < end; i++)
    {
        vector<uint8_t> inputBuf = get<vector<uint8_t>>(data[i]);
        auto preprocessed_frame_item = create_preprocessed_frame_item(inputBuf, WIDTH, HEIGHT, i);
        preprocessed_queue->push(preprocessed_frame_item);
    }
    preprocessed_queue->stop();
    return HAILO_SUCCESS;
}

hailo_status run_preprocess_inputResolution(
    std::shared_ptr<BoundedTSQueue<PreprocessedFrameItem>> preprocessed_queue,
    const vector<VariantType> &data,
    size_t start, size_t end,
    int width, int height)
{
    for (size_t i = start; i < end; i++)
    {
        vector<uint8_t> inputBuf = get<vector<uint8_t>>(data[i]);
        auto item = create_preprocessed_frame_item(inputBuf, width, height, i);
        preprocessed_queue->push(item);
    }
    preprocessed_queue->stop();
    return HAILO_SUCCESS;
}

hailo_status run_inference_async(std::shared_ptr<BoundedTSQueue<PreprocessedFrameItem>> preprocessed_queue, shared_ptr<AsyncModelInfer> model)
{
    while (true)
    {
        PreprocessedFrameItem item;
        if (!preprocessed_queue->pop(item))
            break;

        model->infer(std::make_shared<vector<uint8_t>>(item.resized_for_infer), item.frame_idx);
    }

    return HAILO_SUCCESS;
}

hailo_status run_post_process(std::shared_ptr<BoundedTSQueue<InferenceOutputItem>> results_queue, vector<BMTVisionResult> &batchResult, size_t bs)
{

    size_t i = 0;
    while (true)
    {
        InferenceOutputItem output_item;
        if (!results_queue->pop(output_item))
            break;

        auto frame_idx = output_item.frame_idx;
        // std::cout<<output_item.frame_idx<<std::endl;
        size_t num_elements = 1000;
        std::vector<float> float_data(num_elements);
        std::memcpy(float_data.data(), output_item.output_data_and_infos[0].first, float_data.size() * sizeof(float));
        BMTVisionResult result;
        result.classProbabilities = std::move(float_data);//float_data;
        batchResult[frame_idx] = std::move(result);//result;

        i++;
        if (i == bs)
            results_queue->stop();
    }

    return HAILO_SUCCESS;
}

class Virtual_Submitter_Implementation : public AI_BMT_Interface
{
    std::shared_ptr<BoundedTSQueue<PreprocessedFrameItem>> preprocessed_queue;
    std::shared_ptr<BoundedTSQueue<InferenceOutputItem>> results_queue;
    shared_ptr<AsyncModelInfer> model;
public:
    Virtual_Submitter_Implementation()
    {
    }

    virtual InterfaceType getInterfaceType() override
    {
        return InterfaceType::ImageClassification;
    }

    virtual Optional_Data getOptionalData() override
    {
        Optional_Data data;
        data.cpu_type = "Broadcom BCM2712 quad-core Arm Cortex A76 processor @ 2.4GHz"; // e.g., Intel i7-9750HF
        data.accelerator_type = "Hailo-8";                                              // e.g., DeepX M1(NPU)
        data.submitter = "Hailo(Compiler : 3.31v, Runtime : v4.23, Offlinemode)";                                                       // e.g., DeepX
        data.cpu_core_count = "4";                                                      // e.g., 16
        data.cpu_ram_capacity = "8GB";                                                  // e.g., 32GB
        data.cooling = "Air";                                                           // e.g., Air, Liquid, Passive
        data.cooling_option = "Active";                                                 // e.g., Active, Passive (Active = with fan/pump, Passive = without fan)
        data.cpu_accelerator_interconnect_interface = "PCIe 3.0 4-lane";                // e.g., PCIe Gen5 x16
        data.benchmark_model = "mobilenet_v2_opset10";                                  // e.g., ResNet-50
        data.operating_system = "Ubuntu 24.04.2 LTS";                                   // e.g., Ubuntu 20.04.5 LTS
        return data;
    }

    virtual void initialize(string modelPath) override
    {
        model = make_shared<AsyncModelInfer>();
        model->crt();
        model->PathAndResult(modelPath);
        preprocessed_queue = std::make_shared<BoundedTSQueue<PreprocessedFrameItem>>(MAX_QUEUE_SIZE);
        results_queue = std::make_shared<BoundedTSQueue<InferenceOutputItem>>(MAX_QUEUE_SIZE);
        model->configure(results_queue);
    }

    virtual VariantType preprocessVisionData(const string &imagePath) override
    {
        cv::Mat img = cv::imread(imagePath, cv::IMREAD_COLOR);
        if (img.empty())
        {
            throw std::runtime_error("Image not found or invalid.");
        }

        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
        vector<uint8_t> inputBuf(HEIGHT * WIDTH * 3);
        std::memcpy(inputBuf.data(), img.data, HEIGHT * WIDTH * 3);

        return inputBuf;
    }

    virtual vector<BMTVisionResult> inferVision(const vector<VariantType> &data) override
    {
        size_t frame_count = data.size();
        vector<BMTVisionResult> batchResult(frame_count);
        for (size_t i = 0; i < frame_count; i += MAX_QUEUE_SIZE)
        {
            size_t currentBatchSize = min(MAX_QUEUE_SIZE, frame_count - i);
            size_t start = i;
            size_t end = i + currentBatchSize;
            auto preprocess_thread = std::async(run_preprocess,
                                                preprocessed_queue,
                                                std::ref(data),
                                                start,
                                                end);
            auto inference_thread = std::async(run_inference_async,
                                               preprocessed_queue,
                                               model);
            auto output_parser_thread = std::async(run_post_process,
                                                   results_queue,
                                                   std::ref(batchResult),
                                                   currentBatchSize);
            hailo_status status = wait_and_check_threads(
                preprocess_thread, "Preprocess",
                inference_thread, "Inference",
                output_parser_thread, "Postprocess ");

            if (status != HAILO_SUCCESS)
            {
                throw std::runtime_error("Inference failed");
            }
        }
        preprocessed_queue->reset();
        results_queue->reset();
        model->clear();
        return batchResult;
    }
};

/*
class Virtual_Submitter_Implementation_InputResolution : public AI_BMT_Interface
{
    std::shared_ptr<BoundedTSQueue<PreprocessedFrameItem>> preprocessed_queue;
    std::shared_ptr<BoundedTSQueue<InferenceOutputItem>> results_queue;
    shared_ptr<AsyncModelInfer> model;
    int inputResolution;

public:
    Virtual_Submitter_Implementation_InputResolution(int inputResolution):inputResolution(inputResolution)
    {
        cout << "inputResolution : " <<this->inputResolution<<"x"<<this->inputResolution<<endl;
    }

    virtual InterfaceType getInterfaceType() override
    {
        return InterfaceType::ImageClassification;
    }

    virtual Optional_Data getOptionalData() override
    {
        Optional_Data data;
        data.cpu_type = "Broadcom BCM2712 quad-core Arm Cortex A76 processor @ 2.4GHz"; // e.g., Intel i7-9750HF
        data.accelerator_type = "Hailo-8";                                              // e.g., DeepX M1(NPU)
        data.submitter = "Hailo(Compiler : 3.31v, Runtime : v4.23, Offlinemode)";       // e.g., Deep                                              // e.g., 32GB
        data.cooling = "inputResolution:" + to_string(inputResolution);                 // e.g., Air, Liquid, Passive
        data.operating_system = "Ubuntu 24.04.2 LTS";                                   // e.g., Ubuntu 20.04.5 LTS
        return data;
    }

    virtual void initialize(string modelPath) override
    {
        model = make_shared<AsyncModelInfer>();
        model->crt();
        model->PathAndResult(modelPath);
        preprocessed_queue = std::make_shared<BoundedTSQueue<PreprocessedFrameItem>>(MAX_QUEUE_SIZE);
        results_queue = std::make_shared<BoundedTSQueue<InferenceOutputItem>>(MAX_QUEUE_SIZE);
        model->configure(results_queue);
    }

    virtual VariantType preprocessVisionData(const string &imagePath) override
    {
        cv::Mat img = cv::imread(imagePath, cv::IMREAD_COLOR);
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
        cv::resize(img, img, cv::Size(inputResolution, inputResolution));
        vector<uint8_t> inputBuf(inputResolution * inputResolution * 3);
        std::memcpy(inputBuf.data(), img.data, inputResolution * inputResolution * 3);
        return inputBuf;
    }

    virtual vector<BMTVisionResult> inferVision(const vector<VariantType> &data) override
    {
        size_t frame_count = data.size();
        vector<BMTVisionResult> batchResult(frame_count);
        for (size_t i = 0; i < frame_count; i += MAX_QUEUE_SIZE)
        {
            size_t currentBatchSize = min(MAX_QUEUE_SIZE, frame_count - i);
            size_t start = i;
            size_t end = i + currentBatchSize;
            auto preprocess_thread = std::async(run_preprocess_inputResolution,
                                                preprocessed_queue,
                                                std::ref(data),
                                                start,
                                                end,
                                                inputResolution,
                                                inputResolution);
            auto inference_thread = std::async(run_inference_async,
                                               preprocessed_queue,
                                               model);
            auto output_parser_thread = std::async(run_post_process,
                                                   results_queue,
                                                   std::ref(batchResult),
                                                   currentBatchSize);
            hailo_status status = wait_and_check_threads(
                preprocess_thread, "Preprocess",
                inference_thread, "Inference",
                output_parser_thread, "Postprocess ");

            if (status != HAILO_SUCCESS)
            {
                throw std::runtime_error("Inference failed");
            }
        }
        preprocessed_queue->reset();
        results_queue->reset();
        model->clear();
        return batchResult;
    }
};
*/

int main(int argc, char *argv[])
{
    /*
    aibmtmodelsubmission@gmail.com

    rm -rf CMakeCache.txt CMakeFiles .ninja* build.ninja rules.ninja \
    cmake_install.cmake compile_commands.json qtcsettings.cmake .qtc AI_BMT_GUI_Submitter
    cmake -G "Ninja" ..
    export LD_LIBRARY_PATH=$(pwd)/lib
    cmake --build .
    ./AI_BMT_GUI_Submitter
    */

    try
    {
        //int inputResolution = 448;
        //cout<< "Offline Scenario"<<endl;
        //shared_ptr<AI_BMT_Interface> interface = make_shared<Virtual_Submitter_Implementation_InputResolution>(inputResolution);
        shared_ptr<AI_BMT_Interface> interface = make_shared<Virtual_Submitter_Implementation>();
        return AI_BMT_GUI_CALLER::call_BMT_GUI_For_Single_Task(argc, argv, interface);
    }
    catch (const exception &ex)
    {
        cout << ex.what() << endl;
    }
}

