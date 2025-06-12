//
// Created by siyuan on 25-6-11.
//

#ifndef TRITON_CORE_H
#define TRITON_CORE_H

#include <unordered_map>
#include "infer_core.h"
#include "infer_core_factory.h"
#include <iostream>
#include "tensorrt_blob_buffer.h"
#include "logger.h"
#include <cuda_runtime_api.h>

namespace infer_core
{
    std::shared_ptr<BaseInferCore> CreateTrtInferCore(
        std::string model_path,
        const std::unordered_map<std::string, std::vector<int64_t>>& input_blobs_shape = {},
        const std::unordered_map<std::string, std::vector<int64_t>>& output_blobs_shape = {},
        int mem_buf_size = 5);

    std::shared_ptr<BaseInferCoreFactory> CreateTrtInferCoreFactory(
        std::string model_path,
        const std::unordered_map<std::string, std::vector<int64_t>>& input_blobs_shape = {},
        const std::unordered_map<std::string, std::vector<int64_t>>& output_blobs_shape = {},
        int mem_buf_size = 5);

    class TrtInferCore : public BaseInferCore
    {
    public:
        explicit TrtInferCore(std::string engine_path, int mem_buf_size = 5);
        TrtInferCore(std::string engine_path,
                     const std::unordered_map<std::string, std::vector<int64_t>>& blobs_shape,
                     int mem_buf_size = 5);
        std::shared_ptr<IBlobsBuffer> AllocBlobsBuffer() override;
        bool PreProcess(std::shared_ptr<async_pipeline::IPipelinePackage> buffer) override;
        bool Inference(std::shared_ptr<async_pipeline::IPipelinePackage> buffer) override;
        bool PostProcess(std::shared_ptr<async_pipeline::IPipelinePackage> buffer) override;

    protected:
        ~TrtInferCore() override;

    private:
        void LoadEngine(const std::string& engine_path);
        void ResolveModelInformation(std::unordered_map<std::string, std::vector<int64_t>>& blobs_shape);
        TensorrtLogger logger_{};
        std::unique_ptr<nvinfer1::IRuntime> runtime_{nullptr};
        std::unique_ptr<nvinfer1::ICudaEngine> engine_{nullptr};
        std::unordered_map<std::thread::id, std::shared_ptr<nvinfer1::IExecutionContext>> s_map_tid2context_;
        std::mutex s_context_lck_;
        cudaStream_t preproces_stream_, inference_stream_, postprocess_stream_;

        // some model information mapping
        std::unordered_map<std::string, std::vector<int64_t>> map_blob_name2shape_;
        std::unordered_map<std::string, int>                  map_input_blob_name2index_;
        std::unordered_map<std::string, int>                  map_output_blob_name2index_;
        std::unordered_map<std::string, size_t>               map_blob_name2size_;
    };
}


#endif //TRITON_CORE_H
