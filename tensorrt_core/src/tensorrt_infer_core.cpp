//
// Created by siyuan on 25-6-11.
//
#include "tensorrt_infer_core.h"
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <unordered_set>

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include "infer_core.h"

namespace infer_core
{
    infer_core::TrtInferCore::TrtInferCore(std::string engine_path, int mem_buf_size)
    {
        LoadEngine(engine_path);
        ResolveModelInformation(map_blob_name2shape_);

        BaseInferCore::Init(mem_buf_size);

        cudaStreamCreate(&preproces_stream_);
        cudaStreamCreate(&inference_stream_);
        cudaStreamCreate(&postprocess_stream_);
    }

    infer_core::TrtInferCore::TrtInferCore(std::string engine_path,
                                           const std::unordered_map<std::string, std::vector<int64_t>>& blobs_shape,
                                           int mem_buf_size)
    {
        LoadEngine(engine_path);
        map_blob_name2shape_ = blobs_shape;
        ResolveModelInformation(map_blob_name2shape_);

        BaseInferCore::Init(mem_buf_size);

        cudaStreamCreate(&preproces_stream_);
        cudaStreamCreate(&inference_stream_);
        cudaStreamCreate(&postprocess_stream_);
    }

    std::shared_ptr<infer_core::IBlobsBuffer> infer_core::TrtInferCore::AllocBlobsBuffer()
    {
        auto ret = std::make_shared<TrtBlobBuffer>();

        const int blob_number = engine_->getNbIOTensors();
        CHECK(blob_number >= 2);
        ret->device_blobs_buffer_.resize(blob_number);
        ret->host_blobs_buffer_.resize(blob_number);

        for (int i = 0; i < blob_number; ++i)
        {
            const std::string s_blob_name = engine_->getIOTensorName(i);
            int64_t blob_byte_size = sizeof(float);
            const auto& blob_shape = map_blob_name2shape_[s_blob_name];
            for (const int64_t d : blob_shape)
            {
                blob_byte_size *= d;
            }

            // alloc buffer memory
            // on device
            CHECK(cudaMalloc(&ret->device_blobs_buffer_[i], blob_byte_size) == cudaSuccess);
            CHECK(cudaMemset(ret->device_blobs_buffer_[i], 0, blob_byte_size) == cudaSuccess);
            CHECK(cudaDeviceSynchronize() == cudaSuccess);
            // on host
            ret->host_blobs_buffer_[i] = new u_char[blob_byte_size];

            // maintain buffer ptr
            ret->outer_map_blob2ptr_.emplace(s_blob_name,
                                             std::pair{ret->host_blobs_buffer_[i], DataLocation::HOST});
            // mapping blob_name and buffer_ptr
            ret->inner_map_device_blob2ptr_.emplace(s_blob_name, ret->device_blobs_buffer_[i]);
            ret->inner_map_host_blob2ptr_.emplace(s_blob_name, ret->host_blobs_buffer_[i]);

            // mapping blob_name and default blob_shape
            ret->map_blob_name2shape_.emplace(s_blob_name, blob_shape);
        }

        // initialize the buffer ptr vector which will be used when tensorrt engine do inference.
        ret->buffer_input_core_ = ret->device_blobs_buffer_;

        return ret;
    }

    bool infer_core::TrtInferCore::PreProcess(std::shared_ptr<async_pipeline::IPipelinePackage> buffer)
    {
        CHECK_STATE(buffer != nullptr, "[TrtInferCore] PreProcess got WRONG input data format!");
        auto p_buf = std::dynamic_pointer_cast<TrtBlobBuffer>(buffer->GetInferBuffer());
        CHECK_STATE(p_buf != nullptr, "[TrtInferCore] PreProcess got WRONG p_buf data format!");

        // Set the input buffer data
        for (const auto& p_name_index : map_input_blob_name2index_)
        {
            const std::string& s_blob_name = p_name_index.first;
            const int index = p_name_index.second;

            // Get the customed blob buffer data information, including data ptr and location.
            const auto& p_ptr_loc = p_buf->getOuterBlobBuffer(s_blob_name);
            // Transport buffer data from host to device, if the customed blob data is on host.
            if (p_ptr_loc.second == DataLocation::HOST)
            {
                p_buf->buffer_input_core_[index] = p_buf->inner_map_device_blob2ptr_[s_blob_name];
                cudaMemcpyAsync(p_buf->buffer_input_core_[index], p_ptr_loc.first,
                                map_blob_name2size_[s_blob_name], cudaMemcpyHostToDevice, preproces_stream_);
            }
            else
            {
                p_buf->buffer_input_core_[index] = p_ptr_loc.first;
            }
        }

        // Set the output buffer data ptr. Allways use inner pre-allocated device buffer.
        for (const auto& p_name_index : map_output_blob_name2index_)
        {
            const std::string& s_blob_name = p_name_index.first;
            const int index = p_name_index.second;
            p_buf->buffer_input_core_[index] = p_buf->inner_map_device_blob2ptr_[s_blob_name];
        }

        cudaStreamSynchronize(preproces_stream_);

        return true;
    }

    bool infer_core::TrtInferCore::Inference(std::shared_ptr<async_pipeline::IPipelinePackage> buffer)
    {
        // Create tensorrt context if this is the first time execution of this thread.
        std::thread::id cur_thread_id = std::this_thread::get_id();
        if (s_map_tid2context_.find(cur_thread_id) == s_map_tid2context_.end())
        {
            std::shared_ptr<nvinfer1::IExecutionContext> context{engine_->createExecutionContext()};
            {
                std::unique_lock<std::mutex> u_lck(s_context_lck_);
                s_map_tid2context_.insert({cur_thread_id, context});
            }
        }
        auto context = s_map_tid2context_[cur_thread_id];

        // Get buffer ptr
        CHECK_STATE(buffer != nullptr, "[TrtInferCore] PreProcess got WRONG input data format!");
        auto p_buf = std::dynamic_pointer_cast<TrtBlobBuffer>(buffer->GetInferBuffer());
        CHECK_STATE(p_buf != nullptr, "[TrtInferCore] PreProcess got WRONG p_buf data format!");

        TrtBlobBuffer& buf = *p_buf;
        // Set dynamic blob shape
        for (const auto& p_name_shape : buf.map_blob_name2shape_)
        {
            const auto& s_blob_name = p_name_shape.first;
            const auto& v_shape = p_name_shape.second;

            if (engine_->getTensorIOMode(s_blob_name.c_str()) != nvinfer1::TensorIOMode::kINPUT)
            {
                continue;
            }

            nvinfer1::Dims dynamic_dim;
            dynamic_dim.nbDims = v_shape.size();
            for (size_t i = 0; i < v_shape.size(); ++i)
            {
                dynamic_dim.d[i] = v_shape[i];
            }
            CHECK_STATE(context->setInputShape(s_blob_name.c_str(), dynamic_dim),
                        "[TrtInferCore] Inference execute `context->setInputShape` failed!!!");
        }

#if NV_TENSORRT_MAJOR == 10
  // CHECK_STATE(context->allInputDimensionsSpecified(),
  //             "[TrtInferCore] Got unspecified dimensions of input!!!");

  for (const auto &p_name_index : map_input_blob_name2index_)
  {
    const std::string &s_blob_name   = p_name_index.first;
    const int          index         = p_name_index.second;
    context->setTensorAddress(s_blob_name.c_str(), buf.buffer_input_core_[index]);
  }
  for (const auto &p_name_index : map_output_blob_name2index_)
  {
    const std::string &s_blob_name   = p_name_index.first;
    const int          index         = p_name_index.second;
    context->setTensorAddress(s_blob_name.c_str(), buf.buffer_input_core_[index]);
  }
  context->enqueueV3(inference_stream_);

#else
        // Do inference use `buf.buffer_input_core_` which is prepared by `PreProcess` stage.
        CHECK_STATE(context->enqueueV2(buf.buffer_input_core_.data(), inference_stream_, nullptr),
                    "[TrtInferCore] Inference execute `context->enqueueV2` failed!!!");
#endif

        cudaStreamSynchronize(inference_stream_);
        return true;
    }

    bool infer_core::TrtInferCore::PostProcess(std::shared_ptr<async_pipeline::IPipelinePackage> buffer)
    {
        CHECK_STATE(buffer != nullptr, "[TrtInferCore] PreProcess got WRONG input data format!");
        auto p_buf = std::dynamic_pointer_cast<TrtBlobBuffer>(buffer->GetInferBuffer());
        CHECK_STATE(p_buf != nullptr, "[TrtInferCore] PreProcess got WRONG p_buf data format!");

        for (const auto& p_name_index : map_output_blob_name2index_)
        {
            const std::string& s_blob_name = p_name_index.first;
            const int index = p_name_index.second;
            const auto& p_ptr_loc = p_buf->getOuterBlobBuffer(s_blob_name);
            // Transport output buffer from device to host, if user needs host readable data.
            if (p_ptr_loc.second == DataLocation::HOST)
            {
                cudaMemcpyAsync(p_ptr_loc.first, p_buf->buffer_input_core_[index],
                                map_blob_name2size_[s_blob_name], cudaMemcpyDeviceToHost,
                                postprocess_stream_);
            }
            // Transport output buffer from local device buffer to given device buffer.
            else if (p_ptr_loc.first != p_buf->buffer_input_core_[index])
            {
                cudaMemcpyAsync(p_ptr_loc.first, p_buf->buffer_input_core_[index],
                                map_blob_name2size_[s_blob_name], cudaMemcpyDeviceToDevice,
                                postprocess_stream_);
            }
        }

        cudaStreamSynchronize(postprocess_stream_);
        return true;
    }

    infer_core::TrtInferCore::~TrtInferCore()
    {
        BaseInferCore::Release();
    }

    void infer_core::TrtInferCore::LoadEngine(const std::string& engine_path)
    {
        std::ifstream file(engine_path, std::ios::binary);
        if (!file.good())
        {
            throw std::runtime_error("[TrtInferCore] Failed to read engine file!!!");
        }

        std::vector<char> data;

        file.seekg(0, file.end);
        const auto size = file.tellg();
        file.seekg(0, file.beg);

        data.resize(size);
        file.read(data.data(), size);

        file.close();

        runtime_.reset(nvinfer1::createInferRuntime(logger_));

        engine_.reset(runtime_->deserializeCudaEngine(data.data(), data.size()));
        if (engine_ == nullptr)
        {
            throw std::runtime_error("[TrtInferCore] Failed to create trt engine!!!");
        }
        LOG(INFO) << "[TrtInferCore] created tensorrt engine and "
            "context ! ";
    }

    void infer_core::TrtInferCore::ResolveModelInformation(
        std::unordered_map<std::string, std::vector<int64_t>>& blobs_shape)
    {
        const int blob_number = engine_->getNbIOTensors();
        LOG(INFO) << "[TrtInferCore] model has " << blob_number << " blobs";
        CHECK(blob_number >= 2);

        bool resolve_blob_shape = blobs_shape.empty();

        for (int i = 0; i < blob_number; ++i)
        {
            const char* blob_name = engine_->getIOTensorName(i);
            nvinfer1::Dims dim = engine_->getTensorShape(blob_name);

            const std::string s_blob_name(blob_name);
            if (engine_->getTensorIOMode(blob_name) == nvinfer1::TensorIOMode::kINPUT)
            {
                map_input_blob_name2index_.emplace(s_blob_name, i);
            }
            else
            {
                map_output_blob_name2index_.emplace(s_blob_name, i);
            }

            if (resolve_blob_shape)
            {
                blobs_shape[s_blob_name] = std::vector<int64_t>();
                for (int j = 0; j < dim.nbDims; ++j)
                {
                    // 检查是否包含动态shape，自动解析暂不支持动态shape
                    if (dim.d[j] <= 0)
                    {
                        throw std::runtime_error("[TrtInferCore] unsupport blob dim:" + std::to_string(dim.d[j]) +
                            ", use explicit blob shape consturctor instead");
                    }
                    blobs_shape[s_blob_name].push_back(dim.d[j]);
                }

                std::string s_dim;
                for (auto d : dim.d)
                {
                    s_dim += std::to_string(d) + " ";
                }
                LOG(INFO) << "[TrtInferCore] blob name : " << blob_name << " dims : " << s_dim;
            }

            size_t blob_byte_size = sizeof(float);
            if (blobs_shape.find(s_blob_name) == blobs_shape.end())
            {
                throw std::runtime_error("[TrtInferCore] blob name: " + s_blob_name +
                    " not found in provided blobs_shape map !!!");
            }
            for (const int64_t d : blobs_shape[s_blob_name])
            {
                blob_byte_size *= d;
            }

            map_blob_name2size_[s_blob_name] = blob_byte_size;
        }
    }

    static bool FileSuffixCheck(const std::string& file_path, const std::string& suffix)
    {
        const size_t mark = file_path.rfind('.');
        std::string suf;
        return mark != file_path.npos &&
            (suf = file_path.substr(mark, file_path.size() - mark)) == suffix;
    }

    std::shared_ptr<BaseInferCore> CreateTrtInferCore(std::string model_path, const int mem_buf_size)
    {
        if (!FileSuffixCheck(model_path, ".engine"))
        {
            throw std::invalid_argument("Trt infer core expects file end with `.engine`. But got " +
                model_path + " instead");
        }

        return std::make_shared<TrtInferCore>(model_path, mem_buf_size);
    }

    std::shared_ptr<BaseInferCore> CreateTrtInferCore(
        std::string model_path,
        const std::unordered_map<std::string, std::vector<int64_t>>& input_blobs_shape,
        const std::unordered_map<std::string, std::vector<int64_t>>& output_blobs_shape,
        const int mem_buf_size)
    {
        if (!FileSuffixCheck(model_path, ".engine"))
        {
            throw std::invalid_argument("Trt infer core expects file end with `.engine`. But got " +
                model_path + " instead");
        }

        std::unordered_map<std::string, std::vector<int64_t>> blobs_shape;
        for (const auto& p : input_blobs_shape)
        {
            blobs_shape.insert(p);
        }
        for (const auto& p : output_blobs_shape)
        {
            blobs_shape.insert(p);
        }

        return std::make_shared<TrtInferCore>(model_path, blobs_shape, mem_buf_size);
    }
}
