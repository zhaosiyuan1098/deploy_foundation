//
// Created by siyuan on 25-6-12.
//
#include "infer_core.h"

namespace infer_core
{
    InferCoreType IInferCore::GetType()
    {
        return NOT_PROVIDED;
    }

    std::string IInferCore::GetName()
    {
        return "";
    }

    MemBufferPool::MemBufferPool(IInferCore* infer_core, int pool_size):
    pool_size_(pool_size),dynamic_pool_(pool_size)
    {

        for (int i = 0; i < pool_size; ++i)
        {
            auto blob_buffer = infer_core->AllocBlobsBuffer();
            dynamic_pool_.blockPush(blob_buffer.get());
            static_pool_.insert({blob_buffer.get(), blob_buffer});
        }

    }

    std::shared_ptr<IBlobsBuffer> MemBufferPool::Alloc(bool block)
    {
        auto func_dealloc = [&](IBlobsBuffer *buf) {
            buf->reset();
            this->dynamic_pool_.blockPush(buf);
        };

        auto buf = block ? dynamic_pool_.blockPop() : dynamic_pool_.tryPop();
        return buf.has_value() ? std::shared_ptr<IBlobsBuffer>(buf.value(), func_dealloc) : nullptr;
    }

    void MemBufferPool::release()
    {
        if (dynamic_pool_.size() != pool_size_)
        {
            LOG(WARNING) << "[MemBufPool] does not maintain all buffers when release func called!";
        }
        static_pool_.clear();
    }

    int MemBufferPool::remainSize()
    {
        return dynamic_pool_.size();
    }

    MemBufferPool::~MemBufferPool()
    {
        this->release();
    }

    bool BaseInferCore::SyncInfer(std::shared_ptr<IBlobsBuffer> buffer, int batch_size)
    {
        auto inner_package    = std::make_shared<_InnerSyncInferPackage>();
        inner_package->buffer = buffer;
        CHECK_STATE(PreProcess(inner_package), "[BaseInferCore] SyncInfer Preprocess Failed!!!");
        CHECK_STATE(Inference(inner_package), "[BaseInferCore] SyncInfer Inference Failed!!!");
        CHECK_STATE(PostProcess(inner_package), "[BaseInferCore] SyncInfer PostProcess Failed!!!");
        return true;
    }

    std::shared_ptr<IBlobsBuffer> BaseInferCore::GetBuffer(bool block)
    {
        return mem_buf_pool_->Alloc(block);
    }

    void BaseInferCore::Release()
    {
        BaseAsyncPipeline::close();
        mem_buf_pool_.reset();
    }

    BaseInferCore::BaseInferCore()
    {
        auto preprocess_block = buildPipelineBlock(
      [&](ParsingType unit) -> bool { return PreProcess(unit); }, "BaseInferCore PreProcess");
        auto inference_block = buildPipelineBlock(
            [&](ParsingType unit) -> bool { return Inference(unit); }, "BaseInferCore Inference");
        auto postprocess_block = buildPipelineBlock(
            [&](ParsingType unit) -> bool { return PostProcess(unit); }, "BaseInferCore PostProcess");
        config("InferCore Pipeline", {preprocess_block, inference_block, postprocess_block});
    }

    BaseInferCore::~BaseInferCore()
    {
        BaseAsyncPipeline::close();
        mem_buf_pool_.reset();
    }

    void BaseInferCore::Init(int mem_buf_size)
    {
        if (mem_buf_size <= 0 || mem_buf_size > 100)
        {
            throw std::invalid_argument("mem_buf_size should be between [1,100], Got: " +
                                        std::to_string(mem_buf_size));
        }
        mem_buf_pool_ = std::make_unique<MemBufferPool>(this, mem_buf_size);
        LOG(INFO) << "successfully init mem buf pool with pool_size : " << mem_buf_size;
    }
}