//
// Created by siyuan on 25-6-10.
//

#ifndef INFER_CORE_H
#define INFER_CORE_H

#include <atomic>
#include <memory>
#include <thread>
#include <vector>

#include "common.h"
#include "block_queue.h"
#include "async_pipeline.h"
#include "blob_buffer.h"

namespace infer_core
{
    class IInferCore
    {
    public:
        virtual std::shared_ptr<IBlobsBuffer> AllocBlobsBuffer() = 0;
        virtual InferCoreType GetType();
        virtual std::string GetName();


    protected:
        virtual ~IInferCore() = default;
        virtual bool PreProcess(std::shared_ptr<async_pipeline::IPipelinePackage> buffer) = 0;
        virtual bool Inference(std::shared_ptr<async_pipeline::IPipelinePackage> buffer) = 0;
        virtual bool PostProcess(std::shared_ptr<async_pipeline::IPipelinePackage> buffer) = 0;
    };


    class MemBufferPool
    {
    public:
        MemBufferPool(IInferCore* infer_core, int pool_size);
        std::shared_ptr<IBlobsBuffer> Alloc(bool block);
        void release();
        int remainSize();

        ~MemBufferPool();

    private:
        const int pool_size_;
        BlockQueue<IBlobsBuffer*> dynamic_pool_;
        std::unordered_map<IBlobsBuffer*, std::shared_ptr<IBlobsBuffer>> static_pool_;
    };



    class DummyInferCoreGenResultType
    {
    public:
        bool operator()(const std::shared_ptr<async_pipeline::IPipelinePackage>& /*package*/) const
        {
            return true;
        }
    };

    struct _InnerSyncInferPackage : public async_pipeline::IPipelinePackage {
    public:
        std::shared_ptr<IBlobsBuffer> GetInferBuffer() override
        {
            return buffer;
        }
        std::shared_ptr<IBlobsBuffer> buffer;
    };


    class BaseInferCore : public IInferCore,
                          protected async_pipeline::BaseAsyncPipeline<bool, DummyInferCoreGenResultType>
    {
    public:
        using BaseAsyncPipeline::getContext;
        bool SyncInfer(std::shared_ptr<IBlobsBuffer> buffer, int batch_size = 1);
        std::shared_ptr<IBlobsBuffer> GetBuffer(bool block);
        virtual void Release();

    protected:
        BaseInferCore();
        typedef std::shared_ptr<async_pipeline::IPipelinePackage> ParsingType;
        ~BaseInferCore() override;
        void Init(int mem_buf_size = 5);

    private:
        std::unique_ptr<MemBufferPool> mem_buf_pool_{nullptr};
    };








}
#endif //INFER_CORE_H
