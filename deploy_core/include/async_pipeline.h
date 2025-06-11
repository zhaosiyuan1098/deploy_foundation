//
// Created by siyuan on 25-6-10.
//

#ifndef ASYNC_PIPELINE_H
#define ASYNC_PIPELINE_H

#include <functional>
#include <future>
#include <memory>
#include <thread>
#include <unordered_map>

#include <glog/logging.h>
#include <glog/log_severity.h>
#include "common.h"
#include "blob_buffer.h"
#include "pipeline_instance.h"

namespace async_pipeline
{
    class IpipelineImageData
    {
    public:
        struct ImageDataInfo
        {
            uint8_t* data_pointer;
            int image_height;
            int image_width;
            int image_channels;
            DataLocation location;
            ImageDataFormat format;
        };

        [[nodiscard]] virtual const ImageDataInfo& GetImageDataInfo() const = 0;

    protected:
        virtual ~IpipelineImageData() = default;
    };

    class IPipelinePackage
    {
    public:
        virtual std::shared_ptr<infer_core::IBlobsBuffer> GetInferBuffer() = 0;

    protected:
        virtual ~IPipelinePackage() = default;
    };

    template <typename ResultType, typename GenResult>
    class BaseAsyncPipeline
    {
        using ParsingType = std::shared_ptr<IPipelinePackage>;
        using Block_t = AsyncPipelineBlock<ParsingType>;
        using Context_t = AsyncPipelineContext<ParsingType>;

    public:
        BaseAsyncPipeline() = default;
        const Context_t& getContext() const;
        [[nodiscard]] std::future<ResultType> push(const std::string& pipeline_name,
                                                   const ParsingType& package);
        bool isInitialized(const std::string& pipeline_name);
        void close();
        void stop();
        void init();

    protected:
        ~BaseAsyncPipeline();
        static Block_t buildPipelineBlock(const std::function<bool(ParsingType)>& func,
                                          const std::string& block_name);
        void config(const std::string& pipeline_name, const std::vector<Context_t>& block_list);

    private:
        std::unordered_map<std::string, AsyncPipelineInstance<ParsingType>> map_name2instance_;

        size_t package_index_ = 0;
        std::unordered_map<size_t, std::promise<ResultType>> map_index2result_;
        GenResult gen_result_from_package_;
    };

    template <typename ResultType, typename GenResult>
    const typename BaseAsyncPipeline<ResultType, GenResult>::Context_t& BaseAsyncPipeline<ResultType, GenResult>::
    getContext() const
    {
        if (map_name2instance_.size() != 1)
        {
            throw std::runtime_error("[BaseAsyncPipeline] expect one pipeline, got " +
                                     std::to_string(map_name2instance_.size()));
        }
        return map_name2instance_.begin()->second.GetContext();
    }

    template <typename ResultType, typename GenResult>
    std::future<ResultType> BaseAsyncPipeline<ResultType, GenResult>::push(const std::string& pipeline_name,
        const ParsingType& package)
    {
        if (map_name2instance_.find(pipeline_name) == map_name2instance_.end())
        {
            LOG(ERROR) << "[BaseAsyncPipeline] `PushPipeline` pipeline {" << pipeline_name
                       << "} is not valid !!!";
            return std::future<ResultType>();
        }

        map_index2result_[package_index_] = std::promise<ResultType>();
        auto ret                          = map_index2result_[package_index_].get_future();

        auto callback = [this, package_index = package_index_](const ParsingType &package) -> bool {
            ResultType result = gen_result_from_package_(package);
            map_index2result_[package_index].set_value(std::move(result));
            map_index2result_.erase(package_index);
            return true;
        };
        map_name2instance_[pipeline_name].push(package, callback);

        package_index_++;

        return std::move(ret);
    }

    template <typename ResultType, typename GenResult>
    bool BaseAsyncPipeline<ResultType, GenResult>::isInitialized(const std::string& pipeline_name)
    {
        if (map_name2instance_.find(pipeline_name) == map_name2instance_.end())
        {
            return false;
        }
        return map_name2instance_[pipeline_name].isInitialized();
    }

    template <typename ResultType, typename GenResult>
    void BaseAsyncPipeline<ResultType, GenResult>::close()
    {
        for (auto &p_name_ins : map_name2instance_)
        {
            p_name_ins.second.close();
        }
    }

    template <typename ResultType, typename GenResult>
    void BaseAsyncPipeline<ResultType, GenResult>::stop()
    {
        for (auto &p_name_ins : map_name2instance_)
        {
            p_name_ins.second.stop();
        }
    }

    template <typename ResultType, typename GenResult>
    void BaseAsyncPipeline<ResultType, GenResult>::init()
    {
        for (auto &p_name_ins : map_name2instance_)
        {
            p_name_ins.second.init();
        }
    }

    template <typename ResultType, typename GenResult>
    BaseAsyncPipeline<ResultType, GenResult>::~BaseAsyncPipeline()
    {
        this->close();
    }

    template <typename ResultType, typename GenResult>
    typename BaseAsyncPipeline<ResultType, GenResult>::Block_t BaseAsyncPipeline<ResultType, GenResult>::
    buildPipelineBlock(const std::function<bool(ParsingType)>& func, const std::string& block_name)
    {
        return Block_t(func, block_name);
    }

    template <typename ResultType, typename GenResult>
    void BaseAsyncPipeline<ResultType, GenResult>::config(const std::string& pipeline_name,
        const std::vector<Context_t>& block_list)
    {
        map_name2instance_.emplace(pipeline_name, block_list);
    }
}


#endif //ASYNC_PIPELINE_H
