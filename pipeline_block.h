//
// Created by siyuan on 25-6-10.
//

#ifndef PIPELINE_BLOCK_H
#define PIPELINE_BLOCK_H

#include <functional>
#include <future>

#include <glog/log_severity.h>
#include <glog/logging.h>

#include "deploy_core/block_queue.h"

namespace async_pipeline
{
    template <typename ParsingType>
    class AsyncPipelineBlock
    {
    public:
        AsyncPipelineBlock() = default;

        AsyncPipelineBlock(const std::function<bool(ParsingType)>& func) : func_(func)
        {
        }

        AsyncPipelineBlock(const AsyncPipelineBlock& block): func_(block.func_), block_name_(block.block_name_)
        {
        }

        AsyncPipelineBlock& operator=(const AsyncPipelineBlock& block);

        AsyncPipelineBlock(const std::function<bool(ParsingType)>& func, const std::string& block_name)
            : func_(func), block_name_(block_name)
        {
        }

        const std::string& GetName() const
        {
            return block_name_;
        }

        bool operator()(const ParsingType& pipeline_unit) const
        {
            return func_(pipeline_unit);
        }



    private:
        std::function<bool(ParsingType)> func_;
        std::string block_name_;
    };
}


#endif //PIPELINE_BLOCK_H
