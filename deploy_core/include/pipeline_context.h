//
// Created by siyuan on 25-6-10.
//

#ifndef PIPELINE_CONTEXT_H
#define PIPELINE_CONTEXT_H

#include "pipeline_block.h"
#include <vector>


namespace async_pipeline
{
    template <typename ParsingType>
    class AsyncPipelineContext
    {
        using Block_t = AsyncPipelineBlock<ParsingType>;
        using Context_t = AsyncPipelineContext<ParsingType>;

    public:
        AsyncPipelineContext()=default;

        AsyncPipelineContext(const Block_t &block) : blocks_({block}){};

        AsyncPipelineContext(const std::vector<Block_t> &block_vec);

        AsyncPipelineContext &operator=(const std::vector<Block_t> &block_vec);

        AsyncPipelineContext(const Context_t &context) : blocks_(context.blocks_){};

        AsyncPipelineContext(const std::vector<Context_t> &context_vec);

        AsyncPipelineContext &operator=(const std::vector<Context_t> &context_vec);

        AsyncPipelineContext &operator=(const Context_t &context);


        std::vector<Block_t> blocks_;
    };

    template <typename ParsingType>
    AsyncPipelineContext<ParsingType>::AsyncPipelineContext(const std::vector<Block_t>& block_vec)
    {
        for (const auto &block : block_vec)
        {
            blocks_.push_back(block);
        }
    }

    template <typename ParsingType>
    AsyncPipelineContext<ParsingType>& AsyncPipelineContext<ParsingType>::operator=(
        const std::vector<Block_t>& block_vec)
    {
        for (const auto &block : block_vec)
        {
            blocks_.push_back(block);
        }
        return *this;
    }

    template <typename ParsingType>
    AsyncPipelineContext<ParsingType>::AsyncPipelineContext(const std::vector<Context_t>& context_vec)
    {
        for (const auto &context : context_vec)
        {
            for (const auto &block : context.blocks_)
            {
                blocks_.push_back(block);
            }
        }
    }

    template <typename ParsingType>
    AsyncPipelineContext<ParsingType>& AsyncPipelineContext<ParsingType>::operator=(
        const std::vector<Context_t>& context_vec)
    {
        for (const auto &context : context_vec)
        {
            for (const auto &block : context.blocks_)
            {
                blocks_.push_back(block);
            }
        }
        return *this;
    }

    template <typename ParsingType>
    AsyncPipelineContext<ParsingType>& AsyncPipelineContext<ParsingType>::operator=(const Context_t& context)
    {
        for (const auto &block : context.blocks_)
        {
            blocks_.push_back(block);
        }
        return *this;
    }
}
#endif //PIPELINE_CONTEXT_H
