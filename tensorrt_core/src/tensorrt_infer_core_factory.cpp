//
// Created by siyuan on 25-6-12.
//
#include "tensorrt_infer_core_factory.h"
namespace  infer_core
{
    TrtInferCoreFactory::TrtInferCoreFactory(TrtInferCoreParams params): params_(params)
    {
    }

    std::shared_ptr<BaseInferCore> TrtInferCoreFactory::Create()
    {
        return CreateTrtInferCore(params_.model_path, params_.input_blobs_shape, params_.output_blobs_shape,
                                  params_.mem_buf_size);
    }
}