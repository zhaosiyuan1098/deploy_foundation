//
// Created by siyuan on 25-6-12.
//

#ifndef TENSORRT_INFER_CORE_FACTORY_H
#define TENSORRT_INFER_CORE_FACTORY_H

#include "tensorrt_infer_core.h"
#include"infer_core_factory.h"

namespace infer_core
{
    struct TrtInferCoreParams {
        std::string                                           model_path;
        std::unordered_map<std::string, std::vector<int64_t>> input_blobs_shape;
        std::unordered_map<std::string, std::vector<int64_t>> output_blobs_shape;
        int                                                   mem_buf_size;
    };

    class TrtInferCoreFactory : public BaseInferCoreFactory {
    public:
        TrtInferCoreFactory(TrtInferCoreParams params);
        std::shared_ptr<BaseInferCore> Create() override;
    private:
        TrtInferCoreParams params_;
    };
}

#endif //TENSORRT_INFER_CORE_FACTORY_H
