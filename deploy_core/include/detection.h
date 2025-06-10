//
// Created by siyuan on 25-6-10.
//

#ifndef DETECTION_H
#define DETECTION_H


#include <atomic>
#include <functional>
#include <thread>
#include <vector>

#include <opencv2/opencv.hpp>

#include "async_pipeline.h"
#include "infer_core.h"

namespace detection
{
    class IDetectionPreProcess {
    public:
        virtual float Preprocess(std::shared_ptr<async_pipeline::IPipelineImageData> input_image_data,
                                 std::shared_ptr<inference_core::IBlobsBuffer>       blob_buffer,
                                 const std::string                                  &blob_name,
                                 int                                                 dst_height,
                                 int                                                 dst_width) = 0;
    };


}

#endif //DETECTION_H
