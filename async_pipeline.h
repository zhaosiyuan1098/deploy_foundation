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

class IPipelinePackage {
public:

    virtual std::shared_ptr<infer_core::IBlobBuffer> GetInferBuffer() = 0;

protected:
    virtual ~IPipelinePackage() = default;
};

#endif //ASYNC_PIPELINE_H
