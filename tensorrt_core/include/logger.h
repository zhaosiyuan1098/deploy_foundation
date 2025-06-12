//
// Created by siyuan on 25-6-11.
//

#ifndef LOGGER_H
#define LOGGER_H

#include <NvInfer.h>
#include<glog/logging.h>

class TensorrtLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char *msg) noexcept override
    {
        if (severity == Severity::kINFO)
            LOG(INFO) << "[Tensorrt] : " << msg;
        else if (severity == Severity::kERROR)
            LOG(ERROR) << "[Tensorrt] : " << msg;
        else if (severity == Severity::kWARNING)
            LOG(WARNING) << "[Tensorrt] : " << msg;
    }
};
#endif //LOGGER_H
