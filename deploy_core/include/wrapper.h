//
// Created by root on 25-6-11.
//

#ifndef WRAPPER_H
#define WRAPPER_H

#include "async_pipeline.h"

#include <opencv2/opencv.hpp>

class PipelineCvImageWrapper : public async_pipeline::IpipelineImageData {
public:
    PipelineCvImageWrapper(const cv::Mat &cv_image, bool isRGB = false);
    const ImageDataInfo &GetImageDataInfo() const override;

private:
    IpipelineImageData::ImageDataInfo image_data_info;
    const cv::Mat                     inner_cv_image;
};
#endif //WRAPPER_H
