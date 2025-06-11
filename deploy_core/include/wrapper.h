//
// Created by root on 25-6-11.
//

#ifndef WRAPPER_H
#define WRAPPER_H

#include "async_pipeline.h"

#include <opencv2/opencv.hpp>

class PipelineCvImageWrapper : public async_pipeline::IpipelineImageData {
public:
    PipelineCvImageWrapper(const cv::Mat &cv_image, bool isRGB = false) : inner_cv_image(cv_image)
    {
        image_data_info.data_pointer   = cv_image.data;
        image_data_info.format         = isRGB ? ImageDataFormat::RGB : ImageDataFormat::BGR;
        image_data_info.image_height   = cv_image.rows;
        image_data_info.image_width    = cv_image.cols;
        image_data_info.image_channels = cv_image.channels();
        image_data_info.location       = DataLocation::HOST;
    }

    const ImageDataInfo &GetImageDataInfo() const override
    {
        return image_data_info;
    }

private:
    IpipelineImageData::ImageDataInfo image_data_info;
    const cv::Mat                     inner_cv_image;
};
#endif //WRAPPER_H
