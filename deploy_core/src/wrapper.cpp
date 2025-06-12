//
// Created by siyuan on 25-6-12.
//

#include "wrapper.h"

PipelineCvImageWrapper::PipelineCvImageWrapper(const cv::Mat& cv_image, bool isRGB)
: inner_cv_image(cv_image)
    {
        image_data_info.data_pointer   = cv_image.data;
        image_data_info.format         = isRGB ? ImageDataFormat::RGB : ImageDataFormat::BGR;
        image_data_info.image_height   = cv_image.rows;
        image_data_info.image_width    = cv_image.cols;
        image_data_info.image_channels = cv_image.channels();
        image_data_info.location       = DataLocation::HOST;
    }

const async_pipeline::IpipelineImageData::ImageDataInfo& PipelineCvImageWrapper::GetImageDataInfo() const
{
    return image_data_info;
}
