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
    struct DetectionPipelinePackage : public async_pipeline::IPipelinePackage
    {
        // the wrapped pipeline image data
        std::shared_ptr<async_pipeline::IpipelineImageData> input_image_data;
        // confidence used in postprocess
        float conf_thresh;
        // record the transform factor during image preprocess
        float transform_scale;
        // the detection result
        std::vector<BBox2D> results;

        // maintain the blobs buffer instance
        std::shared_ptr<infer_core::IBlobsBuffer> infer_buffer;

        // override from `IPipelinePackage`, to provide the blobs buffer to inference_core
        std::shared_ptr<infer_core::IBlobsBuffer> GetInferBuffer() override
        {
            if (infer_buffer == nullptr)
            {
                LOG(ERROR) << "[DetectionPipelinePackage] returned nullptr of infer_buffer!!!";
            }
            return infer_buffer;
        }
    };


    class IDetectionPreProcess
    {
    public:
        virtual float Preprocess(std::shared_ptr<async_pipeline::IpipelineImageData> input_image_data,
                                 std::shared_ptr<infer_core::IBlobsBuffer> blob_buffer,
                                 const std::string& blob_name,
                                 int dst_height,
                                 int dst_width) = 0;
    protected:
        virtual ~IDetectionPreProcess() = default;
    };


    class IDetectionPostProcess
    {
    public:
        virtual void Postprocess(const std::vector<void*>& output_blobs_ptr,
                                 std::vector<BBox2D>& results,
                                 float conf_threshold,
                                 float transform_scale) = 0;
    protected:
        virtual ~IDetectionPostProcess() = default;

    };

    class IDetectionModel
    {
    public:
        IDetectionModel() = default;

    protected:
        virtual ~IDetectionModel() = default;
        virtual bool PreProcess(std::shared_ptr<async_pipeline::IPipelinePackage> pipeline_unit) = 0;
        virtual bool PostProcess(std::shared_ptr<async_pipeline::IPipelinePackage> pipeline_unit) = 0;
    };

    class DetectionGenResultType
    {
    public:
        std::vector<BBox2D> operator()(const std::shared_ptr<async_pipeline::IPipelinePackage>& package) const
        {
            auto detection_package = std::dynamic_pointer_cast<DetectionPipelinePackage>(package);
            if (detection_package == nullptr)
            {
                LOG(ERROR) << "[DetectionGenResult] Got INVALID package ptr!!!";
                return {};
            }
            return std::move(detection_package->results);
        }
    };





    class BaseDetectionModel
        : public IDetectionModel,
          public async_pipeline::BaseAsyncPipeline<std::vector<BBox2D>, DetectionGenResultType>
    {
        typedef std::shared_ptr<async_pipeline::IPipelinePackage> ParsingType;

    public:
        BaseDetectionModel(std::shared_ptr<infer_core::BaseInferCore> infer_core);
        bool Detect(const cv::Mat& input_image,
                    std::vector<BBox2D>& det_results,
                    float conf_thresh,
                    bool isRGB = false);
        [[nodiscard]] std::future<std::vector<BBox2D>> DetectAsync(const cv::Mat& input_image,
                                                                   float conf_thresh,
                                                                   bool isRGB = false,
                                                                   bool cover_oldest = false);

    protected:
        using BaseAsyncPipeline::push;
        ~BaseDetectionModel() override;
        std::shared_ptr<infer_core::BaseInferCore> infer_core_{nullptr};
        static std::string detection_pipeline_name_;
    };

    class BaseDetection2DFactory
    {
    public:
        virtual std::shared_ptr<detection::BaseDetectionModel> Create() = 0;
    protected:
        virtual ~BaseDetection2DFactory() = default;
    };

    class BaseDetectionPreprocessFactory
    {
    public:
        virtual std::shared_ptr<detection::IDetectionPreProcess> Create() = 0;
    protected:
        virtual ~BaseDetectionPreprocessFactory() = default;
    };

    class BaseDetectionPostprocessFactory
    {
    public:
        virtual std::shared_ptr<detection::IDetectionPostProcess> Create() = 0;
    protected:
        virtual ~BaseDetectionPostprocessFactory() = default;
    };
}


#endif //DETECTION_H
