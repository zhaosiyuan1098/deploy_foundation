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
#include "wrapper.h"

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

    static std::shared_ptr<DetectionPipelinePackage> CreateDetectionPipelineUnit(
        const cv::Mat& input_image,
        float conf_thresh,
        bool isRGB,
        std::shared_ptr<infer_core::IBlobsBuffer> blob_buffers)
    {
        // 1. construct the image wrapper
        auto image_wrapper = std::make_shared<PipelineCvImageWrapper>(input_image, isRGB);
        // 2. construct `DetectionPipelinePackage`
        auto package = std::make_shared<DetectionPipelinePackage>();
        package->input_image_data = image_wrapper;
        package->conf_thresh = conf_thresh;
        package->infer_buffer = blob_buffers;

        return package;
    }


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


    inline BaseDetectionModel::BaseDetectionModel(std::shared_ptr<infer_core::BaseInferCore> infer_core): infer_core_(
        infer_core)
    {
        // 1. check infer_core
        if (infer_core == nullptr)
        {
            throw std::invalid_argument("[BaseDetectionModel] Input argument `infer_core` is nullptr!!!");
        }

        // 2. configure pipeline
        auto preprocess_block = BaseAsyncPipeline::buildPipelineBlock(
            [=](ParsingType unit) -> bool { return PreProcess(unit); }, "BaseDet PreProcess");

        auto infer_core_context = infer_core->getContext();

        auto postprocess_block = BaseAsyncPipeline::buildPipelineBlock(
            [=](ParsingType unit) -> bool { return PostProcess(unit); }, "BaseDet PostProcess");

        BaseAsyncPipeline::config(detection_pipeline_name_,
                                  {preprocess_block, infer_core_context, postprocess_block});
    }

    inline bool BaseDetectionModel::Detect(const cv::Mat& input_image, std::vector<BBox2D>& det_results,
                                           float conf_thresh, bool isRGB)
    {
        // 1. Get blobs buffer
        auto blob_buffers = infer_core_->GetBuffer(false);
        if (blob_buffers == nullptr)
        {
            LOG(ERROR) << "[BaseDetectionModel] Inference Core run out buffer!!!";
            return false;
        }

        // 2. Create a dummy pipeline package
        auto package = CreateDetectionPipelineUnit(input_image, conf_thresh, isRGB, blob_buffers);

        // 3. preprocess by derived class
        MEASURE_DURATION_AND_CHECK_STATE(PreProcess(package),
                                         "[BaseDetectionModel] Preprocess execute failed!!!");

        // 4. network inference
        MEASURE_DURATION_AND_CHECK_STATE(infer_core_->SyncInfer(blob_buffers),
                                         "[BaseDetectionModel] SyncInfer execute failed!!!");

        // 5. postprocess by derived class
        MEASURE_DURATION_AND_CHECK_STATE(PostProcess(package),
                                         "[BaseDetectionModel] PostProcess execute failed!!!");

        // 6. take output
        det_results = std::move(package->results);

        return true;
    }

    inline std::future<std::vector<BBox2D>> BaseDetectionModel::DetectAsync(const cv::Mat& input_image,
                                                                            float conf_thresh, bool isRGB,
                                                                            bool cover_oldest)
    {
        // 1. check if the pipeline is initialized
        if (!isInitialized(detection_pipeline_name_))
        {
            LOG(ERROR) << "[BaseDetectionModel] Async Pipeline is not init yet!!!";
            return std::future<std::vector<BBox2D>>();
        }

        // 2. get blob buffer
        auto blob_buffers = infer_core_->GetBuffer(true);
        if (blob_buffers == nullptr)
        {
            LOG(ERROR) << "[BaseDetectionModel] Failed to get buffer from inference core!!!";
            return std::future<std::vector<BBox2D>>();
        }

        // 3. create a pipeline package
        auto package = CreateDetectionPipelineUnit(input_image, conf_thresh, isRGB, blob_buffers);

        // 4. push package into pipeline and return `std::future`
        return push(detection_pipeline_name_, package);
    }

    inline BaseDetectionModel::~BaseDetectionModel()
    {
        close();
        infer_core_->Release();
    }
}


#endif //DETECTION_H
