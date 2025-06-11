//
// Created by siyuan on 25-6-10.
//

#ifndef SAM_H
#define SAM_H

#include "infer_core.h"
#include "common.h"
#include"wrapper.h"

#include <opencv2/opencv.hpp>

namespace sam
{
    static bool CheckValidArguments(const cv::Mat                                        &image,
                                    const std::shared_ptr<infer_core::IInferCore> &infer_core,
                                    const std::vector<std::pair<int, int>>               &points,
                                    const std::vector<int> &labels) noexcept
    {
        if (image.empty())
        {
            LOG(ERROR) << "[BaseSamModel] Got empty image!!!";
            return false;
        } else if (infer_core == nullptr)
        {
            LOG(ERROR) << "[BaseSamModel] Infer_core with points as prompt is null!!!";
            return false;
        } else if (points.size() != labels.size() || points.size() < 1)
        {
            LOG(ERROR) << "[BaseSamModel] points/labels size is not valid!!! "
                       << "points.size: " << points.size() << ", labels.size: " << labels.size();
            return false;
        }

        return true;
    }


    static bool CheckValidArguments(const cv::Mat                                        &image,
                                const std::shared_ptr<infer_core::IInferCore> &infer_core,
                                const std::vector<BBox2D> &boxes) noexcept
    {
        if (image.empty())
        {
            LOG(ERROR) << "[BaseSamModel] Got empty image!!!";
            return false;
        } else if (infer_core == nullptr)
        {
            LOG(ERROR) << "[BaseSamModel] Infer_core with boxes as prompt is null!!!";
            return false;
        } else if (boxes.size() < 1)
        {
            LOG(ERROR) << "[BaseSamModel] boxes size is not valid!!! "
                       << "boxes.size: " << boxes.size();
            return false;
        } else if (boxes.size() > 1)
        {
            LOG(WARNING) << "[BaseSamModel] More than one boxes is not support in sam model!!";
        }

        return true;
    }

    struct SamPipelinePackage : public async_pipeline::IPipelinePackage
    {
        std::shared_ptr<infer_core::IBlobsBuffer> image_encoder_blobs_buffer;
        std::shared_ptr<infer_core::IBlobsBuffer> mask_decoder_blobs_buffer;
        std::shared_ptr<async_pipeline::IpipelineImageData> input_image_data;
        std::vector<BBox2D> boxes;
        std::vector<std::pair<int, int>> points;
        std::vector<int> labels;
        float transform_scale;
        cv::Mat mask;
        std::shared_ptr<infer_core::IBlobsBuffer> infer_buffer;

        std::shared_ptr<infer_core::IBlobsBuffer> GetInferBuffer() override
        {
            return infer_buffer;
        }
    };


    class SamGenResultType
    {
    public:
        cv::Mat operator()(const std::shared_ptr<async_pipeline::IPipelinePackage>& package) const
        {
            auto sam_package = std::dynamic_pointer_cast<SamPipelinePackage>(package);
            if (sam_package == nullptr)
            {
                LOG(ERROR) << "[SamGenResultType] Got INVALID package ptr!!!";
                return {};
            }
            return std::move(sam_package->mask);
        }
    };


    class ISamModel
    {
        typedef std::shared_ptr<async_pipeline::IPipelinePackage> ParsingType;

    public:
        virtual bool ImagePreProcess(ParsingType pipeline_unit) = 0;
        virtual bool PromptBoxPreProcess(ParsingType pipeline_unit) = 0;
        virtual bool PromptPointPreProcess(ParsingType pipeline_unit) = 0;
        virtual bool MaskPostProcess(ParsingType pipeline_unit) = 0;

    protected:
        virtual ~ISamModel() = default;
    };

    class BaseSamModel : public ISamModel,
                         public async_pipeline::BaseAsyncPipeline<cv::Mat, SamGenResultType>
    {
        using ParsingType = std::shared_ptr<async_pipeline::IPipelinePackage>;
    public:
        BaseSamModel(const std::string& model_name,
                     std::shared_ptr<infer_core::BaseInferCore> image_encoder_core,
                     std::shared_ptr<infer_core::BaseInferCore> mask_points_decoder_core,
                     std::shared_ptr<infer_core::BaseInferCore> mask_boxes_decoder_core);
        bool GenerateMask(const cv::Mat& image,
                          const std::vector<std::pair<int, int>>& points,
                          const std::vector<int>& labels,
                          cv::Mat& result,
                          bool isRGB = false);
        bool GenerateMask(const cv::Mat& image,
                          const std::vector<BBox2D>& boxes,
                          cv::Mat& result,
                          bool isRGB = false);
        [[nodiscard]] std::future<cv::Mat> GenerateMaskAsync(
            const cv::Mat& image,
            const std::vector<std::pair<int, int>>& points,
            const std::vector<int>& labels,
            bool isRGB = false,
            bool cover_oldest = false);
        [[nodiscard]] std::future<cv::Mat> GenerateMaskAsync(const cv::Mat& image,
                                                             const std::vector<BBox2D>& boxes,
                                                             bool isRGB = false,
                                                             bool cover_oldest = false);
        using BaseAsyncPipeline::push;
        void ConfigureBoxPipeline();
        void ConfigurePointPipeline();

    protected:
        ~BaseSamModel() override;

    private:
        std::shared_ptr<infer_core::BaseInferCore> image_encoder_core_;
        std::shared_ptr<infer_core::BaseInferCore> mask_points_decoder_core_;
        std::shared_ptr<infer_core::BaseInferCore> mask_boxes_decoder_core_;

        const std::string box_pipeline_name_;
        const std::string point_pipeline_name_;
        const std::string model_name_;
    };



    class BaseSamFactory
    {
    public:
        virtual std::shared_ptr<sam::BaseSamModel> Create() = 0;

    protected:
        virtual ~BaseSamFactory() = default;
    };

    inline BaseSamModel::BaseSamModel(const std::string& model_name,
        std::shared_ptr<infer_core::BaseInferCore> image_encoder_core,
        std::shared_ptr<infer_core::BaseInferCore> mask_points_decoder_core,
        std::shared_ptr<infer_core::BaseInferCore> mask_boxes_decoder_core):model_name_(model_name),
      image_encoder_core_(image_encoder_core),
      mask_points_decoder_core_(mask_points_decoder_core),
      mask_boxes_decoder_core_(mask_boxes_decoder_core),
      box_pipeline_name_(model_name + "_SamWithBoxPipeline"),
      point_pipeline_name_(model_name + "_SamWithPointPipeline")
    {
        if (image_encoder_core == nullptr)
        {
            throw std::invalid_argument("`image_encoder_core` should not be null");
        }

        if (mask_points_decoder_core == nullptr && mask_boxes_decoder_core == nullptr)
        {
            throw std::invalid_argument("one of `point/box` decoder should be non-nullptr");
        }

        if (mask_points_decoder_core_ != nullptr)
        {
            ConfigurePointPipeline();
        }
        if (mask_boxes_decoder_core_ != nullptr)
        {
            ConfigureBoxPipeline();
        }
    }

    inline bool BaseSamModel::GenerateMask(const cv::Mat& image, const std::vector<std::pair<int, int>>& points,
        const std::vector<int>& labels, cv::Mat& result, bool isRGB)
    {
        // 0. check
        CHECK_STATE(CheckValidArguments(image, mask_points_decoder_core_, points, labels),
                    "[BaseSamModel] `GenerateMask` with points got invalid arguments");

        // 1. Get blobs buffers
        auto encoder_blob_buffers = image_encoder_core_->GetBuffer(true);
        auto decoder_blob_buffers = mask_points_decoder_core_->GetBuffer(true);

        // 2. Construct `SamPipelinePackage`
        auto package                        = std::make_shared<SamPipelinePackage>();
        package->input_image_data           = std::make_shared<PipelineCvImageWrapper>(image, isRGB);
        package->points                     = points;
        package->labels                     = labels;
        package->image_encoder_blobs_buffer = encoder_blob_buffers;
        package->mask_decoder_blobs_buffer  = decoder_blob_buffers;

        // 3. Carry out workflow
        MEASURE_DURATION_AND_CHECK_STATE(ImagePreProcess(package),
                                         "[BaseSamModel] Image-Preprocess execute failed!!!");

        MEASURE_DURATION_AND_CHECK_STATE(image_encoder_core_->SyncInfer(package->GetInferBuffer()),
                                         "[BaseSamModel] Image-encoder sync infer execute failed!!!");

        MEASURE_DURATION_AND_CHECK_STATE(PromptPointPreProcess(package),
                                         "[BaseSamModel] Prompt-preprocess execute failed!!!");

        MEASURE_DURATION_AND_CHECK_STATE(mask_points_decoder_core_->SyncInfer(package->GetInferBuffer()),
                                         "[BaseSamModel] Prompt-decoder sync infer execute failed!!!");

        MEASURE_DURATION_AND_CHECK_STATE(MaskPostProcess(package),
                                         "[BaseSamModel] Mask-postprocess execute failed!!!");

        // 4. output the result
        result = package->mask;
        return true;
    }

    inline bool BaseSamModel::GenerateMask(const cv::Mat& image, const std::vector<BBox2D>& boxes, cv::Mat& result,
        bool isRGB)
    {
        // 0. check
        CHECK_STATE(CheckValidArguments(image, mask_boxes_decoder_core_, boxes),
                    "[BaseSamModel] `GenerateMask` with boxes got invalid arguments");

        // 1. Get blobs buffers
        auto encoder_blob_buffers = image_encoder_core_->GetBuffer(true);
        auto decoder_blob_buffers = mask_boxes_decoder_core_->GetBuffer(true);

        // 2. Construct `SamPipelinePackage`
        auto package                        = std::make_shared<SamPipelinePackage>();
        package->input_image_data           = std::make_shared<PipelineCvImageWrapper>(image, isRGB);
        package->boxes                      = boxes;
        package->image_encoder_blobs_buffer = encoder_blob_buffers;
        package->mask_decoder_blobs_buffer  = decoder_blob_buffers;

        // 3. Carry out workflow
        MEASURE_DURATION_AND_CHECK_STATE(ImagePreProcess(package),
                                         "[BaseSamModel] Image-Preprocess execute failed!!!");

        MEASURE_DURATION_AND_CHECK_STATE(image_encoder_core_->SyncInfer(package->GetInferBuffer()),
                                         "[BaseSamModel] Image-encoder sync infer execute failed!!!");

        MEASURE_DURATION_AND_CHECK_STATE(PromptBoxPreProcess(package),
                                         "[BaseSamModel] Prompt-preprocess execute failed!!!");

        MEASURE_DURATION_AND_CHECK_STATE(mask_boxes_decoder_core_->SyncInfer(package->GetInferBuffer()),
                                         "[BaseSamModel] Prompt-decoder sync infer execute failed!!!");

        MEASURE_DURATION_AND_CHECK_STATE(MaskPostProcess(package),
                                         "[BaseSamModel] Mask-postprocess execute failed!!!");

        // 4. output the result
        result = package->mask;
        return true;
    }

    inline std::future<cv::Mat> BaseSamModel::GenerateMaskAsync(const cv::Mat& image,
        const std::vector<std::pair<int, int>>& points, const std::vector<int>& labels, bool isRGB, bool cover_oldest)
    {
        // 0. Check
        if (!CheckValidArguments(image, mask_points_decoder_core_, points, labels))
        {
            LOG(ERROR) << "[BaseSamModel] `GenerateMask` with points got invalid arguments";
            return std::future<cv::Mat>();
        }
        if (!BaseAsyncPipeline::isInitialized(point_pipeline_name_))
        {
            LOG(ERROR) << "[BaseSamModel] Async pipeline with points as prompt is not initialized yet!!!";
            return std::future<cv::Mat>();
        }

        // 1. Get blobs buffers
        auto encoder_blob_buffers = image_encoder_core_->GetBuffer(true);
        auto decoder_blob_buffers = mask_points_decoder_core_->GetBuffer(true);

        // 2. Construct `SamPipelinePackage`
        auto package                        = std::make_shared<SamPipelinePackage>();
        package->input_image_data           = std::make_shared<PipelineCvImageWrapper>(image, isRGB);
        package->points                     = points;
        package->labels                     = labels;
        package->image_encoder_blobs_buffer = encoder_blob_buffers;
        package->mask_decoder_blobs_buffer  = decoder_blob_buffers;

        // 3. return `std::future` instance
        return BaseAsyncPipeline::push(point_pipeline_name_, package);
    }

    inline std::future<cv::Mat> BaseSamModel::GenerateMaskAsync(const cv::Mat& image, const std::vector<BBox2D>& boxes,
        bool isRGB, bool cover_oldest)
    {
        // 0. check
        if (!CheckValidArguments(image, mask_boxes_decoder_core_, boxes))
        {
            LOG(ERROR) << "[BaseSamModel] `GenerateMask` with boxes got invalid arguments";
            return std::future<cv::Mat>();
        }

        if (!BaseAsyncPipeline::isInitialized(box_pipeline_name_))
        {
            LOG(ERROR) << "[BaseSamModel] Async pipeline with boxes as prompt is not initialized yet!!!";
            return std::future<cv::Mat>();
        }

        // 1. Get blobs buffers
        auto encoder_blob_buffers = image_encoder_core_->GetBuffer(true);
        auto decoder_blob_buffers = mask_boxes_decoder_core_->GetBuffer(true);

        // 2. Construct `SamPipelinePackage`
        auto package                        = std::make_shared<SamPipelinePackage>();
        package->input_image_data           = std::make_shared<PipelineCvImageWrapper>(image, isRGB);
        package->boxes                      = boxes;
        package->image_encoder_blobs_buffer = encoder_blob_buffers;
        package->mask_decoder_blobs_buffer  = decoder_blob_buffers;

        // 3. return `std::future` instance
        return BaseAsyncPipeline::push(box_pipeline_name_, package);
    }

    inline void BaseSamModel::ConfigureBoxPipeline()
    {
        auto image_preprocess_block = BaseAsyncPipeline::buildPipelineBlock(
      [&](ParsingType unit) -> bool { return ImagePreProcess(unit); },
      "[MobileSam Image PreProcess]");

        auto prompt_preprocess_block = BaseAsyncPipeline::buildPipelineBlock(
            [&](ParsingType unit) -> bool { return PromptBoxPreProcess(unit); },
            "[MobileSam Prompt PreProcess]");

        auto mask_postprocess_block = BaseAsyncPipeline::buildPipelineBlock(
            [&](ParsingType unit) -> bool { return MaskPostProcess(unit); },
            "[MobileSam Mask PostProcess]");

        const auto &image_encoder_context = image_encoder_core_->getContext();

        const auto &mask_decoder_context = mask_boxes_decoder_core_->getContext();

        BaseAsyncPipeline::config(
            box_pipeline_name_, {image_preprocess_block, image_encoder_context, prompt_preprocess_block,
                                 mask_decoder_context, mask_postprocess_block});
    }

    inline void BaseSamModel::ConfigurePointPipeline()
    {
        auto image_preprocess_block = BaseAsyncPipeline::buildPipelineBlock(
     [&](ParsingType unit) -> bool { return ImagePreProcess(unit); },
     "[MobileSam Image PreProcess]");

        auto prompt_preprocess_block = BaseAsyncPipeline::buildPipelineBlock(
            [&](ParsingType unit) -> bool { return PromptPointPreProcess(unit); },
            "[MobileSam Prompt PreProcess]");

        auto mask_postprocess_block = BaseAsyncPipeline::buildPipelineBlock(
            [&](ParsingType unit) -> bool { return MaskPostProcess(unit); },
            "[MobileSam Mask PostProcess]");

        const auto &image_encoder_context = image_encoder_core_->getContext();

        const auto &mask_decoder_context = mask_points_decoder_core_->getContext();

        BaseAsyncPipeline::config(
            point_pipeline_name_, {image_preprocess_block, image_encoder_context, prompt_preprocess_block,
                                   mask_decoder_context, mask_postprocess_block});
    }

    inline BaseSamModel::~BaseSamModel()
    {
        BaseAsyncPipeline::close();

        if (image_encoder_core_ != nullptr)
        {
            image_encoder_core_->Release();
        }
        if (mask_points_decoder_core_ != nullptr)
        {
            mask_points_decoder_core_->Release();
        }
        if (mask_boxes_decoder_core_ != nullptr)
        {
            mask_boxes_decoder_core_->Release();
        }
    }
}
#endif //SAM_H
