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


}
#endif //SAM_H
