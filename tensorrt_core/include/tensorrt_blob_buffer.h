//
// Created by root on 25-6-11.
//

#ifndef TENSORRT_BLOB_BUFFER_H
#define TENSORRT_BLOB_BUFFER_H

#include "blob_buffer.h"
#include <unordered_map>

namespace infer_core
{

    template <typename Type>
    Type CumVector(const std::vector<Type> &vec)
    {
        Type ret = 1;
        for (const auto &nn : vec)
        {
            ret *= nn;
        }

        return ret;
    }

    template <typename Type>
    std::string VisualVec(const std::vector<Type> &vec)
    {
        std::string ret;
        for (const auto &v : vec)
        {
            ret += std::to_string(v) + " ";
        }
        return ret;
    }


    class TrtBlobBuffer : public IBlobsBuffer
    {
    public:
        std::pair<void*, DataLocation> getOuterBlobBuffer(const std::string& blob_name) override;
        bool setBlobBuffer(const std::string& blob_name, void* data_ptr, DataLocation location) override;
        bool setBlobBuffer(const std::string& blob_name, DataLocation location) override;
        bool setBlobShape(const std::string& blob_name, const std::vector<int64_t>& shape) override;
        const std::vector<int64_t>& getBlobShape(const std::string& blob_name) const override;
        size_t size() override;
        void release() override;
        void reset() override;
        ~TrtBlobBuffer() override;
        TrtBlobBuffer() = default;
        TrtBlobBuffer(const TrtBlobBuffer&) = delete;
        TrtBlobBuffer& operator=(const TrtBlobBuffer&) = delete;


        // mapping blob_name and buffer ptrs
        std::unordered_map<std::string, std::pair<void*, DataLocation>> outer_map_blob2ptr_;
        std::unordered_map<std::string, void*> inner_map_device_blob2ptr_;
        std::unordered_map<std::string, void*> inner_map_host_blob2ptr_;

        // buffer ptr vector, used while doing inference with tensorrt engine
        std::vector<void*> buffer_input_core_;

        // maintain buffer ptrs.
        std::vector<void*> device_blobs_buffer_;
        std::vector<void*> host_blobs_buffer_;

        // mapping blob_name and dynamic blob shape
        std::unordered_map<std::string, std::vector<int64_t>> map_blob_name2shape_;
    };
}

#endif //TENSORRT_BLOB_BUFFER_H
