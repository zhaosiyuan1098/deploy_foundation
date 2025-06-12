//
// Created by siyuan on 25-6-11.
//

#include "tensorrt_blob_buffer.h"
#include "glog/logging.h"
#include "common.h"
#include <cuda_runtime.h>


namespace infer_core
{
    std::pair<void*, DataLocation> TrtBlobBuffer::getOuterBlobBuffer(const std::string& blob_name)
    {
        if (outer_map_blob2ptr_.find(blob_name) == outer_map_blob2ptr_.end())
        {
            LOG(ERROR) << "[TrtBlobBuffer] `GetOuterBlobBuffer` Got invalid `blob_name`: " << blob_name;
            return {nullptr, DataLocation::UNKNOWN};
        }
        return outer_map_blob2ptr_[blob_name];
    }

    bool TrtBlobBuffer::setBlobBuffer(const std::string& blob_name, void* data_ptr, DataLocation location)
    {
        if (outer_map_blob2ptr_.find(blob_name) == outer_map_blob2ptr_.end())
        {
            LOG(ERROR) << "[TrtBlobBuffer] `SetBlobBuffer` Got invalid `blob_name`: " << blob_name;
            return false;
        }

        if (location == DataLocation::HOST)
        {
            outer_map_blob2ptr_[blob_name] = {inner_map_host_blob2ptr_[blob_name], location};
        }
        else
        {
            cudaPointerAttributes attr;
            cudaError_t status = cudaPointerGetAttributes(&attr, data_ptr);
            if (status != cudaSuccess || attr.type != cudaMemoryType::cudaMemoryTypeDevice)
            {
                LOG(ERROR) << "[TrtBlobBuffer] `SetBlobBuffer` Got "
                    "invalid `data_ptr` "
                    "which should be "
                    << "allocated by `cudaMalloc`, but it "
                    "is NOT !!!";
                return false;
            }
            outer_map_blob2ptr_[blob_name] = {data_ptr, location};
        }
        return true;
    }

    bool TrtBlobBuffer::setBlobBuffer(const std::string& blob_name, DataLocation location)
    {
        if (outer_map_blob2ptr_.find(blob_name) == outer_map_blob2ptr_.end())
        {
            LOG(ERROR) << "[TrtBlobBuffer] `SetBlobBuffer` Got invalid `blob_name`: " << blob_name;
            return false;
        }

        outer_map_blob2ptr_[blob_name] = {
            (location == DataLocation::HOST
                 ? inner_map_host_blob2ptr_[blob_name]
                 : inner_map_device_blob2ptr_[blob_name]),
            location
        };

        return true;
    }

    bool TrtBlobBuffer::setBlobShape(const std::string& blob_name, const std::vector<int64_t>& shape)
    {
        if (map_blob_name2shape_.find(blob_name) == map_blob_name2shape_.end())
        {
            LOG(ERROR) << "[TrtBlobBuffer] `SetBlobShape` Got invalid `blob_name`: " << blob_name;
            return false;
        }
        const auto& origin_shape = map_blob_name2shape_[blob_name];
        const long long ori_element_count = CumVector(origin_shape);
        const long long dyn_element_count = CumVector(shape);
        if (origin_shape.size() != shape.size() || dyn_element_count > ori_element_count ||
            dyn_element_count < 0)
        {
            const std::string origin_shape_in_str = VisualVec(origin_shape);
            const std::string shape_in_str = VisualVec(shape);
            LOG(ERROR) << "[TrtBlobBuffer] `SetBlobShape` Got invalid `shape` input. "
                << "`shape`: " << shape_in_str << "\t"
                << "`origin_shape`: " << origin_shape_in_str;
            return false;
        }
        map_blob_name2shape_[blob_name] = shape;
        return true;
    }

    const std::vector<int64_t>& TrtBlobBuffer::getBlobShape(const std::string& blob_name) const
    {
        if (map_blob_name2shape_.find(blob_name) == map_blob_name2shape_.end())
        {
            LOG(ERROR) << "[TrtBlobBuffer] `GetBlobShape` Got invalid `blob_name`: " << blob_name;
            static std::vector<int64_t> empty_shape;
            return empty_shape;
        }
        return map_blob_name2shape_.at(blob_name);
    }

    size_t TrtBlobBuffer::size()
    {
        return outer_map_blob2ptr_.size();
    }

    void TrtBlobBuffer::release()
    {
        // release device buffer
        for (void* ptr : device_blobs_buffer_)
        {
            if (ptr != nullptr)
                cudaFree(ptr);
        }
        // release host buffer
        for (void* ptr : host_blobs_buffer_)
        {
            if (ptr != nullptr)
                delete[] reinterpret_cast<u_char*>(ptr);
        }
        device_blobs_buffer_.clear();
        host_blobs_buffer_.clear();
    }

    void TrtBlobBuffer::reset()
    {
        for (const auto& p_name_ptr : inner_map_host_blob2ptr_)
        {
            outer_map_blob2ptr_[p_name_ptr.first] = {p_name_ptr.second, DataLocation::HOST};
        }
    }

    TrtBlobBuffer::~TrtBlobBuffer()
    {
        // release device buffer
        for (void* ptr : device_blobs_buffer_)
        {
            if (ptr != nullptr)
                cudaFree(ptr);
        }
        // release host buffer
        for (void* ptr : host_blobs_buffer_)
        {
            if (ptr != nullptr)
                delete[] reinterpret_cast<u_char*>(ptr);
        }
        device_blobs_buffer_.clear();
    }
}
