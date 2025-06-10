//
// Created by siyuan on 25-6-10.
//

#ifndef BLOB_BUFFER_H
#define BLOB_BUFFER_H
#include <utility>
#include "common.h"
#include <string>
#include <vector>

namespace infer_core
{
    class IBlobBuffer
    {
    public:
        virtual std::pair<void*, DataLocation> getOuterBlobBuffer(const std::string& blob_name) =0;
        virtual bool setBlobBuffer(const std::string& blob_name, void* data_ptr, DataLocation location) =0;
        virtual bool setBlobBuffer(const std::string& blob_name, DataLocation location) =0;
        virtual bool setBlobShape(const std::string& blob_name, const std::vector<int64_t>& shape) =0;
        virtual const std::vector<int64_t>& getBlobShape(const std::string& blob_name) const =0;
        virtual size_t size() =0;
        virtual void reset() =0;

    protected:
        virtual ~IBlobBuffer() = default;
        virtual void release() =0;
    };
}

#endif //BLOB_BUFFER_H
