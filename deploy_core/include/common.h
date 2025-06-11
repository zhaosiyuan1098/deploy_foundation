//
// Created by root on 25-6-10.
//

#ifndef COMMON_H
#define COMMON_H

struct BBox2D {
    float x;
    float y;
    float w;
    float h;
    float conf;
    float cls;
};

enum DataLocation { HOST = 0, DEVICE = 1, UNKNOWN = 2 };

enum ImageDataFormat { YUV = 0, RGB = 1, BGR = 2, GRAY = 3 };

enum InferCoreType { ONNXRUNTIME, TENSORRT, RKNN, NOT_PROVIDED };

#define CHECK_STATE(state, hint) \
{                              \
if (!(state))                \
{                            \
LOG(ERROR) << (hint);      \
return false;              \
}                            \
}

#define MEASURE_DURATION(run)                                                                \
{                                                                                          \
auto start = std::chrono::high_resolution_clock::now();                                  \
(run);                                                                                   \
auto end = std::chrono::high_resolution_clock::now();                                    \
LOG(INFO) << #run << " cost(us): "                                                       \
<< std::chrono::duration_cast<std::chrono::microseconds>(end - start).count(); \
}

#define MEASURE_DURATION_AND_CHECK_STATE(run, hint)                                          \
{                                                                                          \
auto start = std::chrono::high_resolution_clock::now();                                  \
CHECK_STATE((run), hint);                                                                \
auto end = std::chrono::high_resolution_clock::now();                                    \
LOG(INFO) << #run << " cost(us): "                                                       \
<< std::chrono::duration_cast<std::chrono::microseconds>(end - start).count(); \
}

#endif //COMMON_H
