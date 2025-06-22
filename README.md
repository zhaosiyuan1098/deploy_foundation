# deploy_foundation

## 介绍
* 基于C++20 和CUDA 技术栈构建为FoundationPose 模型设计的高性能6D 物体位姿估计部署方案。
* 通过集成 NVIDIA TensorRT 实现模型推理加速。
* 核心架构是一个模块化的异步处理流水线，深度融合基于GPU 的渲染（nvdiffrast）
与视觉处理（CV-CUDA）技术，从RGB-D 数据流中实现高效率、低延迟的物体初始位姿解算与连续跟踪。
## 结构
原算法实现流程如下
![](/pic/all.png)

故使用refiner和scorer，分别完成：

* 使用nvdiffrast并行渲染生成多候选位姿的3维模型与2维图片 

* 使用encoder架构完成最终位姿选择

本项目的代码采用了高度模块化的设计，将核心框架、推理引擎和算法实现进行了解耦，便于维护和扩展。
```
├── deploy_core                  # 核心部署库，提供通用工具
├── tensorrt_core                # TensorRT推理核心封装
├── foundationpose_core  # FoundationPose算法核心实现
├── tests                 # GTest单元测试和性能基准测试
└── foundationpose               # (占位，用于未来扩展)
```

1. **deploy_core**

    这是整个框架的基石

   * 异步流水线 `async_pipeline`: 提供了一个通用的、多阶段的异步处理流水线 `BaseAsyncPipeline`。它允许数据在预处理、推理、后处理等阶段之间以非阻塞的方式流动，从而最大化GPU和CPU的并行效率。
   * 数据结构 `common.h`, `blob_buffer.h`: 定义了项目中通用的数据结构，如2D包围盒BBox2D、数据位置DataLocation以及推理缓冲区接口`IBlobsBuffer`。
2. **tensorrt_core**

   该模块负责与NVIDIA TensorRT引擎的交互

    * 推理核心 `TrtInferCore`: 封装了TensorRT的C++ API，负责加载.engine文件，管理`IExecutionContext`（为保证线程安全，会为每个线程创建独立的上下文），并执行实际的神经网络前向传播。
    * 内存管理 `TrtBlobBuffer`: 实现了`IBlobsBuffer`接口，用于管理推理所需的GPU和CPU内存，并支持动态输入形状（dynamic shape）的设置。

3. **foundationpose_core**

   `FoundationPose`算法的C++和CUDA实现，也是项目的核心业务逻辑所在

   * 主控类 `FoundationPose`: 整个算法流程的调度中心。它整合了采样、渲染和解码等模块，并实现了Register（首次检测）和Track（连续追踪）两个主要的对外接口。
   * 位姿采样器 `FoundationPoseSampler`: 负责在Register模式下，根据输入的深度图和掩码，生成一系列初始的位姿假设（Hypotheses）。
   * 渲染器 `FoundationPoseRenderer`: 项目中最关键和复杂的部分之一。它利用CUDA和nvdiffrast库，为每一个位姿假设高速渲染出对应的3D模型视图（“渲染图”），并从真实图像中裁剪出相应区域（“真实图”）。这两部分数据将被拼接后送入神经网络。
   * 解码器 `FoundationPoseDecoder`: 负责处理Scorer模型的输出。它会在GPU上快速找出得分最高的位姿索引，并进行最终的坐标变换，得到可用的输出位姿。


## 使用方法

### 环境依赖
* NVIDIA CUDA Toolkit
* NVIDIA TensorRT
* OpenCV >= 4.0
* Eigen3
* Google Glog
* Google Test (用于编译和运行测试)
* CVCUDA, NVCV (NVIDIA VPI的一部分)

### 编译
```angular2html
mkdir build
cd build
cmake ..
make -j$(nproc)
```

### 调用实例
代码逻辑参考[](/tests/src/test_foundationpose.cpp)

```c++
#include "foundationpose.hpp"
#include "tensorrt_infer_core.h"
#include "help_func.hpp"

// 1. 创建Refiner和Scorer的TensorRT推理核心
auto refiner_core = infer_core::CreateTrtInferCore("refiner.engine", ...);
auto scorer_core = infer_core::CreateTrtInferCore("scorer.engine", ...);

// 2. 读取相机内参和3D模型路径
Eigen::Matrix3f intrinsic = ReadCamK("cam_K.txt");
std::string mesh_path = "path/to/mesh.obj";
std::string texture_path = "path/to/texture.png";
std::string target_name = "my_object";

// 3. 创建FoundationPose模型实例
auto foundation_pose_model = detection_6d::CreateFoundationPoseModel(
    refiner_core,
    scorer_core,
    target_name,
    mesh_path,
    texture_path,
    intrinsic
);

// 4. 读取第一帧的图像数据
auto [rgb, depth, mask] = ReadRgbDepthMask("rgb0.png", "depth0.png", "mask0.png");

// 5. 使用Register进行首次位姿估计
Eigen::Matrix4f detected_pose;
foundation_pose_model->Register(rgb, depth, mask, target_name, detected_pose);
std::cout << "Detected Pose: \n" << detected_pose << std::endl;

// 6. 对于后续帧，使用Track进行高效追踪
auto [next_rgb, next_depth] = ReadRgbDepth("rgb1.png", "depth1.png");
Eigen::Matrix4f tracked_pose;
foundation_pose_model->Track(next_rgb, next_depth, target_name, tracked_pose);
std::cout << "Tracked Pose: \n" << tracked_pose << std::endl;

```
## 效果展示
运行simple_tests中的测试用例可以验证算法效果和性能。

测试代码test_foundationpose.cpp中包含了完整的调用流程。

该测试会：

1. 加载一个测试物体的RGB、深度图和Mask。
2. 调用`Register()`进行初始化。
3. 加载后续视频帧，并循环调用`Track()`进行追踪。
4. 使用`draw3DBoundingBox`函数将计算出的6D位姿（表现为一个3D包围盒）绘制在原始RGB图像上。
5. 将处理后的图像帧保存为图片序列，并最终合成为一个MP4视频文件`test_foundationpose_result.mp4`。

可以运行测试，并在指定的输出路径查看带有3D包围盒的位姿估计结果视频。

## 参考

https://github.com/NVlabs/FoundationPose

https://github.com/NVlabs/nvdiffrast

https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/foundationpose