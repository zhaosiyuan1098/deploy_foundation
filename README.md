# deploy_foundation

## 介绍
* 基于C++20和CUDA构建视觉大模型推理框架
* 为FoundationPose模型实现部署与推理加速
* 集成 NVIDIA TensorRT


### 与原始模型对比， 第一帧推理加速**9.84**倍，后续帧跟踪加速**14.61**倍。

<div align="center">
  <img src="https://github.com/user-attachments/assets/d5413ee9-d2d7-41f0-8312-6ed1e566160f" width="400" height="300" />
</div>

## 结构
原算法实现流程如下
![](/pic/all.png)


### 模型

#### 输入
* RGB 图像：可支持任何输入分辨率，图像无需任何额外的预处理（例如alpha通道或位深），格式为B X 3 X H X W (批量大小 x 通道 x 高度 x 宽度)
* 深度图像: 深度值。可支持任何分辨率，图像无需任何额外的预处理，格式为B X H X W (批量大小 x 高度 x 宽度)
* CAD 模型: CAD模型为OBJ格式，并在同一文件夹下包含纹理PNG图像，格式为 OBJ 文件
* 相机内参矩阵: 输入需要正确的相机标定信息，包括主点和焦距。格式为txt文件，
* 2D 边界框: 目标物体在第一帧中的坐标，坐标格式为xyxy，格式为B X 1 X 4 (批量大小 X 1 X 边界框坐标)

### 代码
本项目的代码采用了高度模块化的设计，将核心框架、推理引擎和算法实现进行了解耦，便于维护和扩展。
```
├── deploy_core                  # 核心部署库，提供通用工具
├── tensorrt_core                # TensorRT推理核心封装
├── foundationpose_core  # FoundationPose算法核心实现
├── tests                 # GTest单元测试和性能基准测试
└── foundationpose               # (占位，用于未来扩展)
```

1. **deploy_core**

    基本框架

   * 异步流水线 `async_pipeline`: 提供了一个通用的、多阶段的异步处理流水线 `BaseAsyncPipeline`。它允许数据在预处理、推理、后处理等阶段之间以非阻塞的方式流动，从而最大化GPU和CPU的并行效率。
   * 数据结构 `common.h`, `blob_buffer.h`: 定义了项目中通用的数据结构，如2D包围盒BBox2D、数据位置DataLocation以及推理缓冲区接口`IBlobsBuffer`。
2. **tensorrt_core**

   与NVIDIA TensorRT引擎的交互

    * 推理核心 `TrtInferCore`: 封装了TensorRT的C++ API，负责加载.engine文件，管理`IExecutionContext`（为保证线程安全，会为每个线程创建独立的上下文），并执行实际的神经网络前向传播。
    * 内存管理 `TrtBlobBuffer`: 实现了`IBlobsBuffer`接口，用于管理推理所需的GPU和CPU内存，并支持动态输入形状（dynamic shape）的设置。

3. **foundationpose_core**

   项目核心，`FoundationPose`算法的C++和CUDA实现

   * 主控类 `FoundationPose`: 整个算法流程的调度中心。它整合了采样、渲染和解码等模块，并实现了Register（首次检测）和Track（连续追踪）两个主要的对外接口。
   * 位姿采样器 `FoundationPoseSampler`: 负责在Register模式下，根据输入的深度图和掩码，生成一系列初始的位姿假设（Hypotheses）。
   * 渲染器 `FoundationPoseRenderer`: 项目中最关键和复杂的部分之一。它利用CUDA和nvdiffrast库，为每一个位姿假设高速渲染出对应的3D模型视图（“渲染图”），并从真实图像中裁剪出相应区域（“真实图”）。这两部分数据将被拼接后送入神经网络。
   * 解码器 `FoundationPoseDecoder`: 负责处理Scorer模型的输出。它会在GPU上快速找出得分最高的位姿索引，并进行最终的坐标变换，得到可用的输出位姿。


## 使用方法

### 运行环境依赖
* NVIDIA CUDA Toolkit
* NVIDIA TensorRT
* OpenCV >= 4.0
* Eigen3
* Google Glog
* Google Test (用于编译和运行测试)
* CVCUDA, NVCV (NVIDIA VPI的一部分)

### 模型导出

1. 在[Nvidia NGC论坛](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/foundationpose)下载官方开源ONNX模型
2. 使用nvidia TensorRT完成模型转换
   1. 安装 [TensorRT](https://developer.nvidia.com/tensorrt)
   2. ```bash
      //转换scorer模型
      /usr/src/tensorrt/bin/trtexec --onnx=YOUR_PATH/scorer_net.onnx \
      --minShapes=render_input:1x160x160x6,transf_input:1x160x160x6 \
      --optShapes=render_input:252x160x160x6,transf_input:252x160x160x6 \
      --maxShapes=render_input:252x160x160x6,transf_input:252x160x160x6 \
      --fp16 \
      --saveEngine=YOUR_PATH/scorer_hwc_dynamic_fp16.engine
      ```
   3. ```bash
      //转换refiner模型
      /usr/src/tensorrt/bin/trtexec --onnx=YOUR_PATH/refiner_net.onnx \
      --minShapes=render_input:1x160x160x6,transf_input:1x160x160x6 \
      --optShapes=render_input:252x160x160x6,transf_input:252x160x160x6 \
      --maxShapes=render_input:252x160x160x6,transf_input:252x160x160x6 \
      --fp16 \
      --saveEngine=YOUR_PATH/refiner_hwc_dynamic_fp16.engine
      ```

### 编译
```bash
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

使用Nvidia RTX4090D测试
### FoundationPose Register && Track 模式性能对比

下表详细对比了 **Register（首次注册）** 模式和 **Track（连续追踪）** 模式在各个处理阶段的耗时情况。，其中`Register`模式的耗时取其“稳态运行”时的典型值。

| 处理阶段 (Processing Stage) | 主要功能 (Main Function) | 耗时 (µs) - Register模式 | 耗时 (µs) - Track模式 |
| :--- | :--- | :---: | :---: |
| **数据上传** (UploadDataToDevice) | 将RGB/深度图数据从CPU拷贝到GPU | ~250 | ~220 |
| **Refiner预处理** (RefinePreProcess) | 生成/设置位姿假设，为Refiner渲染输入 | ~19,800 | ~1,010 |
| **Refiner模型推理** (refiner_core_->SyncInfer) | 执行Refiner网络，预测位"姿微调量 | ~33,000 | ~1,160 |
| **Scorer预处理** (ScorePreprocess) | *（Track模式跳过此阶段）* | ~18,500 | - |
| **Scorer模型推理** (scorer_core_->SyncInfer) | *（Track模式跳过此阶段）* | ~27,000 | - |
| **Scorer/Track后处理** (PostProcess) | 解码最终位姿 | ~60 | ~1 |
| **总计 (Total)** | **单次流程总耗时** | **~99,110 µs (~99.1 ms)** | **~2,391 µs (~2.4 ms)** |
| **估算帧率 (FPS)** | **Frames Per Second** | **~10 FPS** | **~418 FPS** |


---



## 参考

https://github.com/NVlabs/FoundationPose

https://github.com/NVlabs/nvdiffrast

https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_pose_estimation/tree/main/isaac_ros_foundationpose

https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/foundationpose
