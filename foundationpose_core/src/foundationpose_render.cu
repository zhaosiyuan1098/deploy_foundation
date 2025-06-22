// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include "foundationpose_render.cu.hpp"


void RasterizeCudaFwdShaderKernel(const RasterizeCudaFwdShaderParams p);
void InterpolateFwdKernel(const InterpolateKernelParams p);
void TextureFwdKernelLinear1(const TextureKernelParams p);

__device__ float clamp_func(float f, float a, float b) {
  return fmaxf(a, fminf(f, b));
}

__global__ void clamp_kernel(float* input, float min_value, float max_value, int N) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  // Check the boundaries
  if (idx >= N) {
    return;
  }
  input[idx] = clamp_func(input[idx], min_value, max_value);
}

namespace foundationpose_render {  

/*
This kernel performs:
 1. thresholdingof the point cloud
 2. subtraction of the position of pose array from the pointcloud
 3. downscaling of the point cloud
 
 pose_array_input is of size N*16, where N is the number of poses. 16  = transformation_mat_size
 pointcloud_input is of size N*n_points*3, where N is the number of poses
    and n_points is the number of points in the point cloud.
 
 It subtracts the pose transformation from each point in the cloud,
 1. checks if the z-component of the point is below "min_depth" and sets it to zero if it is
 2. and applies a downscaling factor to reduce the number of points.
 3. Then it checks if the absolute value of any of the x, y, or z components of the point
    is above "max_depth" and sets it to zero if it is.

 The result is stored back in the input array.
*/
__global__ void threshold_and_downscale_pointcloud_kernel(
    float* input, float* pose_array_input, int N, int n_points, float downscale_factor,
    float min_depth, float max_depth) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= N * n_points) {
    return;  // Check the boundaries
  }

  int pose_idx = idx / n_points;

  // 16 is the size of pose transformation matrix
  float pose_x = pose_array_input[16 * pose_idx + 12];
  float pose_y = pose_array_input[16 * pose_idx + 13];
  float pose_z = pose_array_input[16 * pose_idx + 14];

  // Calculate the index for the x, y, and z components of the point
  int x_idx = idx * 3;
  int y_idx = x_idx + 1;
  int z_idx = x_idx + 2;

  bool invalid_flag = false;
  // Any points with z below min_depth is set to 0
  if (input[z_idx] < min_depth) {
    invalid_flag = true;
  }

  input[x_idx] -= pose_x;
  input[y_idx] -= pose_y;
  input[z_idx] -= pose_z;

  // Divide all values by downscale_factor
  input[x_idx] /= downscale_factor;
  input[y_idx] /= downscale_factor;
  input[z_idx] /= downscale_factor;

  // Any points with absolute value(x,y or z) above max_depth is set to 0
  if (fabs(input[x_idx]) > max_depth || invalid_flag) {
    input[x_idx] = 0.0f;
  }
  if (fabs(input[y_idx]) > max_depth || invalid_flag) {
    input[y_idx] = 0.0f;
  }

  if (fabs(input[z_idx]) > max_depth || invalid_flag) {
    input[z_idx] = 0.0f;
  }
  return;
}


// concat two NHWC array on the last dimension
__global__ void concat_kernel(
    float* input_a, float* input_b, float* output, int N, int H, int W, int C1, int C2) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  // Check the boundaries
  if (idx >= N * H * W) {
    return;
  }

  for (int i = 0; i < C1; i++) {
    output[idx * (C1 + C2) + i] = input_a[idx * C1 + i];
  }

  for (int i = 0; i < C2; i++) {
    output[idx * (C1 + C2) + C1 + i] = input_b[idx * C2 + i];
  }
}


void clamp(cudaStream_t stream, float* input, float min_value, float max_value, int N) {
  int block_size = 256;
  int grid_size = (N + block_size - 1) / block_size;

  clamp_kernel<<<grid_size, block_size, 0, stream>>>(input, min_value, max_value, N);
}

void threshold_and_downscale_pointcloud(
    cudaStream_t stream, float* pointcloud_input, float* pose_array_input, int N, int n_points, float downscale_factor,
    float min_depth, float max_depth) {
  // Launch n_points threads
  int block_size = 256;
  int grid_size = ((N * n_points) + block_size - 1) / block_size;

  threshold_and_downscale_pointcloud_kernel<<<grid_size, block_size, 0, stream>>>(
      pointcloud_input, pose_array_input, N, n_points, downscale_factor, min_depth, max_depth);
}

void concat(cudaStream_t stream, float* input_a, float* input_b, float* output, int N, int H, int W, int C1, int C2) {
  // Launch N*H*W threads, each thread handle a vector of size C
  int block_size = 256;
  int grid_size = (N * H * W + block_size - 1) / block_size;

  concat_kernel<<<grid_size, block_size>>>(input_a, input_b, output, N, H, W, C1, C2);
}

void rasterize(
    cudaStream_t stream, CR::CudaRaster* cr, float* pos_ptr, int32_t* tri_ptr, float* out, int pos_count, int tri_count,
    int H, int W, int C) {
  const int32_t* range_ptr = 0;

  bool enablePeel = false;
  cr->setViewportSize(W, H, C);
  cr->setVertexBuffer((void*)pos_ptr, pos_count);
  cr->setIndexBuffer((void*)tri_ptr, tri_count);
  cr->setRenderModeFlags(0);

  cr->deferredClear(0u);
  bool success = cr->drawTriangles(range_ptr, enablePeel, stream);

  // Populate pixel shader kernel parameters.
  RasterizeCudaFwdShaderParams p;
  p.pos = pos_ptr;
  p.tri = tri_ptr;
  p.in_idx = (const int*)cr->getColorBuffer();
  p.out = out;
  p.numTriangles = tri_count;
  p.numVertices = pos_count;
  p.width = W;
  p.height = H;
  p.depth = C;

  p.instance_mode = 1;
  p.xs = 2.f / (float)p.width;
  p.xo = 1.f / (float)p.width - 1.f;
  p.ys = 2.f / (float)p.height;
  p.yo = 1.f / (float)p.height - 1.f;

  // Choose launch parameters.
  dim3 blockSize = getLaunchBlockSize(
      RAST_CUDA_FWD_SHADER_KERNEL_BLOCK_WIDTH, RAST_CUDA_FWD_SHADER_KERNEL_BLOCK_HEIGHT, p.width,
      p.height);
  dim3 gridSize = getLaunchGridSize(blockSize, p.width, p.height, p.depth);

  // Launch CUDA kernel.
  void* args[] = {&p};
  cudaLaunchKernel((void*)RasterizeCudaFwdShaderKernel, gridSize, blockSize, args, 0, stream);
}

void interpolate(
    cudaStream_t stream, float* attr_ptr, float* rast_ptr, int32_t* tri_ptr, float* out, int num_vertices,
    int num_triangles, int attr_dim, int H, int W, int C) {
  int instance_mode = attr_dim > 2 ? 1 : 0;

  InterpolateKernelParams p = {};  // Initialize all fields to zero.
  p.instance_mode = instance_mode;
  p.numVertices = num_vertices;
  p.numAttr = attr_dim;
  p.numTriangles = num_triangles;
  p.height = H;
  p.width = W;
  p.depth = C;

  // Get input pointers.
  p.attr = attr_ptr;
  p.rast = rast_ptr;
  p.tri = tri_ptr;
  p.attrBC = 0;
  p.out = out;

  // Choose launch parameters.
  dim3 blockSize = getLaunchBlockSize(
      IP_FWD_MAX_KERNEL_BLOCK_WIDTH, IP_FWD_MAX_KERNEL_BLOCK_HEIGHT, p.width, p.height);
  dim3 gridSize = getLaunchGridSize(blockSize, p.width, p.height, p.depth);

  // Launch CUDA kernel.
  void* args[] = {&p};
  void* func = (void*)InterpolateFwdKernel;
  cudaLaunchKernel(func, gridSize, blockSize, args, 0, stream);
}

void texture(
    cudaStream_t stream, float* tex_ptr, float* uv_ptr, float* out, int tex_height, int tex_width, int tex_channel,
    int tex_depth, int H, int W, int N) {
  TextureKernelParams p = {};  // Initialize all fields to zero.
  p.enableMip = false;
  p.filterMode = TEX_MODE_LINEAR;
  p.boundaryMode = TEX_BOUNDARY_MODE_WRAP;

  p.texDepth = tex_depth;
  p.texHeight = tex_height;
  p.texWidth = tex_width;
  p.channels = tex_channel;

  p.n = N;
  p.imgHeight = H;
  p.imgWidth = W;

  // Get input pointers.
  p.tex[0] = tex_ptr;
  p.uv = uv_ptr;
  p.mipLevelBias = NULL;

  p.out = out;

  // Choose kernel variants based on channel count.
  void* args[] = {&p};

  // Choose launch parameters for texture lookup kernel.
  dim3 blockSize = getLaunchBlockSize(
      TEX_FWD_MAX_KERNEL_BLOCK_WIDTH, TEX_FWD_MAX_KERNEL_BLOCK_HEIGHT, p.imgWidth, p.imgHeight);
  dim3 gridSize = getLaunchGridSize(blockSize, p.imgWidth, p.imgHeight, p.n);

  void* func = (void*)TextureFwdKernelLinear1;
  cudaLaunchKernel(func, gridSize, blockSize, args, 0, stream);
}

__global__ void transform_points_kernel(
    const float* transform_matrixs, int M, const float* points_vectors, 
    int N, float* transformed_points_vectors)
{
  int row_idx = threadIdx.y + blockIdx.y * blockDim.y;
  int col_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (row_idx >= M || col_idx >= N) return;

  const float* matrix = transform_matrixs + row_idx * 16;  // 指向当前 4x4 变换矩阵
  const float* point = points_vectors + col_idx * 3;       // 指向当前 3D 点
  float* transformed_point = transformed_points_vectors + (row_idx * N + col_idx) * 3;

  float x = point[0], y = point[1], z = point[2];
  // **Column-Major 访问方式**
  transformed_point[0] = matrix[0] * x + matrix[4] * y + matrix[8]  * z + matrix[12];
  transformed_point[1] = matrix[1] * x + matrix[5] * y + matrix[9]  * z + matrix[13];
  transformed_point[2] = matrix[2] * x + matrix[6] * y + matrix[10] * z + matrix[14];
}

static uint16_t ceil_div(uint16_t numerator, uint16_t denominator) {
  uint32_t accumulator = numerator + denominator - 1;
  return accumulator / denominator + 1;
}

void transform_points(cudaStream_t stream, const float* transform_matrixs, int M, const float* points_vectors, 
    int N, float* transformed_points_vectors)
{
  dim3 blockSize = {32, 32};
  dim3 gridSize = {ceil_div(N, 32), ceil_div(M, 32)};

  transform_points_kernel<<<gridSize, blockSize, 0, stream>>>(
      transform_matrixs, M, points_vectors, N, transformed_points_vectors);
}

__global__ void generate_pose_clip_kernel(
    const float* transform_matrixs, const float* bbox2d_matrixs, int M, const float* points_vectors, 
    int N, float* transformed_points_vectors, int rgb_H, int rgb_W)
{
  int row_idx = threadIdx.y + blockIdx.y * blockDim.y;
  int col_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (row_idx >= M || col_idx >= N) return;

  const float* matrix = transform_matrixs + row_idx * 16;  // 指向当前 4x4 变换矩阵
  const float* bbox2d = bbox2d_matrixs + row_idx * 4;      // 指向当前 4x1 bbox2d向量
  const float* point = points_vectors + col_idx * 3;       // 指向当前 3D 点
  float* transformed_point = transformed_points_vectors + (row_idx * N + col_idx) * 4;


  float l = bbox2d[0], t = rgb_H - bbox2d[1], r = bbox2d[2], b = rgb_H - bbox2d[3];
  float a00 = rgb_W / (r - l),           a11 = rgb_H / (t - b), 
        a30 = (rgb_W - r - l) / (r - l), a31 = (rgb_H - t - b) / (t - b);
  float x = point[0], y = point[1], z = point[2];

  // 1. 坐标变换
  float tx = matrix[0] * x + matrix[4] * y + matrix[8]  * z + matrix[12];
  float ty = matrix[1] * x + matrix[5] * y + matrix[9]  * z + matrix[13];
  float tz = matrix[2] * x + matrix[6] * y + matrix[10] * z + matrix[14];
  float tw = matrix[3] * x + matrix[7] * y + matrix[11] * z + matrix[15];

  // 2. 映射
  transformed_point[0] = tx * a00 + tw * a30;
  transformed_point[1] = ty * a11 + tw * a31;
  transformed_point[2] = tz;
  transformed_point[3] = tw;
}


void generate_pose_clip(cudaStream_t stream, const float* transform_matrixs, const float* bbox2d_matrix, 
        int M, const float* points_vectors, int N, float* transformed_points_vectors, int rgb_H, int rgb_W)
{
  dim3 blockSize = {32, 32};
  dim3 gridSize = {ceil_div(N, 32), ceil_div(M, 32)};

  generate_pose_clip_kernel<<<gridSize, blockSize, 0, stream>>>(
      transform_matrixs, bbox2d_matrix, M, points_vectors, N, transformed_points_vectors, rgb_H, rgb_W);
}

}   // namespace foundationpose_render