#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

__forceinline__ __device__ float3 transformPoint4x3(const float3& p, const float* matrix) {
  float3 transformed = {
      matrix[0] * p.x + matrix[1] * p.y + matrix[2] * p.z + matrix[3],
      matrix[4] * p.x + matrix[5] * p.y + matrix[6] * p.z + matrix[7],
      matrix[8] * p.x + matrix[9] * p.y + matrix[10] * p.z + matrix[11],
  };
  return transformed;
}

__forceinline__ __device__ float4 transformPoint4x4(const float3& p, const float* matrix) {
  float4 transformed = {matrix[0] * p.x + matrix[1] * p.y + matrix[2] * p.z + matrix[3],
      matrix[4] * p.x + matrix[5] * p.y + matrix[6] * p.z + matrix[7],
      matrix[8] * p.x + matrix[9] * p.y + matrix[10] * p.z + matrix[11],
      matrix[12] * p.x + matrix[13] * p.y + matrix[14] * p.z + matrix[15]};
  return transformed;
}

__forceinline__ __device__ float3 transformVec4x3(const float3& p, const float* matrix) {
  float3 transformed = {
      matrix[0] * p.x + matrix[1] * p.y + matrix[2] * p.z,
      matrix[4] * p.x + matrix[5] * p.y + matrix[6] * p.z,
      matrix[8] * p.x + matrix[9] * p.y + matrix[10] * p.z,
  };
  return transformed;
}

__forceinline__ __device__ float3 transformVec4x3Transpose(const float3& p, const float* matrix) {
  float3 transformed = {
      matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z,
      matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z,
      matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z,
  };
  return transformed;
}
