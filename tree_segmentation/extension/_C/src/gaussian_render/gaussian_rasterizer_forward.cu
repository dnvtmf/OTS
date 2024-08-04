/*
paper: 3D Gaussian Splatting for Real-Time Radiance Field Rendering, SIGGRAPH 2023
code:  https://github.com/graphdeco-inria/diff-gaussian-rasterization
*/
/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>

#include "gaussian_render.h"
#include "util.cuh"

namespace cg = cooperative_groups;

namespace GaussianRasterizer {

// Helper function to find the next-highest bit of the MSB on the CPU.
uint32_t getHigherMsb(uint32_t n) {
  uint32_t msb  = sizeof(n) * 4;
  uint32_t step = msb;
  while (step > 1) {
    step /= 2;
    if (n >> msb)
      msb += step;
    else
      msb -= step;
  }
  if (n >> msb) msb++;
  return msb;
}

// Generates one key/value pair for all Gaussian / tile overlaps.
// Run once per Gaussian (1:N mapping).
__global__ void duplicateWithKeys(int P, const float2* points_xy, const float* depths, const uint32_t* offsets,
    uint64_t* gaussian_keys_unsorted, uint32_t* gaussian_values_unsorted, int* radii, dim3 grid) {
  auto idx = cg::this_grid().thread_rank();
  if (idx >= P) return;

  // Generate no key/value pair for invisible Gaussians
  if (radii[idx] > 0) {
    // Find this Gaussian's offset in buffer for writing keys/values.
    uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
    uint2 rect_min, rect_max;

    getRect(points_xy[idx], radii[idx], rect_min, rect_max, grid);

    // For each tile that the bounding rect overlaps, emit a
    // key/value pair. The key is |  tile ID  |      depth      |,
    // and the value is the ID of the Gaussian. Sorting the values
    // with this key yields Gaussian IDs in a list, such that they
    // are first sorted by tile and then by depth.
    for (int y = rect_min.y; y < rect_max.y; y++) {
      for (int x = rect_min.x; x < rect_max.x; x++) {
        uint64_t key = y * grid.x + x;
        key <<= 32;
        key |= *((uint32_t*) &depths[idx]);
        gaussian_keys_unsorted[off]   = key;
        gaussian_values_unsorted[off] = idx;
        off++;
      }
    }
  }
}

// Check keys to see if it is at the start/end of one tile's range in
// the full sorted list. If yes, write start/end of this tile.
// Run once per instanced (duplicated) Gaussian ID.
__global__ void identifyTileRanges(int L, uint64_t* point_list_keys, uint2* ranges) {
  auto idx = cg::this_grid().thread_rank();
  if (idx >= L) return;

  // Read tile ID from key. Update start/end of tile range if at limit.
  uint64_t key      = point_list_keys[idx];
  uint32_t currtile = key >> 32;
  if (idx == 0)
    ranges[currtile].x = 0;
  else {
    uint32_t prevtile = point_list_keys[idx - 1] >> 32;
    if (currtile != prevtile) {
      ranges[prevtile].y = idx;
      ranges[currtile].x = idx;
    }
  }
  if (idx == L - 1) ranges[currtile].y = L;
}

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(
    int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped) {
  // The implementation is loosely based on code for
  // "Differentiable Point-Based Radiance Fields for
  // Efficient View Synthesis" by Zhang et al. (2022)
  glm::vec3 pos = means[idx];
  glm::vec3 dir = pos - campos;
  dir           = dir / glm::length(dir);

  glm::vec3* sh    = ((glm::vec3*) shs) + idx * max_coeffs;
  glm::vec3 result = SH_C0 * sh[0];

  if (deg > 0) {
    float x = dir.x;
    float y = dir.y;
    float z = dir.z;
    result  = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

    if (deg > 1) {
      float xx = x * x, yy = y * y, zz = z * z;
      float xy = x * y, yz = y * z, xz = x * z;
      result = result + SH_C2[0] * xy * sh[4] + SH_C2[1] * yz * sh[5] + SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
               SH_C2[3] * xz * sh[7] + SH_C2[4] * (xx - yy) * sh[8];

      if (deg > 2) {
        result = result + SH_C3[0] * y * (3.0f * xx - yy) * sh[9] + SH_C3[1] * xy * z * sh[10] +
                 SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
                 SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
                 SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] + SH_C3[5] * z * (xx - yy) * sh[14] +
                 SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
      }
    }
  }
  result += 0.5f;

  // RGB colors are clamped to positive values. If values are
  // clamped, we need to keep track of this for the backward pass.
  clamped[3 * idx + 0] = (result.x < 0);
  clamped[3 * idx + 1] = (result.y < 0);
  clamped[3 * idx + 2] = (result.z < 0);
  return glm::max(result, 0.0f);
}

// Forward version of 2D covariance matrix computation
__device__ float3 computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy,
    const float* cov3D, const float* viewmatrix) {
  // The following models the steps outlined by equations 29
  // and 31 in "EWA Splatting" (Zwicker et al., 2002).
  // Additionally considers aspect / scaling of viewport.
  // Transposes used to account for row-/column-major conventions.
  float3 t = transformPoint4x3(mean, viewmatrix);

  const float limx = 1.3f * tan_fovx;
  const float limy = 1.3f * tan_fovy;
  const float txtz = t.x / t.z;
  const float tytz = t.y / t.z;
  t.x              = min(limx, max(-limx, txtz)) * t.z;
  t.y              = min(limy, max(-limy, tytz)) * t.z;

  glm::mat3 J = glm::mat3(focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z), 0.0f, focal_y / t.z,
      -(focal_y * t.y) / (t.z * t.z), 0, 0, 0);

  glm::mat3 W = glm::mat3(viewmatrix[0], viewmatrix[4], viewmatrix[8], viewmatrix[1], viewmatrix[5], viewmatrix[9],
      viewmatrix[2], viewmatrix[6], viewmatrix[10]);
  // glm::mat3 W = glm::mat3(viewmatrix[0], viewmatrix[1], viewmatrix[2], viewmatrix[4], viewmatrix[5], viewmatrix[6],
  //     viewmatrix[8], viewmatrix[9], viewmatrix[10]);

  glm::mat3 T = W * J;

  glm::mat3 Vrk = glm::mat3(cov3D[0], cov3D[1], cov3D[2], cov3D[1], cov3D[3], cov3D[4], cov3D[2], cov3D[4], cov3D[5]);

  glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

  // Apply low-pass filter: every Gaussian should be at least
  // one pixel wide/high. Discard 3rd row and column.
  cov[0][0] += 0.3f;
  cov[1][1] += 0.3f;
  return {float(cov[0][0]), float(cov[0][1]), float(cov[1][1])};
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D) {
  // Create scaling matrix
  glm::mat3 S = glm::mat3(1.0f);
  S[0][0]     = mod * scale.x;
  S[1][1]     = mod * scale.y;
  S[2][2]     = mod * scale.z;

  // Normalize quaternion to get valid rotation
  glm::vec4 q = rot;  // / glm::length(rot);
  float r     = q.x;
  float x     = q.y;
  float y     = q.z;
  float z     = q.w;

  // Compute rotation matrix from quaternion
  glm::mat3 R = glm::mat3(1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
      2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x), 2.f * (x * z - r * y),
      2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y));

  glm::mat3 M = S * R;

  // Compute 3D world covariance matrix Sigma
  glm::mat3 Sigma = glm::transpose(M) * M;

  // Covariance is symmetric, only store upper right
  cov3D[0] = Sigma[0][0];
  cov3D[1] = Sigma[0][1];
  cov3D[2] = Sigma[0][2];
  cov3D[3] = Sigma[1][1];
  cov3D[4] = Sigma[1][2];
  cov3D[5] = Sigma[2][2];
}

// Perform initial steps for each Gaussian prior to rasterization.
template <int C>
__global__ void preprocessCUDA(int P, int D, int M, const float* orig_points, const glm::vec3* scales,
    const float scale_modifier, const glm::vec4* rotations, const float* opacities, const float* shs, bool* clamped,
    const float* cov3D_precomp, const float* colors_precomp, const float* viewmatrix, const float* projmatrix,
    const glm::vec3* cam_pos, const int W, int H, const float tan_fovx, float tan_fovy, const float focal_x,
    float focal_y, int* radii, float2* points_xy_image, float* depths, float* cov3Ds, float* rgb, float4* conic_opacity,
    const dim3 grid, uint32_t* tiles_touched, bool prefiltered) {
  auto idx = cg::this_grid().thread_rank();
  if (idx >= P) return;

  // Initialize radius and touched tiles to 0. If this isn't changed, this Gaussian will not be processed further.
  radii[idx]         = 0;
  tiles_touched[idx] = 0;

  // Perform near culling, quit if outside.
  float3 p_view;
  if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view)) return;

  // Transform point by projecting
  float3 p_orig = {orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2]};
  float4 p_hom  = transformPoint4x4(p_orig, projmatrix);
  float p_w     = 1.0f / (p_hom.w + 0.0000001f);
  float3 p_proj = {p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w};

  // If 3D covariance matrix is precomputed, use it, otherwise compute from scaling and rotation parameters.
  const float* cov3D;
  if (cov3D_precomp != nullptr) {
    cov3D = cov3D_precomp + idx * 6;
  } else {
    computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
    cov3D = cov3Ds + idx * 6;
  }

  // Compute 2D screen-space covariance matrix
  float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

  // Invert covariance (EWA algorithm)
  float det = (cov.x * cov.z - cov.y * cov.y);
  if (det == 0.0f) return;
  float det_inv = 1.f / det;
  float3 conic  = {cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv};

  // Compute extent in screen space (by finding eigenvalues of 2D covariance matrix).
  // Use extent to compute a bounding rectangle of screen-space tiles that this Gaussian overlaps with.
  // Quit if rectangle covers 0 tiles.
  float mid          = 0.5f * (cov.x + cov.z);
  float lambda1      = mid + sqrt(max(0.1f, mid * mid - det));
  float lambda2      = mid - sqrt(max(0.1f, mid * mid - det));
  float my_radius    = ceil(3.f * sqrt(max(lambda1, lambda2)));
  float2 point_image = {ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H)};
  uint2 rect_min, rect_max;
  getRect(point_image, my_radius, rect_min, rect_max, grid);
  if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0) return;

  // If colors have been precomputed, use them, otherwise convert spherical harmonics coefficients to RGB color.
  if (colors_precomp == nullptr) {
    glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*) orig_points, *cam_pos, shs, clamped);
    rgb[idx * C + 0] = result.x;
    rgb[idx * C + 1] = result.y;
    rgb[idx * C + 2] = result.z;
  }

  // Store some useful helper data for the next steps.
  depths[idx]          = p_view.z;
  radii[idx]           = my_radius;
  points_xy_image[idx] = point_image;
  // Inverse 2D covariance and opacity neatly pack into one float4
  conic_opacity[idx] = {conic.x, conic.y, conic.z, opacities[idx]};
  tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}

void render_forward(const dim3 grid, dim3 block, const uint2* ranges, const uint32_t* point_list, int W, int H, int E,
    const float2* means2D, const float* colors, const float4* conic_opacity, const float* extra, uint32_t* n_contrib,
    /*const float* bg_color,*/ float* out_color, float* out_opacity, float* out_extra);

void preprocess_forward(int P, int D, int M, const float* means3D, const glm::vec3* scales, const float scale_modifier,
    const glm::vec4* rotations, const float* opacities, const float* shs, bool* clamped, const float* cov3D_precomp,
    const float* colors_precomp, const float* viewmatrix, const float* projmatrix, const glm::vec3* cam_pos,
    const int W, int H, const float focal_x, float focal_y, const float tan_fovx, float tan_fovy, int* radii,
    float2* means2D, float* depths, float* cov3Ds, float* rgb, float4* conic_opacity, const dim3 grid,
    uint32_t* tiles_touched, bool prefiltered) {
  preprocessCUDA<NUM_CHANNELS> KERNEL_ARG((P + 255) / 256, 256)(P, D, M, means3D, scales, scale_modifier, rotations,
      opacities, shs, clamped, cov3D_precomp, colors_precomp, viewmatrix, projmatrix, cam_pos, W, H, tan_fovx, tan_fovy,
      focal_x, focal_y, radii, means2D, depths, cov3Ds, rgb, conic_opacity, grid, tiles_touched, prefiltered);
}

// Forward rendering procedure for differentiable rasterization of Gaussians.
int Rasterizer::forward(std::function<char*(size_t)> geometryBuffer, std::function<char*(size_t)> binningBuffer,
    std::function<char*(size_t)> imageBuffer, const int P, int D, int M, int E, const int width, int height,
    const float* means3D, const float* shs, const float* colors_precomp, const float* opacities, const float* scales,
    const float scale_modifier, const float* rotations, const float* cov3D_precomp, const float* viewmatrix,
    const float* projmatrix, const float* cam_pos, const float tan_fovx, float tan_fovy, const bool prefiltered,
    float* out_color, float* out_opacity, const float* extra, float* out_extra, int* radii, bool debug) {
  const float focal_y = height / (2.0f * tan_fovy);
  const float focal_x = width / (2.0f * tan_fovx);

  size_t chunk_size       = required<GeometryState>(P);
  char* chunkptr          = geometryBuffer(chunk_size);
  GeometryState geomState = GeometryState::fromChunk(chunkptr, P);

  if (radii == nullptr) {
    radii = geomState.internal_radii;
  }

  dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
  dim3 block(BLOCK_X, BLOCK_Y, 1);

  // Dynamically resize image-based auxiliary buffers during training
  size_t img_chunk_size = required<ImageState>(width * height);
  char* img_chunkptr    = imageBuffer(img_chunk_size);
  ImageState imgState   = ImageState::fromChunk(img_chunkptr, width * height);

  if (NUM_CHANNELS != 3 && colors_precomp == nullptr) {
    throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
  }

  // Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs to RGB)

  preprocess_forward(P, D, M, means3D, (glm::vec3*) scales, scale_modifier, (glm::vec4*) rotations, opacities, shs,
      geomState.clamped, cov3D_precomp, colors_precomp, viewmatrix, projmatrix, (glm::vec3*) cam_pos, width, height,
      focal_x, focal_y, tan_fovx, tan_fovy, radii, geomState.means2D, geomState.depths, geomState.cov3D, geomState.rgb,
      geomState.conic_opacity, tile_grid, geomState.tiles_touched, prefiltered);
  CHECK_CUDA_ERROR("preprocess");

  // Compute prefix sum over full list of touched tile counts by Gaussians
  // E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
  cub::DeviceScan::InclusiveSum(
      geomState.scanning_space, geomState.scan_size, geomState.tiles_touched, geomState.point_offsets, P);
  CHECK_CUDA_ERROR("InclusiveSum");
  // Retrieve total number of Gaussian instances to launch and resize aux buffers
  int num_rendered;
  cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost);
  CHECK_CUDA_ERROR("cudaMemcpy");

  size_t binning_chunk_size = required<BinningState>(num_rendered);
  char* binning_chunkptr    = binningBuffer(binning_chunk_size);
  BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

  // For each instance to be rendered, produce adequate [ tile | depth ] key
  // and corresponding dublicated Gaussian indices to be sorted
  duplicateWithKeys KERNEL_ARG((P + 255) / 256, 256)(P, geomState.means2D, geomState.depths, geomState.point_offsets,
      binningState.point_list_keys_unsorted, binningState.point_list_unsorted, radii, tile_grid);
  CHECK_CUDA_ERROR("duplicateWithKeys");

  int bit = getHigherMsb(tile_grid.x * tile_grid.y);

  // Sort complete list of (duplicated) Gaussian indices by keys
  cub::DeviceRadixSort::SortPairs(binningState.list_sorting_space, binningState.sorting_size,
      binningState.point_list_keys_unsorted, binningState.point_list_keys, binningState.point_list_unsorted,
      binningState.point_list, num_rendered, 0, 32 + bit);
  CHECK_CUDA_ERROR("SortPairs");
  cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2));
  CHECK_CUDA_ERROR("cudaMemset");
  // Identify start and end of per-tile workloads in sorted list
  if (num_rendered > 0) {
    identifyTileRanges KERNEL_ARG((num_rendered + 255) / 256, 256)(
        num_rendered, binningState.point_list_keys, imgState.ranges);
    CHECK_CUDA_ERROR("identifyTileRanges");
  }

  // Let each tile blend its range of Gaussians independently in parallel
  const float* feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;

  render_forward(tile_grid, block, imgState.ranges, binningState.point_list, width, height, E, geomState.means2D,
      feature_ptr, geomState.conic_opacity, extra, imgState.n_contrib, out_color, out_opacity, out_extra);

  return num_rendered;
}

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
  auto lambda = [&t](size_t N) {
    t.resize_({(long long) N});
    return reinterpret_cast<char*>(t.contiguous().data_ptr());
  };
  return lambda;
}

std::tuple<int, torch::Tensor, Tensor, Tensor, torch::Tensor, torch::Tensor, torch::Tensor, Tensor>
    RasterizeGaussiansCUDA(
        // const params
        const int image_height, const int image_width, const float tan_fovx, const float tan_fovy, const int degree,
        const float scale_modifier, const bool prefiltered, const bool debug,
        // tenser params
        const torch::Tensor& viewmatrix, const torch::Tensor& projmatrix, const torch::Tensor& campos,
        // inputs
        const torch::Tensor& means3D, const torch::Tensor& opacity, const torch::Tensor& sh,
        const torch::Tensor& scales, const torch::Tensor& rotations, const at::optional<Tensor>& extras,
        const torch::Tensor& colors, const torch::Tensor& cov3D_precomp) {
  if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
    AT_ERROR("means3D must have dimensions (num_points, 3)");
  }

  const int P = means3D.size(0);
  const int H = image_height;
  const int W = image_width;
  const int E = extras.has_value() ? extras.value().size(-1) : 0;

  // auto int_opts   = means3D.options().dtype(torch::kInt32);
  auto float_opts = means3D.options().dtype(torch::kFloat32);

  torch::Tensor out_color   = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
  torch::Tensor out_opacity = torch::full({H, W}, 0.0, float_opts);
  torch::Tensor radii       = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));
  Tensor out_extras;
  if (extras.has_value()) out_extras = torch::zeros({E, H, W}, float_opts);

  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);
  torch::Tensor geomBuffer                 = torch::empty({0}, options.device(device));
  torch::Tensor binningBuffer              = torch::empty({0}, options.device(device));
  torch::Tensor imgBuffer                  = torch::empty({0}, options.device(device));
  std::function<char*(size_t)> geomFunc    = resizeFunctional(geomBuffer);
  std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char*(size_t)> imgFunc     = resizeFunctional(imgBuffer);

  int rendered = 0;
  if (P != 0) {
    int M = 0;
    if (sh.size(0) != 0) {
      M = sh.size(1);
    }

    rendered = Rasterizer::forward(geomFunc, binningFunc, imgFunc, P, degree, M, E, W, H,
        means3D.contiguous().data<float>(), sh.contiguous().data_ptr<float>(), colors.contiguous().data<float>(),
        opacity.contiguous().data<float>(), scales.contiguous().data_ptr<float>(), scale_modifier,
        rotations.contiguous().data_ptr<float>(), cov3D_precomp.contiguous().data<float>(),
        viewmatrix.contiguous().data<float>(), projmatrix.contiguous().data<float>(), campos.contiguous().data<float>(),
        tan_fovx, tan_fovy, prefiltered, out_color.contiguous().data<float>(), out_opacity.data<float>(),
        extras.has_value() ? extras.value().contiguous().data<float>() : nullptr,
        extras.has_value() ? out_extras.data<float>() : nullptr, radii.contiguous().data<int>(), debug);
  }
  return std::make_tuple(rendered, out_color, out_opacity, radii, geomBuffer, binningBuffer, imgBuffer, out_extras);
}

/*
std::tuple<int, torch::Tensor, Tensor, Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    gaussian_rasterize_perpare_forward(const torch::Tensor& means3D, const torch::Tensor& colors,
        const torch::Tensor& opacity, const torch::Tensor& scales, const torch::Tensor& rotations,
        const float scale_modifier, const torch::Tensor& cov3D_precomp, const torch::Tensor& viewmatrix,
        const torch::Tensor& projmatrix, const float tan_fovx, const float tan_fovy, const int image_height,
        const int image_width, const torch::Tensor& sh, const int degree, const torch::Tensor& campos,
        const bool prefiltered, const bool debug) {
  if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
    AT_ERROR("means3D must have dimensions (num_points, 3)");
  }

  const int P = means3D.size(0);
  const int H = image_height;
  const int W = image_width;
  const int E = extras.has_value() ? extras.value().size(-1) : 0;

  // auto int_opts   = means3D.options().dtype(torch::kInt32);
  auto float_opts = means3D.options().dtype(torch::kFloat32);

  torch::Tensor out_color   = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
  torch::Tensor out_opacity = torch::full({H, W}, 0.0, float_opts);
  torch::Tensor radii       = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));

  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);
  torch::Tensor geomBuffer                 = torch::empty({0}, options.device(device));
  torch::Tensor binningBuffer              = torch::empty({0}, options.device(device));
  torch::Tensor imgBuffer                  = torch::empty({0}, options.device(device));
  std::function<char*(size_t)> geomFunc    = resizeFunctional(geomBuffer);
  std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char*(size_t)> imgFunc     = resizeFunctional(imgBuffer);

  int rendered = 0;
  if (P != 0) {
    int M = 0;
    if (sh.size(0) != 0) {
      M = sh.size(1);
    }

    rendered = Rasterizer::forward(geomFunc, binningFunc, imgFunc, P, degree, M, E, W, H,
        means3D.contiguous().data<float>(), sh.contiguous().data_ptr<float>(), colors.contiguous().data<float>(),
        opacity.contiguous().data<float>(), scales.contiguous().data_ptr<float>(), scale_modifier,
        rotations.contiguous().data_ptr<float>(), cov3D_precomp.contiguous().data<float>(),
        viewmatrix.contiguous().data<float>(), projmatrix.contiguous().data<float>(), campos.contiguous().data<float>(),
        tan_fovx, tan_fovy, prefiltered, out_color.contiguous().data<float>(), out_opacity.data<float>(), nullptr,
        nullptr, radii.contiguous().data<int>(), debug);
  }
  return std::make_tuple(rendered, out_color, out_opacity, radii, geomBuffer, binningBuffer, imgBuffer);
}
*/

REGIST_PYTORCH_EXTENSION(nerf_gaussian_render_forward, { m.def("rasterize_gaussians", &RasterizeGaussiansCUDA); })
}  // namespace GaussianRasterizer