#include "util.cuh"
namespace TreeSegment {

template <typename T>
__global__ void get_masks_and_weights_kernel(int K, int M, int V, int F, T eps, const T* __restrict__ P,
    const bool* __restrict__ masks_2d, const int32_t* __restrict__ indices, const bool* __restrict__ masks_view,
    T* __restrict__ weights, T* __restrict__ masks_3d) {
  int k = threadIdx.y + blockIdx.y * blockDim.y;
  int f = threadIdx.x + blockIdx.x * blockDim.x;
  if (k < K && f < F) {
    T weight_sum = 0;
    T mask3d_sum = 0;
    for (int m = 0; m < M; ++m) {
      T p   = P[k * M + m];
      int v = indices[m];
      weight_sum += p * masks_view[v * F + f];
      mask3d_sum += p * masks_2d[m * F + f];
    }
    weights[k * F + f]  = max(weight_sum, eps);
    masks_3d[k * F + f] = mask3d_sum;
  }
}

template <typename T>
__global__ void get_masks_and_weights_packed_kernel(int K, int M, int N, int V, int F, T eps, const T* __restrict__ P,
    const int32_t* __restrict__ masks_2d, const int32_t* __restrict__ indices, const bool* __restrict__ masks_view,
    T* __restrict__ weights, T* __restrict__ masks_3d) {
  int k = threadIdx.y + blockIdx.y * blockDim.y;
  int f = threadIdx.x + blockIdx.x * blockDim.x;
  if (k < K && f < F) {
    T weight_sum = 0;
    for (int m = 0; m < M; ++m) {
      weight_sum += P[k * M + m] * masks_view[indices[m] * F + f];
    }
    T mask3d_sum = 0;
    for (int n = 0; n < N; ++n) {
      int m = masks_2d[n * F + f];
      if (m > 0) {
        mask3d_sum += P[k * M + m - 1];
      }
    }
    weights[k * F + f]  = max(weight_sum, eps);
    masks_3d[k * F + f] = mask3d_sum;
  }
}

template <typename T>
__global__ void get_masks_and_weights_backward_kernel(int K, int M, int V, int F, T eps, const T* __restrict__ P,
    const bool* __restrict__ masks_2d, const int32_t* __restrict__ indices, const bool* __restrict__ masks_view,
    const T* __restrict__ weights, const T* __restrict__ masks_3d, const T* __restrict__ grad_masks_3d,
    T* __restrict__ grad_P) {
  int k       = blockIdx.x;
  int m       = blockIdx.y;
  const int v = indices[m];
  T gp        = 0;

  for (int f = threadIdx.x; f < F; f += blockDim.x) {
    T w = weights[k * F + f];
    if (w > eps) {
      w    = T(1) / w;
      T gm = grad_masks_3d[k * F + f] * w;
      T gw = -grad_masks_3d[k * F + f] * masks_3d[k * F + f] * w;
      gp += masks_view[v * F + f] * gw;
      gp += masks_2d[m * F + f] * gm;
    }
  }
  __syncthreads();
  reduce_sum_block<T, false>(gp);
  if (threadIdx.x == 0) grad_P[k * M + m] = gp;
}

template <typename T>
__global__ void get_masks_and_weights_packed_backward_kernel(int K, int M, int N, int V, int F, T eps,
    const T* __restrict__ P, const int32_t* __restrict__ masks_2d, const int32_t* __restrict__ indices,
    const bool* __restrict__ masks_view, const T* __restrict__ weights, const T* __restrict__ masks_3d,
    const T* __restrict__ grad_masks_3d, T* __restrict__ grad_P) {
  int k       = blockIdx.x;
  int m       = blockIdx.y;
  const int v = indices[m];
  T gp        = 0;

  for (int f = threadIdx.x; f < F; f += blockDim.x) {
    T w = weights[k * F + f];
    if (w > eps) {
      w    = T(1) / w;
      T gm = grad_masks_3d[k * F + f] * w;
      T gw = -grad_masks_3d[k * F + f] * masks_3d[k * F + f] * w;
      gp += masks_view[v * F + f] * gw;
      for (int n = 0; n < N; ++n)
        if (masks_2d[n * F + f] - 1 == m) gp += gm;
    }
  }
  __syncthreads();
  reduce_sum_block<T, false>(gp);
  if (threadIdx.x == 0) grad_P[k * M + m] = gp;
}

vector<Tensor> tree_seg_get_masks(
    Tensor P, Tensor masks_2d, Tensor indices_view, Tensor masks_view, bool packed, double eps) {
  CHECK_NDIM(P, 2);
  CHECK_NDIM(masks_2d, 2);
  CHECK_NDIM(indices_view, 1);
  CHECK_NDIM(masks_view, 2);
  CHECK_CUDA(P);
  CHECK_CUDA(masks_2d);
  CHECK_CUDA_AND_TYPE(indices_view, torch::kInt32);
  CHECK_CUDA_AND_TYPE(masks_view, torch::kBool);
  CHECK_CUDA_AND_TYPE(masks_2d, (packed ? torch::kInt32 : torch::kBool));
  int K = P.size(0);
  int M = P.size(1);
  int N = masks_2d.size(0);
  int F = masks_2d.size(1);
  int V = masks_view.size(0);
  if (!packed) BCNN_ASSERT(N == M, "Error shape for masks_2d");
  BCNN_ASSERT(indices_view.size(0) == M && masks_view.size(1) == F, "Error shape for indicies/masks_view");

  Tensor masks_3d = torch::zeros({K, F}, P.options());
  Tensor weights  = torch::zeros({K, F}, P.options());
  AT_DISPATCH_FLOATING_TYPES(P.scalar_type(), "get_masks_and_weights_kernel", [&] {
    if (packed) {
      get_masks_and_weights_packed_kernel<scalar_t> KERNEL_ARG(dim3((F + 15) / 16, (K + 15) / 16), dim3(16, 16))(K, M,
          N, V, F, scalar_t(eps), P.contiguous().data_ptr<scalar_t>(), masks_2d.contiguous().data_ptr<int32_t>(),
          indices_view.contiguous().data_ptr<int32_t>(), masks_view.contiguous().data_ptr<bool>(),
          weights.data_ptr<scalar_t>(), masks_3d.data_ptr<scalar_t>());
      CHECK_CUDA_ERROR("get_masks_and_weights_packed_kernel");
    } else {
      get_masks_and_weights_kernel<scalar_t> KERNEL_ARG(dim3((F + 15) / 16, (K + 15) / 16), dim3(16, 16))(K, M, V, F,
          scalar_t(eps), P.contiguous().data_ptr<scalar_t>(), masks_2d.contiguous().data_ptr<bool>(),
          indices_view.contiguous().data_ptr<int32_t>(), masks_view.contiguous().data_ptr<bool>(),
          weights.data_ptr<scalar_t>(), masks_3d.data_ptr<scalar_t>());
      CHECK_CUDA_ERROR("get_masks_and_weights_kernel");
    }
  });

  return {masks_3d / weights, weights};
}

Tensor tree_seg_get_masks_backward(Tensor P, Tensor masks_2d, Tensor indices_view, Tensor masks_view, Tensor masks_3d,
    Tensor weights, Tensor grad_masks_3d, bool packed, double eps) {
  int K = P.size(0);
  int M = P.size(1);
  int N = masks_2d.size(0);
  int F = masks_2d.size(1);
  int V = masks_view.size(0);

  Tensor grad_P = torch::zeros_like(P);
  AT_DISPATCH_FLOATING_TYPES(P.scalar_type(), "get_masks_and_weights_backward_kernel", [&] {
    if (packed) {
      get_masks_and_weights_packed_backward_kernel<scalar_t> KERNEL_ARG(dim3(K, M), get_cuda_threads(F))(K, M, N, V, F,
          scalar_t(eps), P.contiguous().data_ptr<scalar_t>(), masks_2d.contiguous().data_ptr<int32_t>(),
          indices_view.contiguous().data_ptr<int32_t>(), masks_view.contiguous().data_ptr<bool>(),
          weights.data_ptr<scalar_t>(), masks_3d.contiguous().data_ptr<scalar_t>(),
          grad_masks_3d.contiguous().data_ptr<scalar_t>(), grad_P.data_ptr<scalar_t>());
      CHECK_CUDA_ERROR("get_masks_and_weights_packed_backward_kernel");
    } else {
      get_masks_and_weights_backward_kernel<scalar_t> KERNEL_ARG(dim3(K, M), get_cuda_threads(F))(K, M, V, F,
          scalar_t(eps), P.contiguous().data_ptr<scalar_t>(), masks_2d.contiguous().data_ptr<bool>(),
          indices_view.contiguous().data_ptr<int32_t>(), masks_view.contiguous().data_ptr<bool>(),
          weights.data_ptr<scalar_t>(), masks_3d.contiguous().data_ptr<scalar_t>(),
          grad_masks_3d.contiguous().data_ptr<scalar_t>(), grad_P.data_ptr<scalar_t>());
      CHECK_CUDA_ERROR("get_masks_and_weights_backward_kernel");
    }
  });
  return grad_P;
}

REGIST_PYTORCH_EXTENSION(other_tree_seg_loss, {
  m.def("tree_seg_get_masks", &tree_seg_get_masks, "tree_seg_get_masks (CUDA)");
  m.def("tree_seg_get_masks_backward", &tree_seg_get_masks_backward, "tree_seg_get_masks_backward (CUDA)");
});
}  // namespace TreeSegment