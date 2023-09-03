#include "common.h"

namespace tree_seg {
using torch::Tensor;

template <typename Ta = int32_t, typename Tb = int32_t>
void intersect_cpu(int N, int M, int H, int W, const int32_t* a_start, const Ta* a_counts, const int32_t* b_start,
    const Tb* b_counts, float_t* out) {
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < M; ++j) {
      float_t sum = 0;
      for (int k = 0; k < H; ++k) {
        int sa = a_start[i * H + k], sb = b_start[j * H + k];
        auto *ca = a_counts + sa, *cb = b_counts + sb;
        int na = *ca, nb = *cb;
        bool a = 0, b = 0;
        while (na < W && nb < W) {
          if (na < nb) {
            a ^= 1;
            ca++;
            na += *ca;
          } else {
            b ^= 1;
            cb++;
            nb += *cb;
          }
          if (a && b) sum += min(na, nb) - max(na - *ca, nb - *cb);
        }
      }
    }
  }
}

void intersect(int W, Tensor& a_start, Tensor& a_counts, Tensor& b_start, Tensor& b_counts, Tensor& output) {
  BCNN_ASSERT(a_start.device() == b_start.device(), "a, b must be same device");
  BCNN_ASSERT(a_start.scalar_type() == at::kInt && b_start.scalar_type() == at::kInt, "dtype of a, b must be int32");
  BCNN_ASSERT(a_start.size(-1) == b_start.size(-1), "a, b must have same (H, W)");
  const int H = a_start.size(-1);
  const int N = a_start.numel() / H, M = b_start.numel() / H;
  BCNN_ASSERT(output.numel() == N * M && output.scalar_type() == at::kFloat, "Error output shape or dtype");
  if (a_start.is_cuda()) {
    BCNN_ASSERT(false, "Not Implemented");
  } else {
    intersect_cpu(N, M, H, W, a_start.data_ptr<int32_t>(), a_counts.to(at::kInt).data_ptr<int32_t>(),
        b_start.data_ptr<int32_t>(), b_counts.to(at::kInt).data_ptr<int32_t>(), output.data<float>());
  }
}

}  // namespace tree_seg
REGIST_PYTORCH_EXTENSION(tree_seg_mask, { m.def("intersect", &tree_seg::intersect, "intersect"); });