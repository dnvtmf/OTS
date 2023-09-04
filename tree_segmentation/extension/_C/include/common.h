#pragma once
#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include <string>
#include <vector>

namespace py = pybind11;
using at::Tensor;
using std::tuple;
using std::vector;

#ifdef TORCH_CHECK
#define BCNN_ASSERT(cond, ...) TORCH_CHECK(cond, __VA_ARGS__)
#elif defined(AT_CHECK)
#define BCNN_ASSERT AT_CHECK
#elif AT_ASSERTM
#define BCNN_ASSERT AT_ASSERTM
#else
#define BCNN_ASSERT AT_ASSERT
#endif

// C++ interface
#define CHECK_CUDA(x) BCNN_ASSERT(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) BCNN_ASSERT(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

#define BIT_AT(val, idx) (((val) >> (idx)) & 1)

#define QLEN 32
#if QLEN == 32
#define QType uint32_t
#else
#define QType uint64_t
#endif

typedef void (*pybind11_init_fp)(py::module_ &m);

class PyTrochExtenstionRegistry {
 public:
  static void AddExtension(pybind11_init_fp f);
  static void InitAllExtension(py::module_ &m);

 private:
  static vector<pybind11_init_fp> &Registry();
};

class PyTorchExtensionRegistrer {
 public:
  PyTorchExtensionRegistrer(pybind11_init_fp f);
};

#define REGIST_PYTORCH_EXTENSION(name, ...) \
  PyTorchExtensionRegistrer pytorch_extenstion_##name([](py::module_ &m) { __VA_ARGS__; });
