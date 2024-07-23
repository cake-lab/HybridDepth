#include <torch/torch.h>

#include <vector>

// CUDA forward declarations

std::vector<at::Tensor> gauss_psf_cuda_forward(
    at::Tensor input,
    at::Tensor weights,
    at::Tensor G_x,
    at::Tensor G_y);

std::vector<at::Tensor> gauss_psf_cuda_backward(
    at::Tensor grad,
    at::Tensor input,
    at::Tensor fwd_output,
    at::Tensor weights,
    at::Tensor wsum,
    at::Tensor G_x,
    at::Tensor G_y);

// C++ interface

#define CHECK_CUDA(x) AT_ASSERT(x.type().is_cuda())
#define CHECK_CONTIGUOUS(x) AT_ASSERT(x.is_contiguous())
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<at::Tensor> gauss_psf_forward(
    at::Tensor input,
    at::Tensor weights,
    at::Tensor G_x,
    at::Tensor G_y) {
  CHECK_INPUT(input);
  CHECK_INPUT(weights);
  CHECK_INPUT(G_x);
  CHECK_INPUT(G_y);

  return gauss_psf_cuda_forward(input, weights, G_x, G_y);
}

std::vector<at::Tensor> gauss_psf_backward(
    at::Tensor grad,
    at::Tensor input,
    at::Tensor fwd_output,
    at::Tensor weights,
    at::Tensor wsum,
    at::Tensor G_x,
    at::Tensor G_y) {
  CHECK_INPUT(grad);
  CHECK_INPUT(input);
  CHECK_INPUT(weights);
  CHECK_INPUT(wsum);
  CHECK_INPUT(G_x);
  CHECK_INPUT(G_y);

  return gauss_psf_cuda_backward(
      grad,
      input,
      fwd_output,
      weights,
      wsum,
      G_x,
      G_y);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &gauss_psf_forward, "gauss_psf forward (CUDA)");
  m.def("backward", &gauss_psf_backward, "gauss_psf backward (CUDA)");
}