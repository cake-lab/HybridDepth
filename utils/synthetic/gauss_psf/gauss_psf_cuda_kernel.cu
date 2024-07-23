#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace {
template <typename scalar_t>
__device__ __forceinline__ scalar_t gauss_psf(scalar_t c, scalar_t x, scalar_t y) {
  return (1.0 / (c * c)) * exp(-2.0 * (x * x + y * y) / (c * c));
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t d_gauss_psf(scalar_t c, scalar_t x, scalar_t y) {
  return (2.0 / (c * c * c)) * exp(-2.0 * (x * x + y * y) / (c * c)) * (2 * (x * x + y * y) / (c * c) - 1.0);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t theta(scalar_t c, scalar_t x, scalar_t y) {
  return 2.0 * (x * x + y * y) / (c * c);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t gauss_psf_short(scalar_t c, scalar_t t) {
  return (1.0 / (c * c)) * exp(-t);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t d_gauss_psf_short(scalar_t c, scalar_t t, scalar_t g) {
  return (2.0 / c) * (t - 1.0) * g;
}

template <typename scalar_t>
__global__ void gauss_psf_cuda_forward_kernel(
    const scalar_t* __restrict__ vInput,
    const scalar_t* __restrict__ vWeights,
    const scalar_t* __restrict__ G_x,
    const scalar_t* __restrict__ G_y,
    scalar_t* __restrict__ output,
    scalar_t* __restrict__ wsum,
    size_t batch,
    size_t height,
    size_t width,
    size_t kernel_size) {

  const int column = blockIdx.x * blockDim.x + threadIdx.x;
  const int index = blockIdx.y * width * height + column;

  auto kernel_sum = 0.0;
  auto value = 0.0;
  const int mid = kernel_size / 2;
  const int n = index / height / width;
  const int h = (index / width) % height;
  const int w = index % width;

  if (n < batch && h < height && w < width) {
    if (vWeights[index] > 1) {
        for (int i=0; i<kernel_size; i++) {
            if ((h + i - mid) >= 0 && (h + i - mid) < height) {
                for (int j=0; j<kernel_size; j++) {
                    if ((w + j - mid) >= 0 && (w + j - mid) < width) {
                        if (vWeights[n*width*height + (h+ i - mid)*width + w + j - mid] > 1){
                            const auto g = gauss_psf(vWeights[n*width*height + (h+ i - mid)*width + w + j - mid],
                                                     G_x[i * kernel_size + j],
                                                     G_y[i * kernel_size + j]);
                            kernel_sum += g;
                            value += g * vInput[n*width*height + (h+ i - mid)*width + w + j - mid];
                        }
                    }
                }
            }
        }
        output[index] = value / kernel_sum;
        wsum[index] = kernel_sum;
    } else {
        output[index] = vInput[index];
    }
  }
}

template <typename scalar_t>
__global__ void gauss_psf_cuda_backward_kernel(
    const scalar_t* __restrict__ vGrad,
    const scalar_t* __restrict__ vInput,
    const scalar_t* __restrict__ vOutput,
    const scalar_t* __restrict__ vWeights,
    const scalar_t* __restrict__ wsum,
    const scalar_t* __restrict__ G_x,
    const scalar_t* __restrict__ G_y,
    scalar_t* __restrict__ output_i,
    scalar_t* __restrict__ output_w,
    size_t batch,
    size_t height,
    size_t width,
    size_t kernel_size) {

  const int column = blockIdx.x * blockDim.x + threadIdx.x;
  const int index = blockIdx.y * width * height + column;

  auto value_i = 0.0;
  auto value_w = 0.0;
  const int mid = kernel_size / 2;
  const int n = index / height / width;
  const int h = (index / width) % height;
  const int w = index % width;

  if (n < batch && h < height && w < width) {
    if (vWeights[index] > 1) {
        for (int i=0; i<kernel_size; i++) {
            if ((h + i - mid) >= 0 && (h + i - mid) < height) {
                for (int j=0; j<kernel_size; j++) {
                    if ((w + j - mid) >= 0 && (w + j - mid) < width) {
                        if (vWeights[n*width*height + (h+ i - mid)*width + w + j - mid] > 1){
                            const auto t = theta(vWeights[index],
                                                     G_x[i * kernel_size + j],
                                                     G_y[i * kernel_size + j]);
                            const auto g = gauss_psf_short(vWeights[index], t);
                            const auto dg = d_gauss_psf_short(vWeights[index], t, g);

                            value_i += vGrad[n*width*height + (h+ i - mid)*width + w + j - mid] *
                                       g / wsum[n*width*height + (h+ i - mid)*width + w + j - mid];

                            value_w += vGrad[n*width*height + (h+ i - mid)*width + w + j - mid] * (
                                       dg / wsum[n*width*height + (h+ i - mid)*width + w + j - mid] *
                                       (vInput[index] - vOutput[n*width*height + (h+ i - mid)*width + w + j - mid]));
                        }
                    }
                }
            }
        }
        output_i[index] = value_i;
        output_w[index] = value_w;
    }
    else {
        output_i[index] = vGrad[index];
        output_w[index] = 0;
    }
  }
}

} // namespace

std::vector<at::Tensor> gauss_psf_cuda_forward(
    at::Tensor input,
    at::Tensor weights,
    at::Tensor G_x,
    at::Tensor G_y) {

  const auto kernel_size = G_x.size(0);
  const auto batch = input.size(0);
  const auto channel = input.size(1);
  const auto height = input.size(2);
  const auto width = input.size(3);
  auto vInput = input.view({-1, height, width});
  auto vWeights = weights.view({-1, height, width});

  const auto batch_size = vInput.size(0);

  auto output = at::zeros_like(vInput);
  auto wsum = at::ones_like(vWeights);

  const int threads = 1024;
  const dim3 blocks((height * width + threads - 1) / threads, batch_size);

  AT_DISPATCH_FLOATING_TYPES(vInput.type(), "gauss_psf_forward_cuda", ([&] {
    gauss_psf_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
        vInput.data<scalar_t>(),
        vWeights.data<scalar_t>(),
        G_x.data<scalar_t>(),
        G_y.data<scalar_t>(),
        output.data<scalar_t>(),
        wsum.data<scalar_t>(),
        batch_size,
        height,
        width,
        kernel_size);
  }));

  output = output.view({batch, channel, height, width});
  wsum = wsum.view({batch, channel, height, width});

  return {output, wsum};
}

std::vector<at::Tensor> gauss_psf_cuda_backward(
    at::Tensor grad,
    at::Tensor input,
    at::Tensor fwd_output,
    at::Tensor weights,
    at::Tensor wsum,
    at::Tensor G_x,
    at::Tensor G_y) {
    const auto kernel_size = G_x.size(0);
    const auto batch = input.size(0);
    const auto channel = input.size(1);
    const auto height = input.size(2);
    const auto width = input.size(3);
    auto vGrad = grad.view({-1, height, width});
    auto vInput = input.view({-1, height, width});
    auto vOutput = fwd_output.view({-1, height, width});
    auto vWeights = weights.view({-1, height, width});

    const auto batch_size = vInput.size(0);

    auto output_i = at::zeros_like(vInput);
    auto output_w = at::zeros_like(vWeights);

    const int threads = 1024;
    const dim3 blocks((height * width + threads - 1) / threads, batch_size);

    AT_DISPATCH_FLOATING_TYPES(vInput.type(), "gauss_psf_backward_cuda", ([&] {
      gauss_psf_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
          vGrad.data<scalar_t>(),
          vInput.data<scalar_t>(),
          vOutput.data<scalar_t>(),
          vWeights.data<scalar_t>(),
          wsum.data<scalar_t>(),
          G_x.data<scalar_t>(),
          G_y.data<scalar_t>(),
          output_i.data<scalar_t>(),
          output_w.data<scalar_t>(),
          batch_size,
          height,
          width,
          kernel_size);
    }));

    output_i = output_i.view({batch, channel, height, width});
    output_w = output_w.view({batch, channel, height, width});

    return {output_i, output_w};
}