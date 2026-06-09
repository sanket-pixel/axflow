#include "axflow/utils/dequantize.h"
#include "axflow/utils/tensor_layout.h"

#include <stdexcept>
#include <string>

namespace axflow
{
    Tensor dequantize_nhwc_to_nchw(const axrTensorInfo& info,
                                   const std::vector<int8_t>& int8_buffer,
                                   const std::string& name)
    {
        if (info.ndims != 4)
        {
            throw std::runtime_error(
                "axflow::dequantize_nhwc_to_nchw: only 4D tensors supported, got "
                + std::to_string(info.ndims) + " dims for '" + name + "'");
        }

        // unpadded dims
        const int N = static_cast<int>(info.dims[0] - info.padding[0][0] - info.padding[0][1]);
        const int H = static_cast<int>(info.dims[1] - info.padding[1][0] - info.padding[1][1]);
        const int W = static_cast<int>(info.dims[2] - info.padding[2][0] - info.padding[2][1]);
        const int C = static_cast<int>(info.dims[3] - info.padding[3][0] - info.padding[3][1]);

        if (N <= 0 || H <= 0 || W <= 0 || C <= 0)
        {
            throw std::runtime_error(
                "axflow::dequantize_nhwc_to_nchw: invalid unpadded shape for '" + name + "'");
        }

        // padded strides
        const int padded_H = static_cast<int>(info.dims[1]);
        const int padded_W = static_cast<int>(info.dims[2]);
        const int padded_C = static_cast<int>(info.dims[3]);
        const int pad_h = static_cast<int>(info.padding[1][0]);
        const int pad_w = static_cast<int>(info.padding[2][0]);
        const int pad_c = static_cast<int>(info.padding[3][0]);

        const float scale = info.scale;
        const int zp = info.zero_point;

        Tensor out;
        out.name = name;
        out.shape = {N, C, H, W};
        out.data.resize(static_cast<std::size_t>(N) * C * H * W);

        for (int n = 0; n < N; ++n)
        {
            for (int h = 0; h < H; ++h)
            {
                for (int w = 0; w < W; ++w)
                {
                    for (int c = 0; c < C; ++c)
                    {
                        const int in_idx = ((n * padded_H + (h + pad_h)) * padded_W
                                + (w + pad_w)) * padded_C
                            + (c + pad_c);
                        const int out_idx = ((n * C + c) * H + h) * W + w;
                        out.data[out_idx] = (static_cast<float>(int8_buffer[in_idx]) - zp) * scale;
                    }
                }
            }
        }
        return out;
    }
} // namespace axflow
