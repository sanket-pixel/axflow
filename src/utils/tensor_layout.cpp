#include "axflow/utils/tensor_layout.h"

#include <stdexcept>
#include <string>

namespace axflow
{
    std::vector<int> shape_of(const axrTensorInfo& info)
    {
        return std::vector<int>(info.dims, info.dims + info.ndims);
    }

    std::vector<std::array<std::size_t, 2>> padding_of(const axrTensorInfo& info)
    {
        std::vector<std::array<std::size_t, 2>> padding(info.ndims);
        for (int i = 0; i < info.ndims; ++i)
        {
            padding[i] = {info.padding[i][0], info.padding[i][1]};
        }
        return padding;
    }

    std::vector<int> unpadded_shape(const axrTensorInfo& info)
    {
        std::vector<int> shape(info.ndims);
        for (int i = 0; i < info.ndims; ++i)
        {
            const int dim = static_cast<int>(info.dims[i] - info.padding[i][0] - info.padding[i][1]);
            if (dim <= 0)
            {
                throw std::runtime_error(
                    "axflow::unpadded_shape: invalid unpadded dim " + std::to_string(i)
                    + " = " + std::to_string(dim));
            }
            shape[i] = dim;
        }
        return shape;
    }
} // namespace axflow
