#pragma once

#include "axflow/config/preprocessor_config.h"
#include "axflow/data_types/input_buffer.h"

#include <opencv2/core.hpp>

namespace axflow
{
    // turns a cv::Mat into a chip-ready int8 NHWC padded buffer.
    //
    // assumes:
    //   - input cv::Mat is HWC BGR uint8, 3 channels
    //   - target InputBuffer is 4D NHWC with N=1 and C_unpadded == 3
    //   - resize mode is "stretch" (letterbox not yet implemented)
    //
    // throws on any violation of the above.
    //
    // usage:
    //   Preprocessor pp(cfg);
    //   auto in = model.get_input("image");
    //   pp.run(image, in);
    //
    class Preprocessor
    {
    public:
        explicit Preprocessor(const PreprocessingConfig& cfg);

        void run(const cv::Mat& image, InputBuffer& buffer) const;

    private:
        PreprocessingConfig cfg_;
    };
} // namespace axflow
