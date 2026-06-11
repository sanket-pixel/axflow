// BASE_FILE: examples/common/args.h
#pragma once
#include <string>
#include <optional>

namespace common
{
    struct DetectionArgs
    {
        std::string config_path;
        std::string image_path;
        std::string output_path;
    };

    std::optional<DetectionArgs> parse_detection_args(int argc, char** argv);
}
