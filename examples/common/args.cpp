// BASE_FILE: examples/common/args.cpp
#include "args.h"
#include <iostream>

namespace common
{
    std::optional<DetectionArgs> parse_detection_args(int argc, char** argv)
    {
        if (argc < 4)
        {
            std::cerr << "usage: " << argv[0] << " <config.yaml> <image.png> <output.png>\n";
            return std::nullopt;
        }

        DetectionArgs args;
        args.config_path = argv[1];
        args.image_path = argv[2];
        args.output_path = argv[3];
        return args;
    }
}
