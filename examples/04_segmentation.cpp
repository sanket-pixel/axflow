#include "axflow/axflow.h"
#include "opencv2/opencv.hpp"
#include <iostream>

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << " <config.yaml> <image.jpg>\n";
        return 1;
    }

    cv::Mat image = cv::imread(argv[2]);
    if (image.empty())
    {
        std::cerr << "Failed to read image: " << argv[2] << "\n";
        return 1;
    }

    axflow::Device dev;
    axflow::AxFlow flow(dev, argv[1]);

    flow.preprocess(image);
    flow.inference();
    auto tensors = flow.postamble();

    std::cout << "Outputs: " << tensors.size() << "\n";
    for (std::size_t i = 0; i < tensors.size(); ++i)
    {
        std::cout << "  [" << i << "] " << tensors[i].name << " [";
        for (std::size_t d = 0; d < tensors[i].shape.size(); ++d)
        {
            std::cout << tensors[i].shape[d];
            if (d + 1 < tensors[i].shape.size()) std::cout << ",";
        }
        std::cout << "]\n";
    }

    return 0;
}
