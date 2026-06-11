#include "viz.h"
#include <opencv2/imgproc.hpp>
#include <string>

namespace common
{
    void draw_detection(cv::Mat& image, const Detection& det)
    {
        if (det.score <= 0.0f) return;

        cv::Point p1(static_cast<int>(det.x1), static_cast<int>(det.y1));
        cv::Point p2(static_cast<int>(det.x2), static_cast<int>(det.y2));

        // draw box
        cv::rectangle(image, p1, p2, cv::Scalar(0, 255, 0), 2);

        // draw label background
        std::string label = "drone: " + std::to_string(static_cast<int>(det.score * 100)) + "%";
        int base_line = 0;
        cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &base_line);

        cv::Point label_bg_p2(p1.x + label_size.width, p1.y - label_size.height - 5);
        cv::rectangle(image, p1, label_bg_p2, cv::Scalar(0, 255, 0), cv::FILLED);

        // write text
        cv::putText(image, label, cv::Point(p1.x, p1.y - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }
}
