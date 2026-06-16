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

        cv::rectangle(image, p1, p2, cv::Scalar(0, 255, 0), 2);

        // Use COCO class name if valid
        std::string class_name = (det.class_id >= 0 &&
                                     det.class_id < (int)COCO_CLASSES.size())
                                     ? COCO_CLASSES[det.class_id]
                                     : "unknown";
        std::string label = class_name + ": " + std::to_string(static_cast<int>(det.score * 100)) + "%";

        int base_line = 0;
        cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX,
                                              0.5, 1, &base_line);
        cv::rectangle(image, p1,
                      cv::Point(p1.x + label_size.width, p1.y - label_size.height - 5),
                      cv::Scalar(0, 255, 0), cv::FILLED);
        cv::putText(image, label, cv::Point(p1.x, p1.y - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }

    void draw_detections(cv::Mat& image, const std::vector<Detection>& dets)
    {
        for (const auto& d : dets)
            draw_detection(image, d);
    }
}
