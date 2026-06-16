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

    // Add to existing viz.cpp
    static const std::vector<cv::Scalar> SEG_COLORS = {
        {255, 56, 56}, {255, 157, 151}, {255, 112, 31},
        {255, 178, 29}, {207, 210, 49}, {72, 249, 10},
        {146, 204, 23}, {61, 219, 134}, {26, 147, 52},
        {0, 212, 187}, {44, 153, 168}, {0, 194, 255},
        {52, 69, 147}, {100, 115, 255}, {0, 24, 236},
        {132, 56, 255}, {82, 0, 133}, {203, 56, 255},
        {255, 149, 200}, {255, 55, 199}
    };

    void draw_seg_detections(cv::Mat& image,
                             const std::vector<SegDetection>& dets,
                             float mask_alpha)
    {
        cv::Mat overlay = image.clone();

        for (std::size_t i = 0; i < dets.size(); ++i)
        {
            const auto& det = dets[i];
            const auto& color = SEG_COLORS[i % SEG_COLORS.size()];

            // draw filled mask on overlay
            if (!det.mask.empty())
            {
                overlay.setTo(color, det.mask);
            }

            // draw bounding box
            cv::Point p1(static_cast<int>(det.x1), static_cast<int>(det.y1));
            cv::Point p2(static_cast<int>(det.x2), static_cast<int>(det.y2));
            cv::rectangle(image, p1, p2, color, 2);

            // draw label
            std::string class_name = (det.class_id >= 0 &&
                                         det.class_id < (int)COCO_CLASSES.size())
                                         ? COCO_CLASSES[det.class_id]
                                         : "unknown";
            std::string label = class_name + ": " +
                std::to_string(static_cast<int>(det.score * 100)) + "%";

            int base_line = 0;
            cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX,
                                                  0.5, 1, &base_line);
            cv::rectangle(image, p1,
                          cv::Point(p1.x + label_size.width, p1.y - label_size.height - 5),
                          color, cv::FILLED);
            cv::putText(image, label, cv::Point(p1.x, p1.y - 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
        }

        // blend mask overlay with original
        cv::addWeighted(overlay, mask_alpha, image, 1.0f - mask_alpha, 0, image);
    }
}
