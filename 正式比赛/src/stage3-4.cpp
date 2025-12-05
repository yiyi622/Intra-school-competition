#include <cv_bridge/cv_bridge.h>
#include <cmath>
#include <geometry_msgs/msg/point.hpp>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <referee_pkg/msg/multi_object.hpp>
#include <referee_pkg/msg/object.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <std_msgs/msg/header.hpp>
#include "sensor_msgs/msg/image.hpp"

using namespace std;
using namespace rclcpp;
using namespace cv;

class VisionNode : public rclcpp::Node
{
private:
    void camera_callback(sensor_msgs::msg::Image::SharedPtr msg);

    vector<referee_pkg::msg::Object> detect_targets(const cv::Mat &image);
    vector<referee_pkg::msg::Object> detect_armor(const cv::Mat &image, const cv::Mat &mask);

    vector<Point2f> get_light_bar_endpoints(const vector<Point> &contour);
    Point2f refine_endpoint(const vector<Point> &contour, const Point &extreme_point, bool is_top);

    // 颜色阈值结构体
    struct ColorThreshold
    {
        Scalar lower;
        Scalar upper;
        string color_name;
    };

    vector<ColorThreshold> color_thresholds_;

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr camera_subscriber;
    rclcpp::Publisher<referee_pkg::msg::MultiObject>::SharedPtr result_publisher;

    // 数字识别相关
    std::vector<cv::Mat> digit_templates;
    std::vector<int> template_labels;
    void load_digit_templates();
    cv::Mat extract_digit_roi(const cv::Mat &image, const std::vector<cv::Point2f> &armor_points);
    cv::Mat normalize_digit(const cv::Mat &digit, cv::Size size = cv::Size(45, 80)); // 修改为实际尺寸
    int match_digit(const cv::Mat &input_digit, double threshold = 0.6);

public:
    VisionNode(const string &name) : Node(name)
    {
        RCLCPP_INFO(this->get_logger(), "Initializing Vision_node");

        // 初始化颜色阈值（示例值，需要根据实际调整）
        color_thresholds_ = {
            {Scalar(80, 100, 100), Scalar(100, 255, 255), "cyan"},
            {Scalar(-50, 200, 60), Scalar(20, 255, 255), "red1"},
            {Scalar(10, 120, 70), Scalar(190, 255, 255), "red2"},
            {Scalar(40, 50, 50), Scalar(85, 230, 255), "green"}};

        camera_subscriber = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera/image_raw", 10,
            bind(&VisionNode::camera_callback, this, placeholders::_1));

        result_publisher = this->create_publisher<referee_pkg::msg::MultiObject>(
            "/vision/target", 10);

        namedWindow("Detected Result", WINDOW_AUTOSIZE);
        RCLCPP_INFO(this->get_logger(), "Vision_Node initialized successfully");

        // 初始化数字识别（使用真实模板）
        load_digit_templates();
        RCLCPP_INFO(this->get_logger(), "Digit recognition with real templates initialized");
    }

    ~VisionNode()
    {
        destroyWindow("Detected Result");
    }
};

// 改进的装甲板检测（基于比例关系，直接使用灯条端点）
vector<referee_pkg::msg::Object> VisionNode::detect_armor(const cv::Mat &image, const cv::Mat &mask)
{
    vector<referee_pkg::msg::Object> armors;
    vector<vector<Point>> contours;
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    vector<pair<Point2f, Point2f>> light_bars; // 存储每个灯条的(top, bottom)端点

    // 第一步：提取所有灯条的上下端点
    for (const auto &contour : contours)
    {
        double area = contourArea(contour);
        if (area < 5)
            continue;

        // 获取灯条的上下端点（基于轮廓的极值点）
        auto endpoints = get_light_bar_endpoints(contour);
        if (endpoints.size() < 2)
            continue;

        Point2f top_point = endpoints[0];
        Point2f bottom_point = endpoints[1];

        float light_length = norm(top_point - bottom_point);

        // 基本过滤：灯条长度应该合理
        if (light_length > 3 && light_length < 200)
        {
            light_bars.push_back({top_point, bottom_point});

            // 可视化灯条端点
            circle(image, top_point, 3, Scalar(0, 255, 0), 2);    // 绿色：上端点
            circle(image, bottom_point, 3, Scalar(255, 0, 0), 2); // 蓝色：下端点
        }
    }

    // 第二步：基于比例关系配对灯条（支持绕X轴旋转）
    for (size_t i = 0; i < light_bars.size(); i++)
    {
        for (size_t j = i + 1; j < light_bars.size(); j++)
        {
            auto &left_light = light_bars[i];
            auto &right_light = light_bars[j];

            Point2f left_top = left_light.first;
            Point2f left_bottom = left_light.second;
            Point2f right_top = right_light.first;
            Point2f right_bottom = right_light.second;

            // 确保左右顺序
            if (left_top.x > right_top.x)
            {
                swap(left_light, right_light);
                left_top = left_light.first;
                left_bottom = left_light.second;
                right_top = right_light.first;
                right_bottom = right_light.second;
            }

            // 计算几何特征
            float left_length = norm(left_top - left_bottom);
            float right_length = norm(right_top - right_bottom);
            float horizontal_distance = (right_top.x + right_bottom.x) / 2 - (left_top.x + left_bottom.x) / 2;

            // 基于比例关系：间距在2-3倍灯条长度之间
            float length_ratio = horizontal_distance / ((left_length + right_length) / 2);

            // 计算绕X轴旋转的角度（基于两个灯条的倾斜）
            float left_angle = atan2(left_bottom.x - left_top.x, left_bottom.y - left_top.y) * 180 / CV_PI;
            float right_angle = atan2(right_bottom.x - right_top.x, right_bottom.y - right_top.y) * 180 / CV_PI;
            float angle_diff = abs(left_angle - right_angle);

            // 放宽的配对条件：允许一定的角度差异（绕X轴旋转）
            if (length_ratio > 2.5 && length_ratio < 5 && angle_diff < 45.0)
            {
                // 计算梯形特征
                float top_distance = right_top.x - left_top.x;
                float bottom_distance = right_bottom.x - left_bottom.x;
                float trapezoid_ratio = top_distance / bottom_distance;

                // 使用灯条端点作为装甲板角点
                vector<Point2f> armor_points;
                armor_points.push_back(left_bottom);  // 左下 (1)
                armor_points.push_back(right_bottom); // 右下 (2)
                armor_points.push_back(right_top);    // 右上 (3)
                armor_points.push_back(left_top);     // 左上 (4)

                // === 数字识别开始 ===
                cv::Mat digit_roi = extract_digit_roi(image, armor_points);
                int digit = -1;

                if (!digit_roi.empty())
                {
                    // 尝试识别数字
                    digit = match_digit(digit_roi, 0.4); // 装甲板运动影响识别，使用稍低的阈值0.4

                    // 可视化数字区域（调试用）
                    // cv::rectangle(image, digit_roi, cv::Scalar(255, 255, 0), 2);

                    // 显示识别结果
                    std::string digit_text = "Digit: ";
                    if (digit != -1)
                    {
                        digit_text += std::to_string(digit);
                    }
                    else
                    {
                        digit_text += "Unknown";
                    }

                    cv::Point text_position(armor_points[0].x, armor_points[0].y - 15);
                    cv::putText(image, digit_text, text_position,
                                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0), 2);

                    RCLCPP_INFO(this->get_logger(), "Digit ROI extracted, recognition result: %s",
                                digit == -1 ? "Unknown" : std::to_string(digit).c_str());
                }
                else
                {
                    RCLCPP_DEBUG(this->get_logger(), "No digit ROI extracted from armor");
                }
                // === 数字识别结束 ===

                // 创建ROS消息
                referee_pkg::msg::Object obj;

                // 根据识别到的数字设置目标类型
                if (digit >= 1 && digit <= 5)
                {
                    obj.target_type = "armor_red_" + std::to_string(digit);
                    RCLCPP_INFO(this->get_logger(), "Detected armor_red_%d", digit);
                }
                else
                {
                    // 如果数字识别失败，使用默认类型
                    obj.target_type = "armor_red_1"; // 默认设为1号装甲板
                    RCLCPP_WARN(this->get_logger(), "Digit recognition failed, using default type armor_red_1");
                }

                // 填充角点信息
                for (const auto &point : armor_points)
                {
                    geometry_msgs::msg::Point corner;
                    corner.x = point.x;
                    corner.y = point.y;
                    corner.z = 0.0;
                    obj.corners.push_back(corner);
                }
                armors.push_back(obj);

                // 可视化装甲板边界
                for (int k = 0; k < 4; k++)
                {
                    line(image, armor_points[k], armor_points[(k + 1) % 4], Scalar(0, 0, 255), 3);
                }

                // 标注角点序号（左下→右下→右上→左上）
                vector<string> point_labels = {"1", "2", "3", "4"};
                vector<Scalar> colors = {
                    Scalar(0, 255, 255), // 黄色 - 1
                    Scalar(255, 255, 0), // 青色 - 2
                    Scalar(255, 0, 255), // 粉色 - 3
                    Scalar(0, 255, 0)    // 绿色 - 4
                };

                for (int k = 0; k < 4; k++)
                {
                    // 绘制角点
                    circle(image, armor_points[k], 8, colors[k], -1);

                    // 标注序号（带背景框确保可读性）
                    string label = point_labels[k];
                    Point text_position(armor_points[k].x + 12, armor_points[k].y - 12);

                    // 先绘制背景框
                    putText(image, label, text_position,
                            FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 0), 4);

                    // 再绘制前景文字
                    putText(image, label, text_position,
                            FONT_HERSHEY_SIMPLEX, 0.8, colors[k], 2);
                }

                // 调试信息
                RCLCPP_INFO(this->get_logger(),
                            "Armor matched: points labeled 1-4, trapezoid_ratio=%.2f, length_ratio=%.2f, angle_diff=%.1f",
                            trapezoid_ratio, length_ratio, angle_diff);
            }
        }
    }

    return armors;
}

// 获取灯条的上下端点（基于轮廓极值）
vector<Point2f> VisionNode::get_light_bar_endpoints(const vector<Point> &contour)
{
    vector<Point2f> endpoints;

    if (contour.empty())
        return endpoints;

    // 找到y最小和最大的点（上下端点）
    Point top_point = contour[0];
    Point bottom_point = contour[0];

    for (const auto &point : contour)
    {
        if (point.y < top_point.y)
            top_point = point;
        if (point.y > bottom_point.y)
            bottom_point = point;
    }

    // 对上下端点进行微调，找到x坐标更接近中心的位置
    Point2f refined_top = refine_endpoint(contour, top_point, true);
    Point2f refined_bottom = refine_endpoint(contour, bottom_point, false);

    endpoints.push_back(refined_top);
    endpoints.push_back(refined_bottom);

    return endpoints;
}

// 精炼端点位置（避免使用轮廓的极端尖角）
Point2f VisionNode::refine_endpoint(const vector<Point> &contour, const Point &extreme_point, bool is_top)
{
    // 简单实现：在极值点附近找几个点取平均
    vector<Point> candidates;
    int search_range = 3;

    for (const auto &point : contour)
    {
        if (is_top)
        {
            if (abs(point.y - extreme_point.y) <= search_range)
            {
                candidates.push_back(point);
            }
        }
        else
        {
            if (abs(point.y - extreme_point.y) <= search_range)
            {
                candidates.push_back(point);
            }
        }
    }

    if (!candidates.empty())
    {
        // 取x坐标的中值，避免异常点影响
        sort(candidates.begin(), candidates.end(),
             [](const Point &a, const Point &b)
             { return a.x < b.x; });
        return Point2f(candidates[candidates.size() / 2].x, extreme_point.y);
    }

    return Point2f(extreme_point.x, extreme_point.y);
}

// 提取数字ROI
cv::Mat VisionNode::extract_digit_roi(const cv::Mat &image, const std::vector<cv::Point2f> &armor_points)
{
    if (armor_points.size() != 4)
    {
        return cv::Mat();
    }

    try
    {
        // 计算装甲板边界框
        float min_x = std::min({armor_points[0].x, armor_points[1].x, armor_points[2].x, armor_points[3].x});
        float max_x = std::max({armor_points[0].x, armor_points[1].x, armor_points[2].x, armor_points[3].x});
        float min_y = std::min({armor_points[0].y, armor_points[1].y, armor_points[2].y, armor_points[3].y});
        float max_y = std::max({armor_points[0].y, armor_points[1].y, armor_points[2].y, armor_points[3].y});

        float width = max_x - min_x;
        float height = max_y - min_y;

        // 扩展区域用于数字检测
        float expand_height = height * 0.6f; // 扩展比例

        int roi_x = std::max(0, static_cast<int>(min_x));
        int roi_y = std::max(0, static_cast<int>(min_y - expand_height));
        int roi_width = static_cast<int>(width);
        int roi_height = static_cast<int>(height + 2 * expand_height);

        // 确保不超出图像边界
        roi_width = std::min(roi_width, image.cols - roi_x);
        roi_height = std::min(roi_height, image.rows - roi_y);

        if (roi_width <= 10 || roi_height <= 10)
        {
            return cv::Mat();
        }

        cv::Rect roi_rect(roi_x, roi_y, roi_width, roi_height);

        // 提取ROI并进行预处理
        cv::Mat roi = image(roi_rect).clone();
        cv::Mat gray, binary;
        cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);

        // 尝试不同的二值化阈值
        cv::threshold(gray, binary, 180, 255, cv::THRESH_BINARY); // 降低阈值

        // 形态学操作去除噪声
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
        cv::morphologyEx(binary, binary, cv::MORPH_OPEN, kernel);

        // 查找数字轮廓
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        if (contours.empty())
        {
            return cv::Mat();
        }

        // 找到最大的轮廓（应该是数字）
        auto max_contour = std::max_element(contours.begin(), contours.end(),
                                            [](const std::vector<cv::Point> &a, const std::vector<cv::Point> &b)
                                            {
                                                return cv::contourArea(a) < cv::contourArea(b);
                                            });

        cv::Rect digit_rect = cv::boundingRect(*max_contour);

        if (digit_rect.area() < 25) // 降低最小面积要求
        {
            return cv::Mat();
        }

        // 扩展数字区域
        int margin = 2;
        int digit_x = std::max(0, digit_rect.x - margin);
        int digit_y = std::max(0, digit_rect.y - margin);
        int digit_width = std::min(binary.cols - digit_x, digit_rect.width + 2 * margin);
        int digit_height = std::min(binary.rows - digit_y, digit_rect.height + 2 * margin);

        if (digit_width <= 5 || digit_height <= 5)
        {
            return cv::Mat();
        }

        cv::Rect final_digit_rect(digit_x, digit_y, digit_width, digit_height);
        cv::Mat digit_roi = binary(final_digit_rect).clone();

        RCLCPP_DEBUG(this->get_logger(), "Extracted digit ROI: %dx%d", digit_roi.cols, digit_roi.rows);
        return digit_roi;
    }
    catch (const std::exception &e)
    {
        RCLCPP_ERROR(this->get_logger(), "Error in extract_digit_roi: %s", e.what());
        return cv::Mat();
    }
}

// 加载真实数字模板 - 改进版本
void VisionNode::load_digit_templates()
{
    digit_templates.clear();
    template_labels.clear();

    std::vector<std::string> digit_files = {
        "digit_1.png", "digit_2.png", "digit_3.png", "digit_4.png", "digit_5.png"};

    // 首先确定所有模板的平均尺寸
    cv::Size avg_size(0, 0);
    int valid_templates = 0;

    for (const auto &file_name : digit_files)
    {
        cv::Mat template_img = cv::imread(file_name, cv::IMREAD_GRAYSCALE);
        if (!template_img.empty())
        {
            avg_size.width += template_img.cols;
            avg_size.height += template_img.rows;
            valid_templates++;
        }
    }

    if (valid_templates > 0)
    {
        avg_size.width /= valid_templates;
        avg_size.height /= valid_templates;
        RCLCPP_INFO(this->get_logger(), "Average template size: %dx%d", avg_size.width, avg_size.height);
    }
    else
    {
        avg_size = cv::Size(50, 80); // 默认尺寸
    }

    for (size_t i = 0; i < digit_files.size(); i++)
    {
        cv::Mat template_img = cv::imread(digit_files[i], cv::IMREAD_GRAYSCALE);
        if (!template_img.empty())
        {
            // 使用平均尺寸进行归一化
            cv::Mat normalized = normalize_digit(template_img, avg_size);
            digit_templates.push_back(normalized);
            template_labels.push_back(i + 1);
            RCLCPP_INFO(this->get_logger(), "Loaded template: %s (%dx%d -> %dx%d)",
                        digit_files[i].c_str(),
                        template_img.cols, template_img.rows,
                        normalized.cols, normalized.rows);
        }
        else
        {
            RCLCPP_ERROR(this->get_logger(), "Failed to load template: %s", digit_files[i].c_str());
        }
    }

    if (!digit_templates.empty())
    {
        RCLCPP_INFO(this->get_logger(), "Successfully loaded %zu templates, size: %dx%d",
                    digit_templates.size(),
                    digit_templates[0].cols, digit_templates[0].rows);
    }
}

// 归一化数字图像 ，避免强制缩放
cv::Mat VisionNode::normalize_digit(const cv::Mat &digit, cv::Size size)
{
    cv::Mat resized;

    // 计算缩放比例，保持宽高比
    double scale_x = (double)size.width / digit.cols;
    double scale_y = (double)size.height / digit.rows;
    double scale = std::min(scale_x, scale_y); // 使用较小的比例，确保完全包含

    cv::Size new_size(digit.cols * scale, digit.rows * scale);
    cv::resize(digit, resized, new_size);

    // 创建目标图像并居中放置
    cv::Mat result = cv::Mat::zeros(size, digit.type());
    int x = (size.width - resized.cols) / 2;
    int y = (size.height - resized.rows) / 2;
    resized.copyTo(result(cv::Rect(x, y, resized.cols, resized.rows)));

    return result;
}

// 模板匹配识别数字
int VisionNode::match_digit(const cv::Mat &input_digit, double threshold)
{
    if (digit_templates.empty())
    {
        RCLCPP_WARN(this->get_logger(), "No digit templates loaded");
        return -1;
    }

    if (input_digit.empty())
    {
        RCLCPP_DEBUG(this->get_logger(), "Input digit is empty");
        return -1;
    }

    RCLCPP_INFO(this->get_logger(), "Input digit: %dx%d, Template: %dx%d",
                input_digit.cols, input_digit.rows,
                digit_templates[0].cols, digit_templates[0].rows);

    // 预处理输入图像
    cv::Mat processed_input = input_digit.clone();

    // 归一化到模板大小
    cv::Mat resized_input = normalize_digit(processed_input, digit_templates[0].size());

    double best_score = 0.0;
    int best_label = -1;

    for (size_t i = 0; i < digit_templates.size(); i++)
    {
        // 尝试两种匹配方法
        cv::Mat result1, result2;
        double score1, score2;

        // 标准匹配
        cv::matchTemplate(resized_input, digit_templates[i], result1, cv::TM_CCOEFF_NORMED);
        cv::minMaxLoc(result1, nullptr, &score1);

        // 取较高分数
        double max_score = std::max(score1, score2);

        RCLCPP_DEBUG(this->get_logger(), "Digit %d scores: normal=%.3f, inverted=%.3f",
                     template_labels[i], score1, score2);

        if (max_score > best_score)
        {
            best_score = max_score;
            best_label = template_labels[i];
        }
        /*
                // 在 match_digit 函数中添加调试保存
                if (best_score < 0.3)
                { // 如果匹配很差，保存调试信息
                    static int debug_count = 0;
                    std::string debug_name = "poor_match_" + std::to_string(debug_count) + ".png";
                    cv::imwrite(debug_name, resized_input);
                    RCLCPP_WARN(this->get_logger(), "Saved poor match case: %s", debug_name.c_str());
                    debug_count++;
                }*/
    }

    // 根据实际匹配情况调整阈值
    double adjusted_threshold = 0.25;

    RCLCPP_INFO(this->get_logger(), "Best match: digit %d, score: %.3f (threshold: %.3f)",
                best_label, best_score, adjusted_threshold);

    if (best_score > adjusted_threshold)
    {
        RCLCPP_INFO(this->get_logger(), "Digit matched: %d (score: %.3f)", best_label, best_score);
        return best_label;
    }
    else
    {
        RCLCPP_INFO(this->get_logger(), "No digit matched (best score: %.3f)", best_score);
        return -1;
    }
}

// 主检测函数
vector<referee_pkg::msg::Object> VisionNode::detect_targets(const cv::Mat &image)
{
    vector<referee_pkg::msg::Object> all_targets;

    Mat hsv;
    cvtColor(image, hsv, COLOR_BGR2HSV);

    // 调试：显示HSV图像和颜色分布
    imshow("HSV Image", hsv);

    // 创建所有颜色的合并掩码用于调试
    Mat debug_mask = Mat::zeros(image.size(), CV_8UC1);

    // 多颜色检测
    for (const auto &color : color_thresholds_)
    {
        Mat mask;
        inRange(hsv, color.lower, color.upper, mask);

        // 添加到调试掩码
        debug_mask = debug_mask | mask;

        // 形态学操作（傻逼形态学操作，坑了我俩小时）
        // Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
        // morphologyEx(mask, mask, MORPH_CLOSE, kernel);
        // morphologyEx(mask, mask, MORPH_OPEN, kernel);

        // 显示每个颜色的掩码
        imshow("Mask " + color.color_name, mask);

        // 统计掩码中的像素数量
        int pixel_count = countNonZero(mask);
        RCLCPP_INFO(this->get_logger(), "Color %s: %d pixels detected",
                    color.color_name.c_str(), pixel_count);

        // 检测保留检测装甲板类型
        auto armors = detect_armor(image, mask);

        // 合并结果
        all_targets.insert(all_targets.end(), armors.begin(), armors.end());
    }
    // 显示合并的调试掩码
    imshow("All Colors Mask", debug_mask);
    waitKey(1);

    return all_targets;
}

void VisionNode::camera_callback(sensor_msgs::msg::Image::SharedPtr msg)
{
    try
    {
        // 转换ROS图像消息到OpenCV图像
        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        Mat image = cv_ptr->image;

        if (image.empty())
        {
            RCLCPP_WARN(this->get_logger(), "Received empty image");
            return;
        }

        Mat result_image = image.clone();
        auto detected_objects = detect_targets(result_image);

        // 构造MultiObject消息
        referee_pkg::msg::MultiObject multi_object_msg;
        multi_object_msg.header = msg->header; // 使用图像消息的时间戳
        multi_object_msg.num_objects = detected_objects.size();

        // 设置目标信息
        for (const auto &obj : detected_objects)
        {
            referee_pkg::msg::Object obj_msg;
            obj_msg.target_type = obj.target_type;

            // 填充角点
            for (const auto &corner : obj.corners)
            {
                geometry_msgs::msg::Point point;
                point.x = corner.x;
                point.y = corner.y;
                point.z = corner.z; // 设置z=0，因为这是2D检测
                obj_msg.corners.push_back(point);
            }

            multi_object_msg.objects.push_back(obj_msg);
        }

        // 发布MultiObject消息
        result_publisher->publish(multi_object_msg);

        // 可视化处理后的图像
        imshow("Detected Result", result_image);
        waitKey(1);

        // 统计不同类型目标数量
        int armors = 0;
        for (const auto &obj : detected_objects)
        {
            if (obj.target_type.find("armor") != string::npos)
                armors++;
        }

        RCLCPP_INFO(this->get_logger(), "Detected %d objects:  %d armor, publishing to /vision/target",
                    multi_object_msg.num_objects, armors);

        // 调试信息（显示目标类型）
        RCLCPP_INFO(this->get_logger(), "Before sorting: %zu objects", detected_objects.size());
        for (const auto &obj : detected_objects)
        {
            RCLCPP_INFO(this->get_logger(), "  - %s", obj.target_type.c_str());
        }
    }
    catch (const exception &e)
    {
        RCLCPP_ERROR(this->get_logger(), "Processing error: %s", e.what());
    }
}

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = make_shared<VisionNode>("my_vision_node");
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}