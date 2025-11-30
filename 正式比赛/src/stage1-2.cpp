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
    vector<referee_pkg::msg::Object> detect_ring(const cv::Mat &image, const cv::Mat &mask);
    vector<Point2f> calculate_ring_points(const Point2f &center, float radius);
    vector<referee_pkg::msg::Object> detect_arrow(const cv::Mat &image, const cv::Mat &mask);
    vector<Point2f> sort_rectangle_points(Point2f vertices[4], float rect_angle);

    // 新增函数
    Point2f calculate_arrow_direction(const vector<Point> &approx);
    RotatedRect calculate_oriented_bounding_box(const vector<Point> &points, const Point2f &direction);

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

public:
    VisionNode(const string &name) : Node(name)
    {
        RCLCPP_INFO(this->get_logger(), "Initializing Vision_node");

        // 初始化颜色阈值 - 主要针对红色圆环
        color_thresholds_ = {
            {Scalar(0, 120, 60), Scalar(10, 255, 255), "red1"},
            {Scalar(170, 120, 70), Scalar(180, 255, 255), "red2"},
            {Scalar(80, 100, 100), Scalar(100, 255, 255), "cyan"},
            {Scalar(40, 50, 50), Scalar(85, 230, 255), "green"},
            {Scalar(100, 50, 50), Scalar(130, 255, 255), "blue"}};

        camera_subscriber = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera/image_raw", 10,
            bind(&VisionNode::camera_callback, this, placeholders::_1));

        result_publisher = this->create_publisher<referee_pkg::msg::MultiObject>(
            "/vision/target", 10);

        namedWindow("Detected Result", WINDOW_AUTOSIZE);
        RCLCPP_INFO(this->get_logger(), "Vision_Node initialized successfully");
    }

    ~VisionNode()
    {
        destroyWindow("Detected Result");
    }
};

// 修改：圆环检测函数 - 基于原有的球体检测逻辑
vector<referee_pkg::msg::Object> VisionNode::detect_ring(const cv::Mat &image, const cv::Mat &mask)
{
    vector<referee_pkg::msg::Object> rings;
    vector<vector<Point>> contours;
    findContours(mask, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

    // 存储所有检测到的圆形
    vector<tuple<Point2f, float, double>> detected_circles; // (center, radius, circularity)

    // 第一遍：检测所有可能的圆形（使用原有的检测逻辑）
    for (const auto &contour : contours)
    {
        double area = contourArea(contour);
        if (area < 500) // 面积阈值
            continue;

        // 计算最小外接圆
        Point2f center;
        float radius;
        minEnclosingCircle(contour, center, radius);

        // 计算圆形度
        double perimeter = arcLength(contour, true);
        double circularity = 4 * CV_PI * area / (perimeter * perimeter);

        // 使用原有的过滤条件
        if (circularity > 0.7 && radius > 15 && radius < 200)
        {
            detected_circles.push_back(make_tuple(center, radius, circularity));

            // 可视化：绘制所有检测到的圆形（调试用）
            circle(image, center, static_cast<int>(radius), Scalar(0, 255, 0), 2);

            // 显示半径和圆形度信息
            string info_text = "R:" + to_string((int)radius) + " C:" + to_string(circularity).substr(0, 4);
            putText(image, info_text, Point(center.x - 25, center.y - 15),
                    FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 2);
        }
    }

    // 第二遍：配对内外圆（圆环检测）
    // 按半径从大到小排序，便于配对
    sort(detected_circles.begin(), detected_circles.end(),
         [](const auto &a, const auto &b)
         { return get<1>(a) > get<1>(b); });

    vector<bool> used(detected_circles.size(), false);

    for (size_t i = 0; i < detected_circles.size(); i++)
    {
        if (used[i])
            continue;

        auto [center1, radius1, circularity1] = detected_circles[i];

        // 寻找匹配的内圆
        for (size_t j = i + 1; j < detected_circles.size(); j++)
        {
            if (used[j])
                continue;

            auto [center2, radius2, circularity2] = detected_circles[j];

            // 计算两个圆心的距离
            double center_distance = norm(center1 - center2);

            // 圆环配对条件：
            // 1. 圆心距离很近（同心或近似同心）
            // 2. 半径差异适中（形成环状）
            // 3. 半径大的为外圆，小的为内圆
            if (center_distance < 20.0 && radius1 > radius2 * 1.1)
            {
                Point2f outer_center = center1;
                float outer_radius = radius1;
                Point2f inner_center = center2;
                float inner_radius = radius2;

                RCLCPP_INFO(this->get_logger(),
                            "Ring detected: outer_radius=%.1f, inner_radius=%.1f, center_distance=%.1f",
                            outer_radius, inner_radius, center_distance);

                // 创建圆环消息 - 按照比赛要求：objects[0]为外圆，objects[1]为内圆
                referee_pkg::msg::Object outer_obj;
                referee_pkg::msg::Object inner_obj;

                outer_obj.target_type = "Ring_red";
                inner_obj.target_type = "Ring_red";

                // 计算外圆和内圆的角点
                auto outer_points = calculate_ring_points(outer_center, outer_radius);
                auto inner_points = calculate_ring_points(inner_center, inner_radius);

                // 填充外圆角点
                for (const auto &point : outer_points)
                {
                    geometry_msgs::msg::Point corner;
                    corner.x = point.x;
                    corner.y = point.y;
                    corner.z = 0.0;
                    outer_obj.corners.push_back(corner);
                }

                // 填充内圆角点
                for (const auto &point : inner_points)
                {
                    geometry_msgs::msg::Point corner;
                    corner.x = point.x;
                    corner.y = point.y;
                    corner.z = 0.0;
                    inner_obj.corners.push_back(corner);
                }

                // 按照比赛要求顺序添加：外圆在前，内圆在后
                rings.push_back(outer_obj);
                rings.push_back(inner_obj);

                // 标记为已使用
                used[i] = true;
                used[j] = true;

                // 可视化：用不同颜色绘制内外圆
                circle(image, outer_center, static_cast<int>(outer_radius), Scalar(255, 0, 0), 3); // 蓝色-外圆
                circle(image, inner_center, static_cast<int>(inner_radius), Scalar(0, 0, 255), 3); // 红色-内圆

                // 标注内外圆信息
                string outer_text = "Outer R:" + to_string((int)outer_radius);
                string inner_text = "Inner R:" + to_string((int)inner_radius);
                putText(image, outer_text, Point(outer_center.x - 35, outer_center.y - 30),
                        FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 0, 0), 2);
                putText(image, inner_text, Point(inner_center.x - 35, inner_center.y + 40),
                        FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 255), 2);

                // 标注角点序号（
                vector<Scalar> colors = {Scalar(0, 255, 255), Scalar(255, 255, 0),
                                         Scalar(255, 0, 255), Scalar(0, 255, 0)};

                // 外圆角点标注
                for (int k = 0; k < 4; k++)
                {
                    circle(image, outer_points[k], 8, colors[k], -1);
                    string label = "O" + to_string(k + 1);
                    putText(image, label, Point(outer_points[k].x + 12, outer_points[k].y - 12),
                            FONT_HERSHEY_SIMPLEX, 0.6, colors[k], 2);
                }

                // 内圆角点标注
                for (int k = 0; k < 4; k++)
                {
                    circle(image, inner_points[k], 6, colors[k], -1);
                    string label = "I" + to_string(k + 1);
                    putText(image, label, Point(inner_points[k].x + 12, inner_points[k].y - 12),
                            FONT_HERSHEY_SIMPLEX, 0.5, colors[k], 2);
                }

                // 计算并显示圆心距离（比赛评分项）
                double center_error = norm(outer_center - inner_center);
                string distance_text = "CenterErr:" + to_string(center_error).substr(0, 5);
                putText(image, distance_text, Point((outer_center.x + inner_center.x) / 2 - 40, (outer_center.y + inner_center.y) / 2),
                        FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 2);

                RCLCPP_INFO(this->get_logger(), "Ring center error: %.3f", center_error);

                break; // 找到配对后跳出内层循环
            }
        }
    }

    return rings;
}

// 圆环角点计算函数
vector<Point2f> VisionNode::calculate_ring_points(const Point2f &center, float radius)
{
    vector<Point2f> points;
    // 按照比赛要求：左点→下点→右点→上点（逆时针方向）
    points.push_back(Point2f(center.x - radius, center.y)); // 左点 (1)
    points.push_back(Point2f(center.x, center.y + radius)); // 下点 (2)
    points.push_back(Point2f(center.x + radius, center.y)); // 右点 (3)
    points.push_back(Point2f(center.x, center.y - radius)); // 上点 (4)

    return points;
}

// 主检测函数
vector<referee_pkg::msg::Object> VisionNode::detect_targets(const cv::Mat &image)
{
    vector<referee_pkg::msg::Object> all_targets;

    Mat hsv;
    cvtColor(image, hsv, COLOR_BGR2HSV);

    imshow("HSV Image", hsv);
    Mat debug_mask = Mat::zeros(image.size(), CV_8UC1);

    for (const auto &color : color_thresholds_)
    {
        Mat mask;
        inRange(hsv, color.lower, color.upper, mask);
        debug_mask = debug_mask | mask;

        if (color.color_name == "red1" || color.color_name == "red2")
        {
            Mat kernel_dilate = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
            Mat kernel_erode = getStructuringElement(MORPH_ELLIPSE, Size(2, 2));
            morphologyEx(mask, mask, MORPH_DILATE, kernel_dilate);
            morphologyEx(mask, mask, MORPH_ERODE, kernel_erode);
        }
        else
        {
            Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
            morphologyEx(mask, mask, MORPH_CLOSE, kernel);
            morphologyEx(mask, mask, MORPH_OPEN, kernel);
        }

        imshow("Mask " + color.color_name, mask);

        if (color.color_name == "red1" || color.color_name == "red2")
        {
            auto rings = detect_ring(image, mask);
            all_targets.insert(all_targets.end(), rings.begin(), rings.end());

            auto arrows = detect_arrow(image, mask);
            all_targets.insert(all_targets.end(), arrows.begin(), arrows.end());
        }
        else
        {
            auto rings = detect_ring(image, mask);
            all_targets.insert(all_targets.end(), rings.begin(), rings.end());
        }
    }

    imshow("All Colors Mask", debug_mask);
    waitKey(1);
    return all_targets;
}

// 箭头检测函数
vector<referee_pkg::msg::Object> VisionNode::detect_arrow(const cv::Mat &image, const cv::Mat &mask)
{
    vector<referee_pkg::msg::Object> arrows;
    vector<vector<Point>> contours;
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    for (size_t idx = 0; idx < contours.size(); idx++)
    {
        const auto &contour = contours[idx];
        double area = contourArea(contour);

        if (area < 50 || area > 2000)
            continue;

        vector<Point> approx;
        double epsilon = 0.02 * arcLength(contour, true);
        approxPolyDP(contour, approx, epsilon, true);

        if (approx.size() < 4)
            continue;

        Point2f direction = calculate_arrow_direction(approx);
        RotatedRect oriented_rect = calculate_oriented_bounding_box(approx, direction);

        referee_pkg::msg::Object arrow_obj;
        arrow_obj.target_type = "arrow";

        Point2f vertices[4];
        oriented_rect.points(vertices);
        vector<Point2f> sorted_vertices = sort_rectangle_points(vertices, oriented_rect.angle);

        for (const auto &point : sorted_vertices)
        {
            geometry_msgs::msg::Point corner;
            corner.x = point.x;
            corner.y = point.y;
            corner.z = 0.0;
            arrow_obj.corners.push_back(corner);
        }

        arrows.push_back(arrow_obj);
    }

    return arrows;
}

// 计算箭头方向
Point2f VisionNode::calculate_arrow_direction(const vector<Point> &approx)
{
    double max_length = 0;
    Point2f direction(0, 0);

    for (size_t i = 0; i < approx.size(); i++)
    {
        Point2f p1 = approx[i];
        Point2f p2 = approx[(i + 1) % approx.size()];
        Point2f edge = p2 - p1;
        double length = norm(edge);

        if (length > max_length)
        {
            max_length = length;
            direction = edge;
        }
    }

    double dir_length = norm(direction);
    if (dir_length > 0)
    {
        direction.x /= dir_length;
        direction.y /= dir_length;
    }

    return direction;
}

// 基于方向计算外接矩形
RotatedRect VisionNode::calculate_oriented_bounding_box(const vector<Point> &points, const Point2f &direction)
{
    Point2f perpendicular(-direction.y, direction.x);
    vector<float> main_proj, perp_proj;

    Point2f center(0, 0);
    for (const auto &p : points)
    {
        center += Point2f(p.x, p.y);
        main_proj.push_back(p.x * direction.x + p.y * direction.y);
        perp_proj.push_back(p.x * perpendicular.x + p.y * perpendicular.y);
    }
    center.x /= points.size();
    center.y /= points.size();

    auto min_main = *min_element(main_proj.begin(), main_proj.end());
    auto max_main = *max_element(main_proj.begin(), main_proj.end());
    auto min_perp = *min_element(perp_proj.begin(), perp_proj.end());
    auto max_perp = *max_element(perp_proj.begin(), perp_proj.end());

    float width = max_main - min_main;
    float height = max_perp - min_perp;
    float angle = atan2(direction.y, direction.x) * 180 / CV_PI;

    return RotatedRect(center, Size2f(width, height), angle);
}

// 修复的角点排序函数 - 避免std::pair比较问题
vector<Point2f> VisionNode::sort_rectangle_points(Point2f vertices[4], float rect_angle)
{
    vector<Point2f> points(vertices, vertices + 4);
    Point2f center(0, 0);
    for (const auto &p : points)
        center += p;
    center.x /= 4;
    center.y /= 4;

    double angle_rad = rect_angle * CV_PI / 180.0;
    Point2f main_dir(cos(angle_rad), sin(angle_rad));
    Point2f perp_dir(-main_dir.y, main_dir.x);

    // 使用结构体来避免pair比较问题
    struct PointProjection
    {
        double main_proj;
        double perp_proj;
        Point2f point;
        int index;
    };

    vector<PointProjection> projections;
    for (int i = 0; i < 4; i++)
    {
        Point2f vec = points[i] - center;
        projections.push_back({vec.x * main_dir.x + vec.y * main_dir.y,
                               vec.x * perp_dir.x + vec.y * perp_dir.y,
                               points[i],
                               i});
    }

    // 排序函数
    sort(projections.begin(), projections.end(),
         [](const PointProjection &a, const PointProjection &b)
         {
             if (a.main_proj != b.main_proj)
                 return a.main_proj < b.main_proj;
             return a.perp_proj < b.perp_proj;
         });

    vector<Point2f> sorted_points(4);
    for (const auto &proj : projections)
    {
        if (proj.main_proj < 0 && proj.perp_proj < 0)
            sorted_points[3] = proj.point; // 左下
        else if (proj.main_proj < 0 && proj.perp_proj > 0)
            sorted_points[2] = proj.point; // 右下
        else if (proj.main_proj > 0 && proj.perp_proj > 0)
            sorted_points[1] = proj.point; // 右上
        else if (proj.main_proj > 0 && proj.perp_proj < 0)
            sorted_points[0] = proj.point; // 左上
    }

    return sorted_points;
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
                point.z = corner.z;
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
        int ring_pairs = 0;
        for (const auto &obj : detected_objects)
        {
            if (obj.target_type == "Ring_red")
                ring_pairs++;
        }
        ring_pairs = ring_pairs / 2; // 每个圆环包含2个对象

        RCLCPP_INFO(this->get_logger(),
                    "Detected %d objects: %d ring pairs, publishing to /vision/target",
                    multi_object_msg.num_objects, ring_pairs);

        // 调试信息
        RCLCPP_INFO(this->get_logger(), "Detected objects: %zu", detected_objects.size());
        for (const auto &obj : detected_objects)
        {
            RCLCPP_INFO(this->get_logger(), "  - %s with %zu corners",
                        obj.target_type.c_str(), obj.corners.size());
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