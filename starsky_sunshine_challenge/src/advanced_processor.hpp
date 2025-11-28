#pragma once
#include <vector>
#include <algorithm>
#include <cmath>
#include <referee_pkg/msg/multi_object.hpp>
#include <referee_pkg/msg/object.hpp>
#include <geometry_msgs/msg/point.hpp>

class AdvancedProcessor
{
public:
    // Level 2: 多目标排序
    static std::vector<referee_pkg::msg::Object> sort_targets_by_priority(
        const std::vector<referee_pkg::msg::Object> &targets);

    // Level 2: 应用得分权重
    static std::vector<referee_pkg::msg::Object> apply_score_weights(
        const std::vector<referee_pkg::msg::Object> &targets);

    // Level 3: 简单弹道计算 - 修复：使用double类型
    static bool calculate_trajectory(
        const std::vector<geometry_msgs::msg::Point> &model_points,
        double gravity,
        double &yaw, double &pitch, double &roll);
};