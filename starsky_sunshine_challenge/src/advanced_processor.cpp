#include "advanced_processor.hpp"
#include <rclcpp/rclcpp.hpp>
#include <cmath>
#include <vector>
using namespace std;

// Level 2: 多目标排序 - 装甲板优先
vector<referee_pkg::msg::Object> AdvancedProcessor::sort_targets_by_priority(
    const vector<referee_pkg::msg::Object> &targets)
{
    // 保持原有代码不变
    vector<referee_pkg::msg::Object> sorted_targets = targets;

    sort(sorted_targets.begin(), sorted_targets.end(),
         [](const referee_pkg::msg::Object &a, const referee_pkg::msg::Object &b)
         {
             int priority_a = 0, priority_b = 0;

             if (a.target_type.find("armor") != string::npos)
                 priority_a = 3;
             else if (a.target_type == "sphere")
                 priority_a = 2;
             else if (a.target_type == "rect")
                 priority_a = 1;

             if (b.target_type.find("armor") != string::npos)
                 priority_b = 3;
             else if (b.target_type == "sphere")
                 priority_b = 2;
             else if (b.target_type == "rect")
                 priority_b = 1;

             return priority_a > priority_b;
         });

    return sorted_targets;
}

// Level 2: 应用得分权重 - 复制目标以模拟权重
vector<referee_pkg::msg::Object> AdvancedProcessor::apply_score_weights(
    const vector<referee_pkg::msg::Object> &targets)
{
    // 保持原有代码不变
    vector<referee_pkg::msg::Object> weighted_targets;

    for (const auto &target : targets)
    {
        int copies = 1;

        if (target.target_type.find("armor") != string::npos)
        {
            copies = 2;
        }
        else if (target.target_type == "sphere")
        {
            copies = 1;
        }
        else if (target.target_type == "rect")
        {
            copies = 1;
        }

        for (int i = 0; i < copies; i++)
        {
            weighted_targets.push_back(target);
        }
    }

    return weighted_targets;
}

// Level 3: 简单弹道计算 - 修复：使用double类型
bool AdvancedProcessor::calculate_trajectory(
    const vector<geometry_msgs::msg::Point> &model_points,
    double gravity, double &yaw, double &pitch, double &roll) // 改为double
{
    // 简单实现：直接指向目标中心
    if (model_points.size() >= 4)
    {
        // 计算目标中心
        double center_x = 0, center_y = 0; // 改为double
        for (const auto &point : model_points)
        {
            center_x += point.x;
            center_y += point.y;
        }
        center_x /= model_points.size();
        center_y /= model_points.size();

        // 简单计算角度
        yaw = atan2(center_x - 320, 500) * 180 / M_PI;
        pitch = atan2(center_y - 240, 500) * 180 / M_PI;
        roll = 0.0;

        return true;
    }

    return false;
}