项目简介

基于ROS2和OpenCV的视觉识别系统，用于检测球体、矩形和装甲板目标。
快速开始
1. 环境配置
bash

# 安装依赖
sudo apt install ros-humble-vision-opencv
sudo apt install ros-humble-cv-bridge

2. 构建项目
bash

cd Vision_Arena_2025
colcon build
source install/setup.bash

3. 运行程序
bash

# 启动视觉节点
ros2 launch teamX_challenge vision.launch.py

# 或者直接运行
ros2 run teamX_challenge vision_node

文件结构
text

teamX_challenge/
├── CMakeLists.txt
├── package.xml
├── config/
│   └── params.yaml          # 参数配置文件
├── launch/
│   └── vision.launch.py     # 启动文件
└── src/
    └── vision_node.cpp      # 主程序

主要功能

     装甲板检测与数字识别

     球体检测

     矩形检测

     弹道计算服务

参数调整

编辑 config/params.yaml 调整颜色阈值、检测参数等。

