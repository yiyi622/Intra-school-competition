cmake_minimum_required(VERSION 3.8...3.10)
project(player_pkg)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 14)

# 查找依赖包
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(gazebo_msgs REQUIRED)
find_package(geometry_msgs REQUIRED) 
find_package(OpenCV REQUIRED)
find_package(pose_msg REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(referee_pkg REQUIRED)
find_package(sensor_msgs REQUIRED)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wno-deprecated-declarations)
endif()

#编译文件
add_executable(TestNode src/test.cpp)
add_executable(my_vision_node src/my_vision_node.cpp src/advanced_processor.cpp)
add_executable(debug src/debug.cpp src/advanced_processor.cpp)

# 依赖项
ament_target_dependencies(TestNode
  rclcpp
  std_msgs
  OpenCV 
  geometry_msgs
  cv_bridge
  pose_msg
  referee_pkg
  sensor_msgs
)

ament_target_dependencies(my_vision_node
  rclcpp
  std_msgs
  OpenCV 
  geometry_msgs
  cv_bridge
  pose_msg
  referee_pkg
  sensor_msgs
)
ament_target_dependencies(debug
  rclcpp
  std_msgs
  OpenCV 
  geometry_msgs
  cv_bridge
  pose_msg
  referee_pkg
  sensor_msgs
)
# 链接OpenCV库
target_link_libraries(TestNode ${OpenCV_LIBS})
target_link_libraries(my_vision_node ${OpenCV_LIBS})
target_link_libraries(debug ${OpenCV_LIBS})

# 包含OpenCV头文件
include_directories(${OpenCV_INCLUDE_DIRS})

# 安装
install(TARGETS TestNode my_vision_node debug
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
