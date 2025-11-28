FROM vision-vrena-2025:v0.1.2

COPY ./install /home/va/Vision-Vrena-2025/install
COPY ./src/player_pkg /home/va/Vision-Vrena-2025/src/player_pkg
COPY ./results /home/va/Vision-Vrena-2025/results

WORKDIR /home/va/Vision-Vrena-2025

RUN apt-get update && apt-get upgrade -y && \
    mkdir -p src/referee_pkg/results && \
    apt-get install -y ros-humble-xacro ros-humble-gazebo-ros-pkgs

# 先激活环境，再编译
RUN bash -c "source /opt/ros/humble/setup.bash && source install/setup.bash && colcon build --packages-select player_pkg"
