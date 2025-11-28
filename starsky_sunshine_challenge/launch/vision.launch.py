from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # 获取包共享目录路径
    pkg_share_dir = get_package_share_directory('teamX_challenge')
    
    # 参数文件路径
    params_file = os.path.join(pkg_share_dir, 'config', 'params.yaml')
    
    return LaunchDescription([
        Node(
            package='teamX_challenge',
            executable='vision_node',
            name='vision_node',
            output='screen',
            parameters=[params_file],
            remappings=[
                ('/camera/image_raw', '/camera/image_raw'),  # 根据需要重映射
                ('/vision/target', '/vision/target')
            ]
        )
    ])
