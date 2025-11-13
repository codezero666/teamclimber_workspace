cmake_minimum_required(VERSION 3.8)
project(teamclimber_challenge)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(gazebo_msgs REQUIRED)
find_package(geometry_msgs REQUIRED) 
find_package(OpenCV REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(referee_pkg REQUIRED)
find_package(pose_msg REQUIRED)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

add_executable(vision_node 
  src/vision_node.cpp
  src/image_deal.cpp
  src/shape_tools.cpp)
  
ament_target_dependencies(vision_node
  rclcpp
  std_msgs
  OpenCV 
  geometry_msgs
  cv_bridge
  pose_msg
  referee_pkg
)

# 链接OpenCV库
target_link_libraries(vision_node ${OpenCV_LIBS})

# 为Vision_test添加消息类型支持（新方法）
target_link_libraries(vision_node ${typesupport_target})

install(TARGETS
vision_node
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME})

# 包含OpenCV头文件及自己编写的头文件
include_directories(
  include
  ${OpenCV_INCLUDE_DIRSee})

ament_package()
