#ifndef IMAGE_DEAL_H
#define IMAGE_DEAL_H

#include <string>
#include <memory>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <referee_pkg/msg/multi_object.hpp>
#include <referee_pkg/msg/object.hpp>
#include <cv_bridge/cv_bridge.h>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/timer.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <std_msgs/msg/header.hpp>

#include "shape_tools.h"

class vision_node : public rclcpp::Node
{
public:
  // 构造函数
  vision_node(std::string name) : Node(name)
  {
    RCLCPP_INFO(this->get_logger(), "%s节点已经启动.", name.c_str());

    Image_sub = this->create_subscription<sensor_msgs::msg::Image>(
        "/camera/image_raw", 10,
        bind(&vision_node::callback_camera, this, std::placeholders::_1));

    Target_pub = this->create_publisher<referee_pkg::msg::MultiObject>(
        "/vision/target", 10);

    RCLCPP_INFO(this->get_logger(), "vision_node 启动成功");
  }

  // 析构函数
  ~vision_node() { cv::destroyWindow("Detection Result"); }

private:
  void callback_camera(sensor_msgs::msg::Image::SharedPtr msg);

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr Image_sub;
  rclcpp::Publisher<referee_pkg::msg::MultiObject>::SharedPtr Target_pub;
  std::vector<cv::Point2f> Point_V;
};

#endif