#include "shape_tools.h"

std::vector<cv::Point2f> shape_tools::calculateStableSpherePoints(const cv::Point2f &center,float radius)
{
  std::vector<cv::Point2f> points;

  // 简单稳定的几何计算，避免漂移
  // 左、下、右、上
  points.push_back(cv::Point2f(center.x - radius, center.y)); // 左点 (1)
  points.push_back(cv::Point2f(center.x, center.y + radius)); // 下点 (2)
  points.push_back(cv::Point2f(center.x + radius, center.y)); // 右点 (3)
  points.push_back(cv::Point2f(center.x, center.y - radius)); // 上点 (4)

  return points;
}