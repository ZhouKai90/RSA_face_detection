#ifndef __IMAGE_FILE_OPT_HPP__
#define __IMAGE_FILE_OPT_HPP__
#include <string>
#include <vector>
#include "opencv2/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"


std::vector<std::string> readImageToCvMap(const std::string dirPath, std::vector<cv::Mat> &imageMatList, bool Isrand, unsigned int cnt = 0);
int getAllImageName_xfs(std::string dirPath, std::vector<std::string> &images);
int video_capture(const std::string &video_path);
#endif