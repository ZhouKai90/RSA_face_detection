#ifndef __RSA_FACE_DETECTION_API_HH__
#define __RSA_FACE_DETECTION_API_HH__

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

typedef void *faceDetectorHandle;
typedef int infResult;

#define INF_SUCCESS     0
#define INF_FAIL        -1

struct faceLandmark 
{
    std::vector<double> bbox;
    std::vector<cv::Point2f> keyPoints;
    float score; 
};

typedef std::vector<struct faceLandmark> facesLandmarkPerImg;
typedef std::vector<facesLandmarkPerImg> imgsFaceLandmarkList;

/*************************************************
Function: inf_face_detection_init
Description: 人脸检测模块初始化
Input:  modelPath 模型和训练完成的权值文件
        gpuId　显卡编号，可支持多卡
Output: 无
Return: 正常返回faceDetectorHandle不为空
Others: 
*************************************************/
faceDetectorHandle inf_face_detection_init(const std::string &modelPath, int gpuId = 0);
/*************************************************
Function: inf_face_detection_uint
Description: 人脸检测模块销毁
Input:  handle 人脸检测模块操作句柄
Output: 无
Return: infResult
Others: 
*************************************************/
infResult inf_face_detection_uint(faceDetectorHandle handle);
/*************************************************
Function: inf_face_landmark_detected
Description: 人脸检测
Input:  handle 人脸检测模块操作句柄
        imgs_mat　待识别的图像文件
Output: face_attr_feature   图像中人脸框．５个关键点信息
Return: infResult
Others: 
*************************************************/
infResult inf_face_landmark_detected(faceDetectorHandle handle, const std::vector<cv::Mat>& imgsMat, imgsFaceLandmarkList &list);


#endif