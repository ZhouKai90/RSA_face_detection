#include <iostream>
#include <string>
#include <sstream>
#include <chrono>
#include "image_file_opt.hpp"
#include "inf_rsa_face_detection_api.hh"


#define FACE_BOX_WIDTH 112
#define FACE_BOX_HEIGTH 152

const std::string ROOT_PATH = "/kyle/workspace/project/RSA/";

void plot_face_rect_keypoint(const std::string landmaerk_path, std::string img_name, const cv::Mat & img_mat, 
                            const facesLandmarkPerImg &landmarks, bool keyPoints = true)
{
    std::size_t index1 = img_name.find_last_of('/');
    std::string name = img_name.substr(index1, img_name.length());

    cv::Mat mat = img_mat.clone();

    bool first_face = true;
    for (auto ldPerImg : landmarks) {
        if (keyPoints) {
        // 图片画出关键点
            for (int j = 0; j < 5; j++) 
                cv::circle(mat, ldPerImg.keyPoints[j], 1, cv::Scalar(0, 0, 255), 1);
        }
        // 图片画出人脸框
        if (first_face) {
            first_face = false;
            cv::rectangle(mat, cv::Point(ldPerImg.bbox[0], ldPerImg.bbox[1]), \
                            cv::Point(ldPerImg.bbox[2], ldPerImg.bbox[3]), \
                            cv::Scalar(0, 0, 255), 1, 1, 0);
        } else {
            cv::rectangle(mat, cv::Point(ldPerImg.bbox[0], ldPerImg.bbox[1]), \
                            cv::Point(ldPerImg.bbox[2], ldPerImg.bbox[3]), \
                            cv::Scalar(0, 0, 255), 1, 1, 0);
            continue;
        }
    }
    cv::imwrite(landmaerk_path + name, mat);
}

void crop_face(const std::string &cropPath, std::string img_name, const cv::Mat & img_mat, const facesLandmarkPerImg &faces, bool single_face)
{
    int face_index = 1;
    for (auto face : faces) {
        cv::Point2f orig_left_top(face.bbox[0], face.bbox[1]);
        cv::Point2f orig_right_bottom(face.bbox[2], face.bbox[3]);

        double rect_w = orig_right_bottom.x - orig_left_top.x;
        double rect_h = orig_right_bottom.y - orig_left_top.y;

        double exp_letf = std::max(0., orig_left_top.x - 0.5 * rect_w);
        double exp_top = std::max(0., orig_left_top.y - 1.3 * rect_h);
        double exp_right = std::min((double)img_mat.cols, orig_right_bottom.x + 0.5 * rect_w);
        double exp_bottom = std::min((double)img_mat.rows, orig_right_bottom.y + 0.5 * rect_h);

        // std::cout << "Point(" << orig_left_top.x << ',' << orig_left_top.y << "," << orig_right_bottom.x << ',' << orig_right_bottom.y << ')' << std::endl;
        // std::cout << "rect_w: " << rect_w << "  " << "rect_h: " << rect_h << std::endl;
        // std::cout << "Point(" << exp_letf << ',' << exp_top << ") (" << exp_right << ',' << exp_bottom << ')' << std::endl;
        // std::cout << "(" << exp_right - exp_letf << "," << exp_bottom - exp_top << ")" << std::endl;
        cv::Rect rect(static_cast<int> (exp_letf), 
            static_cast<int> (exp_top), 
            static_cast<int> (exp_right - exp_letf),
            static_cast<int> (exp_bottom - exp_top));

        std::size_t index1 = img_name.find_last_of('/');
        std::string name = img_name.substr(index1, img_name.length());
        std::string out_put_name;
        if (single_face) {
            out_put_name = cropPath + name;
        } else {
            std::size_t index2 = name.find_last_of('.');
            std::stringstream ss;
            ss << '_' << face_index++;
            std::string add;
            ss >> add;
            out_put_name = cropPath + name.insert(index2, add);
        }
        // std::cout << output << std::endl;
        cv::Mat mat = img_mat.clone();
        cv::Mat ROI = mat(rect);
        cv::Size resize(FACE_BOX_WIDTH, FACE_BOX_HEIGTH);
        // cv::Size resize(112, 112 * ((exp_bottom - exp_top)/(exp_right - exp_letf)));
        cv::Mat crop_mat; 
        cv::resize(ROI, crop_mat, resize);
        cv::imwrite(out_put_name, crop_mat);
        if (single_face)
            break;
    }
}

int main(int argc, char* argv[])
{
    if (argc < 1) {
        std::cout << "No images path" << std::endl;
        return -1;
    }
    std::string imgsPath(argv[1]);
    std::string dataset = imgsPath + "/original";
    std::string landmark_path = imgsPath + "/landmark";
    std::string cropPath = imgsPath + "/crop";

    std::string modelPath = ROOT_PATH + "model";
    faceDetectorHandle handle = inf_face_detection_init(modelPath, 0);
    std::vector<std::string> imgsName; 
    getAllImageName_xfs(dataset, imgsName);
    for (auto imgName: imgsName) {
        std::cout << ">>>>>>>>>" << imgName << "<<<<<<<<<" << std::endl;
        cv::Mat imgMat = cv::imread(imgName);
        if (imgMat.empty()) {
            std::cout << "Decode image failed." << std::endl;
            continue;
        }
        std::vector<cv::Mat> imgsMatList;
        imgsMatList.push_back(imgMat);

        imgsFaceLandmarkList landMarkList;
        inf_face_landmark_detected(handle, imgsMatList, landMarkList);
      
        for (auto ldPerImg : landMarkList) {
            plot_face_rect_keypoint(landmark_path, imgName, imgMat, ldPerImg);
            // void crop_face(const std::string &cropPath, std::string img_name, const cv::Mat & img_mat, const facesLandmarkPerImg &faces, bool single_face)
            crop_face(cropPath, imgName, imgMat, ldPerImg, false);
        }
        imgsMatList.clear();
    }
	return 0;
}
