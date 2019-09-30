#include "inf_rsa_face_detection_api.hh"
#include "rsa_face_detection.hpp"
#include <fstream>
faceDetectorHandle inf_face_detection_init(const std::string &modelPath, int gpuId)
{
    const std::string sfnNet = modelPath + "/model_def/res_pool2/test.prototxt";
    const std::string sfnWeight = modelPath + "/model_weight/ResNet_3b_s16/tot_wometa_1epoch";
    const std::string rsaNet = modelPath + "/model_def/ResNet_3b_s16_fm2fm_pool2_deep/test.prototxt";
    const std::string rsaWeight = modelPath + "/model_weight/ResNet_3b_s16_fm2fm_pool2_deep/65w";
    const std::string lrnNet = modelPath + "/model_def/ResNet_3b_s16_f2r/test.prototxt";
    const std::string lrnWeight = modelPath + "/model_weight/ResNet_3b_s16/tot_wometa_1epoch";

    RsaFaceDetector *detector = new RsaFaceDetector(gpuId, sfnNet, sfnWeight, 
                                                    rsaNet, rsaWeight, lrnNet, lrnWeight);
    std::cout << "Init face detection model succeed." << std::endl;                                                
    return (faceDetectorHandle) detector;
}

infResult inf_face_detection_uint(faceDetectorHandle handle)
{
    if (handle != NULL) {
        RsaFaceDetector *detector = (RsaFaceDetector*) handle;
        delete detector;
    }
    return INF_SUCCESS;
}


infResult inf_face_landmark_detected(faceDetectorHandle handle, \
                                    const std::vector<cv::Mat>& imgsMat, imgsFaceLandmarkList &list)
{
    if (handle == NULL)
        return INF_FAIL;
    RsaFaceDetector *detector = (RsaFaceDetector*) handle;

    for (auto perImgMat : imgsMat) {
        facesLandmarkPerImg landmark = detector->detect_(perImgMat);
        // int faceCnt = 0;
        // for (auto face : faces) {
        //     std::cout << "face["<< ++faceCnt << "]:"<<std::endl;
        //     std::cout << "quality:" << face.score << std::endl;
        //     std::cout << "face box:[" << (float)(face.bbox[0]) << "," << (float)(face.bbox[1]) << "," \
        //                         << (float)(face.bbox[2]) << ","<< (float)(face.bbox[3]) << "]" << std::endl;
        // }

        // plot_face_rect_keypoint(image, bitMap, faces, false);
        // crop_face(image, bitMap, faces, false);
        list.push_back(landmark);
        landmark.clear();
    }
    return INF_SUCCESS;
}

