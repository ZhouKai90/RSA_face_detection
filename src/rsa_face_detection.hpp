#ifndef _RSA_FACE_DETECTED_HPP_
#define _RSA_FACE_DETECTED_HPP_
 
#include <vector>
#include <string>
#include <boost/smart_ptr.hpp>
#include <boost/make_shared.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "caffe/caffe.hpp"
#include "config.hpp"
#include <glog/logging.h>

#include "inf_rsa_face_detection_api.hh"

class RsaFaceDetector {
    public:
        explicit RsaFaceDetector(unsigned int gpuId, \
                    const std::string & sfnNet, const std::string & sfnWeight, \
                    const std::string & rsaNet, const std::string & rsaWeight, \
                    const std::string & lrnNet, const std::string & lrnWeight);
        ~RsaFaceDetector() {};
        imgsFaceLandmarkList detect_(std::vector<cv::Mat> iamges);
        facesLandmarkPerImg detect_(cv::Mat image);
        void sfnProcess_(const std::vector<cv::Mat> images);
        void sfnProcess_(const cv::Mat & img);
        void rsaProcess_(void);
        void lrnProcess_(imgsFaceLandmarkList & faceResult);
        void lrnProcess_(facesLandmarkPerImg & faceResult);

    private:
        std::string sfnNetDef_;
        std::string sfnNetWeight_;
        std::string rsaNetDef_;
        std::string rsaNetWeight_;
        std::string lrnNetDef_;
        std::string lrnNetWeight_;
        unsigned int gpuId_;

        std::vector<std::shared_ptr<caffe::Blob<float>>> transFeatMaps_;
        caffe::Blob<float> *sfnNetOutput_;
        caffe::Blob<float> *inputLayer_;
        caffe::Blob<float> *rsaInputLayer_;
        caffe::Blob<float> *lrnInputLayer_;
        std::vector<cv::Mat> inputChannels_;

        double resizeFactor_;
        std::vector<float> anchorBoxLen_;
        double threshScore_;
        double stride_;
        double anchorCenter_;
        std::vector<int> scale_;
        std::shared_ptr<caffe::Net<float>> sfnNet_;
        std::shared_ptr<caffe::Net<float>> rsaNet_;
        std::shared_ptr<caffe::Net<float>> lrnNet_;
};

#endif