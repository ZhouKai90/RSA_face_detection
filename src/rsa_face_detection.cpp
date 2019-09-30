#include "rsa_face_detection.hpp"
#include "gpu_nms.hpp"
#include <Eigen/LU>
#include <Eigen/Dense>
#include <chrono>

using milli = std::chrono::milliseconds;
static bool comp(const struct faceLandmark & a, const struct faceLandmark & b){
	return a.score > b.score;
}

static Eigen::MatrixXd findNonreflectiveSimilarity(const cv::Point2f uv[], const cv::Point2f xy[])
{
    Eigen::MatrixXd X(10, 4);
    Eigen::MatrixXd U(10, 1);
    for (int i = 0; i < 5; i++) {
        X(i, 0) = xy[i].x;
        X(i, 1) = xy[i].y;
        X(i, 2) = 1.0;
        X(i, 3) = 0.0;
        X(i + 5, 0) = xy[i].y;
        X(i + 5, 1) = -xy[i].x;
        X(i + 5, 2) = 0.0;
        X(i + 5, 3) = 1.0;

        U(i, 0) = uv[i].x;
        U(i + 5, 0) = uv[i].y;
    }

    Eigen::MatrixXd r = X.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(U);
    double sc = r(0, 0);
    double ss = r(1, 0);
    double tx = r(2, 0);
    double ty = r(3, 0);

    Eigen::MatrixXd Tinv(3, 3);
    Tinv(0, 0) = sc;
    Tinv(0, 1) = -ss;
    Tinv(0, 2) = 0.0;
    Tinv(1, 0) = ss;
    Tinv(1, 1) = sc;
    Tinv(1, 2) = 0.0;
    Tinv(2, 0) = tx;
    Tinv(2, 1) = ty;
    Tinv(2, 2) = 1.0;
    Eigen::MatrixXd T = Tinv.inverse();
    T(0, 2) = 0.0;
    T(1, 2) = 0.0;
    T(2, 2) = 1.0;
    return T;
}

static cv::Mat getSimilarityTransform(const cv::Point2f uv[], const cv::Point2f xy[])
{
	Eigen::MatrixXd trans1 = findNonreflectiveSimilarity(uv, xy);
	cv::Point2f xyNew[5];

	for (int i = 0; i < 5; ++i) {
		xyNew[i].x = -xy[i].x;
		xyNew[i].y = xy[i].y;
	}

	Eigen::MatrixXd trans2r = findNonreflectiveSimilarity(uv, xyNew);

	Eigen::MatrixXd TreflectY(3, 3);
	TreflectY(0, 0) = -1.0;
	TreflectY(0, 1) = 0.0;
	TreflectY(0, 2) = 0.0;
	TreflectY(1, 0) = 0.0;
	TreflectY(1, 1) = 1.0;
	TreflectY(1, 2) = 0.0;
	TreflectY(2, 0) = 0.0;
	TreflectY(2, 1) = 0.0;
	TreflectY(2, 2) = 1.0;

	Eigen::MatrixXd trans2 = trans2r * TreflectY;
	Eigen::MatrixXd trans1Inv = trans1.inverse();
	Eigen::MatrixXd trans2Inv = trans2.inverse();
	for (int i = 0; i < trans1Inv.rows() - 1; ++i) {
		trans1Inv(i, trans1.cols() - 1) = 0;
		trans2Inv(i, trans1.cols() - 1) = 0;
	}
	trans1(trans1Inv.rows() - 1, trans1.cols() - 1) = 1;
	trans2(trans2Inv.rows() - 1, trans1.cols() - 1) = 1;

	Eigen::MatrixXd matrixUv(5, 3),matrixXy(5, 3);
	for (int i = 0; i < 5; ++i) {
		matrixUv(i, 0) = uv[i].x;
		matrixUv(i, 1) = uv[i].y;
		matrixUv(i, 2) = 1;
		matrixXy(i, 0) = xy[i].x;
		matrixXy(i, 1) = xy[i].y;
		matrixXy(i, 2) = 1;
	}

	Eigen::MatrixXd trans1_block = trans1.block<3, 2>(0, 0);
	Eigen::MatrixXd trans2_block = trans2.block<3, 2>(0, 0);

	double norm1 = (matrixUv * trans1_block - matrixXy.block<5, 2>(0, 0)).norm();
	double norm2 = (matrixUv * trans2_block - matrixXy.block<5, 2>(0, 0)).norm();

	cv::Mat M(2, 3, CV_64F);
	double* m = M.ptr<double>();

	if(norm1 <= norm2){
		m[0] = trans1Inv(0, 0);
		m[1] = trans1Inv(1, 0);
		m[2] = trans1Inv(2, 0);
		m[3] = trans1Inv(0, 1);
		m[4] = trans1Inv(1, 1);
		m[5] = trans1Inv(2, 1);
	}
	else{
		m[0] = trans2Inv(0, 0);
		m[1] = trans2Inv(1, 0);
		m[2] = trans2Inv(2, 0);
		m[3] = trans2Inv(0, 1);
		m[4] = trans2Inv(1, 1);
		m[5] = trans2Inv(2, 1);
	}
	return M;
}

static void getTripPoints(std::vector<cv::Point2f> &dstRect, cv::Point2f srcKeyPoint[])
{
    cv::Point2f dstKeyPoint[5];
    dstKeyPoint[0] = cv::Point2f(0.2, 0.2);
    dstKeyPoint[1] = cv::Point2f(0.8, 0.2);
    dstKeyPoint[2] = cv::Point2f(0.5, 0.5);
    dstKeyPoint[3] = cv::Point2f(0.3, 0.75);
    dstKeyPoint[4] = cv::Point2f(0.7, 0.75);

    cv::Mat warpMat = getSimilarityTransform(srcKeyPoint, dstKeyPoint);
    std::vector<cv::Point2f> srcRect;
    srcRect.push_back(cv::Point2f(0.5, 0.5));
    srcRect.push_back(cv::Point2f(0, 0));
    srcRect.push_back(cv::Point2f(1.0, 0));
    for (int h = 0; h < 3; h++) {
        dstRect[h].x = srcRect[h].x * warpMat.ptr<double>(0)[0] + \
                        srcRect[h].y *warpMat.ptr<double>(0)[1] + \
                        warpMat.ptr<double>(0)[2];
        dstRect[h].y = srcRect[h].x * warpMat.ptr<double>(1)[0] + \
                        srcRect[h].y *warpMat.ptr<double>(1)[1] + \
                        warpMat.ptr<double>(1)[2];
    }

}

RsaFaceDetector::RsaFaceDetector(unsigned int gpuId, \
        const std::string & sfnNet, const std::string & sfnWeight, \
        const std::string & rsaNet, const std::string & rsaWeight, \
        const std::string & lrnNet, const std::string & lrnWeight)
        :gpuId_(gpuId),
        sfnNetDef_(sfnNet), sfnNetWeight_(sfnWeight),
        rsaNetDef_(rsaNet), rsaNetWeight_(rsaWeight),
        lrnNetDef_(lrnNet), lrnNetWeight_(lrnWeight)
{
    caffe::Caffe::set_mode(caffe::Caffe::GPU);
    caffe::Caffe::SetDevice(gpuId_);

    sfnNet_.reset(new caffe::Net<float>(sfnNet, caffe::TEST));
    sfnNet_->CopyTrainedLayersFrom(sfnWeight);

    rsaNet_.reset(new caffe::Net<float>(rsaNet, caffe::TEST));
    rsaNet_->CopyTrainedLayersFrom(rsaWeight);

    lrnNet_.reset(new caffe::Net<float>(lrnNet, caffe::TEST));
    lrnNet_->CopyTrainedLayersFrom(lrnWeight);

    inputLayer_ = sfnNet_->input_blobs()[0];            //sfn人脸像素大小预测网络输入接口
    rsaInputLayer_ = rsaNet_->input_blobs()[0];         //rsa网络 
    lrnInputLayer_ = lrnNet_->input_blobs()[0];         //lrn网络

    anchorBoxLen_.push_back(ANCHOR_BOX[2] - ANCHOR_BOX[0]);
    anchorBoxLen_.push_back(ANCHOR_BOX[3] - ANCHOR_BOX[1]);

    threshScore_ = THRESH_SCORE;
    stride_ = STRIDE;
    anchorCenter_ = ANCHOR_CENTER;

    for (int i = 5; i >= 1 ; i--)
        scale_.push_back(i);
}

void RsaFaceDetector::sfnProcess_(const cv::Mat &image)
{
    int width = image.cols;
    int height = image.rows;
    int channels = image.channels();

    if (width > height) {       // 以最长边为基准，等比例缩放图像
        resizeFactor_ = static_cast<double>(width) / static_cast<double>(MAX_IMG);
        height = static_cast<int>(MAX_IMG / static_cast<float>(width) * height);
        width = MAX_IMG;
    } else {
        resizeFactor_ = static_cast<double>(height) / static_cast<double>(MAX_IMG);
        width = static_cast<int>(MAX_IMG / static_cast<float>(height) * width);
        height = MAX_IMG;
    }

    cv::Size inputGeometry(width, height);

    inputLayer_->Reshape(1, channels, height, width);       //为图片输入创建blob，分配内存
    sfnNet_->Reshape();                                     //前向传播之前，reshape网络
    float *inputData = inputLayer_->mutable_cpu_data();     //读写访问cpu data

    inputChannels_.clear();
    for (int i = 0; i < inputLayer_->channels(); i++) {
        cv::Mat channel(height, width, CV_32FC1, (void*)inputData); //将RGB每个通道的图像数据和blob数据在内存中位置相对应
        inputChannels_.push_back(channel);
        inputData += width * height;
    }

    cv::Mat imgResized;
    if (image.size() != inputGeometry)
        cv::resize(image, imgResized, inputGeometry);
    else
        imgResized = image;

    cv::Mat imgFloat;
    imgResized.convertTo(imgFloat, CV_32FC3);
    cv::Mat meanMat(inputGeometry, CV_32FC3, cv::Scalar(127.0, 127.0, 127.0));  //中灰色均值
    cv::Mat imgNormalized;
    cv::subtract(imgFloat, meanMat, imgNormalized);                 //图像数据减去均值
    cv::split(imgNormalized, inputChannels_);                       //将多通道的单个图像分离成单个通道的多个图像，并分别传入caffe中

    CHECK (reinterpret_cast<float*> (inputChannels_.at(0).data) == \
             sfnNet_->input_blobs()[0]->cpu_data()) \
            << "Input channels aren't wrapping the input layer of the network.";

    sfnNet_->Forward();
    sfnNetOutput_ = sfnNet_->output_blobs()[0];
}

/* 
* 将sfn网络得到的featureMap多次送进rsa网络进行缩放，然后得到不同size的feature
 */
void RsaFaceDetector::rsaProcess_()
{
    std::shared_ptr<caffe::Blob<float>> transFeatmapOri(new caffe::Blob<float>);
    transFeatmapOri->CopyFrom(*sfnNetOutput_, false, true);         //从上一个sfn网络拷贝出输出
    transFeatMaps_.push_back(transFeatmapOri);

    int diffCnt;
    std::shared_ptr<caffe::Blob<float>> transFeatmap(new caffe::Blob<float>);
    std::shared_ptr<caffe::Blob<float>> inFeatmap(new caffe::Blob<float>);

    for (int i = 1; i < scale_.size(); i++) {                       //对featureMap做5次循环
        int diffCnt = scale_[i - 1] - scale_[i];
        inFeatmap->CopyFrom(*(transFeatMaps_[i - 1]), false, true);
        for (int j = 0; j < diffCnt; j++) {                         //只进行一次前向传递
            rsaInputLayer_->CopyFrom(*inFeatmap, false, true);
            rsaNet_->Reshape();
            rsaNet_->Forward();
            inFeatmap->CopyFrom(*rsaNet_->output_blobs()[0], false, true);
        }
        transFeatMaps_.push_back(std::shared_ptr<caffe::Blob<float>>(new caffe::Blob<float>));
        //将每次rsa网络缩放的featureMap进行保存
        //并且每一次featureMap进行rsa得到的结果都是下一次rsa的输入
        transFeatMaps_[transFeatMaps_.size() - 1]->CopyFrom(*rsaNet_->output_blobs()[0], false, true);  
    }
}

void RsaFaceDetector::lrnProcess_(facesLandmarkPerImg & faceResult)
{
    std::vector<std::vector<cv::Point2f>> ptsAll;
    std::vector<std::vector<double>> rectsAll;
    std::vector<float> validScoreAll;
    boost::shared_ptr<caffe::Blob<float> > blobRpnCls;      //注意这个不是std::shared_ptr
    boost::shared_ptr<caffe::Blob<float> > blobRpnReg;

    for (int i = 0; i < transFeatMaps_.size(); i++) {
        lrnInputLayer_->CopyFrom(*transFeatMaps_[i], false, true);
        lrnNet_->Reshape();
        lrnNet_->Forward();
        blobRpnCls = lrnNet_->blob_by_name("rpn_cls");
        blobRpnReg = lrnNet_->blob_by_name("rpn_reg");

        int fmwidth = blobRpnCls->shape(3);
        int fmheight = blobRpnCls->shape(2);

        std::vector<float> validScore;
        validScore.clear();
        std::vector<std::vector<int>> validIndex;
        validIndex.clear();
        for (int x = 0; x < fmwidth; x++) {
            for (int y = 0; y < fmheight; y++) {
                if (blobRpnCls->data_at(0, 0, y, x) > threshScore_) {
                    std::vector<int> index(2);
                    index[0] = x;
                    index[1] = y;
                    validIndex.push_back(index);
                    validScore.push_back(blobRpnCls->data_at(0, 0, y, x));
                    validScoreAll.push_back(blobRpnCls->data_at(0, 0, y, x));
                }
            }
        }

        /* 5个关键点的获取和处理 */
        std::vector<std::vector<cv::Point2f>> ptsOut(validIndex.size(), std::vector<cv::Point2f>(5, cv::Point2f(0, 0)));
        std::vector<std::vector<double>> rects(validIndex.size(), std::vector<double>(4, 0.0));
        for (int j = 0; j < validIndex.size(); j++) {
            std::vector<float> anchorCenterNow(2);
            anchorCenterNow[0] = validIndex[j][0] * stride_ + anchorCenter_;        //??????
            anchorCenterNow[1] = validIndex[j][1] * stride_ + anchorCenter_;        //??????
            for (int h = 0; h < 5; h++) {
                float anchorPointNowX = anchorCenterNow[0] + *(ANCHOR_PTS + h*2) * anchorBoxLen_[0];
                float anchorPointNowY = anchorCenterNow[1] + *(ANCHOR_PTS + h*2 + 1) * anchorBoxLen_[0];
                float ptsDeltaX = blobRpnReg->data_at(0, 2*h, validIndex[j][1], validIndex[j][0]) * anchorBoxLen_[0];
                float ptsDeltaY = blobRpnReg->data_at(0, 2*h + 1, validIndex[j][1], validIndex[j][0]) * anchorBoxLen_[0];
                ptsOut[j][h].x = ptsDeltaX + anchorPointNowX;
                ptsOut[j][h].y = ptsDeltaY + anchorPointNowY;
            }

            std::vector<cv::Point2f> dstRect(3, cv::Point2f(0, 0));
            cv::Point2f srcFivePoint[5];
            for (int h = 0; h < 5; h++) 
                srcFivePoint[h] = ptsOut[j][h];
            getTripPoints(dstRect, srcFivePoint);
            double scaleDouble = pow(2, scale_[i] - 5);
            double rectWidth = sqrt(pow((dstRect[1].x - dstRect[2].x), 2) + \
                                    pow((dstRect[1].y -dstRect[2].y), 2));
            rects[j][0] = round((dstRect[0].x - rectWidth/2) / scaleDouble * resizeFactor_);
            rects[j][1] = round((dstRect[0].y - rectWidth/2) / scaleDouble * resizeFactor_);
            rects[j][2] = round((dstRect[0].x + rectWidth/2) / scaleDouble * resizeFactor_);
            rects[j][3] = round((dstRect[0].y + rectWidth/2) / scaleDouble * resizeFactor_);

            rectsAll.push_back(rects[j]);

            for (int h = 0; h < 5; h++) {
                ptsOut[j][h].x = round(ptsOut[j][h].x / scaleDouble * resizeFactor_);
                ptsOut[j][h].y = round(ptsOut[j][h].y / scaleDouble * resizeFactor_);
            }
            ptsAll.push_back(ptsOut[j]);
        }
    }

    transFeatMaps_.clear();
    if (!ptsAll.empty()) {
        float *boxes = new float[ptsAll.size() * 5];
        int *keep = new int[ptsAll.size() * 5];
        facesLandmarkPerImg faces;

        for (int i = 0; i < ptsAll.size(); i++) {
            struct faceLandmark faceTmp;
            faceTmp.bbox = rectsAll[i];
            faceTmp.keyPoints = ptsAll[i];
            faceTmp.score = validScoreAll[i];
            faces.push_back(faceTmp);
        }
        std::sort(faces.begin(), faces.end(), comp);    //排序
        for (int i = 0; i < faces.size(); i++) {
            boxes[i*5 + 0] = faces[i].bbox[0];
            boxes[i*5 + 1] = faces[i].bbox[1];
            boxes[i*5 + 2] = faces[i].bbox[2];
            boxes[i*5 + 3] = faces[i].bbox[3];
            boxes[i*5 + 4] = faces[i].score;
        }
        int numOut;
        _nms(keep, &numOut, boxes, faces.size(), 5, NMS_THRESH, gpuId_);
        for (int i = 0; i < numOut; i++)
            faceResult.push_back(faces[*(keep + i)]);
        delete [] boxes;
        delete [] keep;
    }
}

facesLandmarkPerImg RsaFaceDetector::detect_(cv::Mat image)
{
	auto start_sfn = std::chrono::high_resolution_clock::now();
    sfnProcess_(image);
	auto end_sfn = std::chrono::high_resolution_clock::now();
	auto start_rsa = std::chrono::high_resolution_clock::now();
    rsaProcess_();
	auto end_rsa = std::chrono::high_resolution_clock::now();
    facesLandmarkPerImg faces;
	auto start_lrn = std::chrono::high_resolution_clock::now();
    lrnProcess_(faces);
	auto end_lrn = std::chrono::high_resolution_clock::now();
	// std::cout << "sfn took " << std::chrono::duration_cast<milli>(end_sfn - start_sfn).count() << " ms\n";
	// std::cout << "rsa took " << std::chrono::duration_cast<milli>(end_rsa - start_rsa).count() << " ms\n";
	// std::cout << "lrn took " << std::chrono::duration_cast<milli>(end_lrn - start_lrn).count() << " ms\n";
    return faces;
}