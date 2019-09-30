#ifndef _CONFIG_HPP_
#define _CONFIG_HPP_
#include <string>

#define NMS_THRESH 0.4
#define THRESH_CLS 3.0
#define THRESH_SCORE 8.0
#define ANCHOR_CENTER 7.5
#define STRIDE 16.0
#define MAX_IMG 2048
#define MIN_IMG 64


// const std::string SFN_NET_DEF = ROOT_PATH + "/model/model_def/res_pool2/test.prototxt";
// const std::string SFN_NET_WEIGHT = ROOT_PATH + "model/model_weight/ResNet_3b_s16/tot_wometa_1epoch";
// const std::string RSA_NET_DEF = ROOT_PATH + "model/model_def/ResNet_3b_s16_fm2fm_pool2_deep/test.prototxt";
// const std::string RSA_NET_WEIGHT = ROOT_PATH + "model/model_weight/ResNet_3b_s16_fm2fm_pool2_deep/65w";
// const std::string LRN_NET_DEF = ROOT_PATH + "model/model_def/ResNet_3b_s16_f2r/test.prototxt";
// const std::string LRN_NET_WEIGHT = ROOT_PATH + "model/model_weight/ResNet_3b_s16/tot_wometa_1epoch";


const float ANCHOR_BOX[] = {-44.754833995939045, -44.754833995939045, 44.754833995939045, 44.754833995939045};
const float ANCHOR_PTS[] = {-0.1719448, -0.2204161, 0.1719447, -0.2261145, -0.0017059, -0.0047045, -0.1408936, 0.2034478, 0.1408936, 0.1977626};




#endif