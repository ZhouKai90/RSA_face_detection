#include "rsa_util.hpp"

static const double dst_landmark[10] = {30.2946, 65.5318, 48.0252, 33.5493, 62.7299,51.6963, 51.5014, 71.7366, 92.3655, 92.2041 };

#计算旋转的角度
inline float GetRotateAngle(cv::Point2f &p1, cv::Point2f &p2, cv::Point2f &p3, cv::Point2f &p4)
{
	float vx1, vy1, vx2, vy2;
	float dot, cross;
	float n1, n2;
	float theta;

	vx1 = p2.x - p1.x;
	vy1 = p2.y - p1.y;
	vx2 = p4.x - p3.x;
	vy2 = p4.y - p3.y;
	dot = vx1*vx2 + vy1*vy2;
	cross = vx1*vy2 - vx2*vy1;
	n1 = sqrt(vx1*vx1 + vy1*vy1);
	n2 = sqrt(vx2*vx2 + vy2*vy2);

	theta = acos(dot / (n1*n2));
	if (cross<0)
	{
		theta = 3.14 * 2 - theta;
	}
	return theta;
}


