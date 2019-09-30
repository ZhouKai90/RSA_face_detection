
#include <sstream>
#include <sys/stat.h>
#include <dirent.h>

#include <iostream>
#include <fstream>
#include "image_file_opt.hpp"
/*
*递归获取文件夹下的所有图片的文件名
*/
int getAllImageName_xfs(std::string dirPath, std::vector<std::string> &images)
{
	DIR *dirHandle =  opendir(dirPath.c_str());
	if (dirHandle == NULL) {
		printf("Open dir[%s] failed.\r\n", dirPath.c_str());
		return -1;
	}

	struct dirent *filePtr = NULL;
	while ((filePtr = readdir(dirHandle)) != NULL) {
		if (strcmp(filePtr->d_name, ".") == 0 || strcmp(filePtr->d_name, "..") == 0) //skip current dir and parent dir
			continue;

		std::string fileName(dirPath + "/" + filePtr->d_name);

		struct stat sb;
		if (stat(fileName.c_str(), &sb) == -1)
			continue;

		if (S_ISREG(sb.st_mode)) {	
			// std::cout << fileName << std::endl;
			if (fileName.rfind(".jpg") == std::string::npos \
				&& fileName.rfind(".tiff") == std::string::npos \
				&& fileName.rfind(".png") == std::string::npos)
				continue;
								//for regular file
			images.push_back(fileName);
		} else if (S_ISDIR(sb.st_mode))	{						//for directory
			std::vector<std::string> subImageList;
			getAllImageName_xfs(fileName, subImageList);

			for (auto subImage : subImageList)
				images.push_back(subImage);
		} else if(S_ISLNK(sb.st_mode)) {						//for A symbolic link
		} else if(S_ISCHR(sb.st_mode)) {						//for a character device
		} else if(S_ISSOCK(sb.st_mode)) {						//for a local-domain socket
		} else if(S_ISBLK(sb.st_mode)) {						//for block device
		}
	}
	closedir(dirHandle);
	return 0;
}

/*
*从图片库中随机获取一定数量的图片，并进行解码
*/
std::vector<std::string> readImageToCvMap(const std::string dirPath, std::vector<cv::Mat> &imageMatList, bool Isrand, unsigned int cnt)
{
    std::vector<std::string> imageList;
    getAllImageName_xfs(dirPath, imageList);
	std::vector<std::string> imageName;
	unsigned int toDecode = 0;

	if (!imageList.size())
		return imageName;

	if (cnt == 0) 
		toDecode = imageList.size();
	else
		toDecode = cnt;

	for (int i = 0; i < toDecode; i++) {
		unsigned randNum = i;
		if (Isrand)
			randNum = rand()%(imageList.size());
		// std::cout << "rand Num : " << randNum << std::endl;

		cv::Mat bitMap = cv::imread(imageList[randNum].c_str());
		if (bitMap.empty()) {
			// std::cout << "Decode " << imageList[randNum] << "failed." << std::endl;
			--i;
			continue;
		}
        imageMatList.push_back(bitMap);
		imageName.push_back(imageList[randNum]);
	}
	return imageName;
}

int video_capture(const std::string &video_path)
{
	cv::VideoCapture cap(video_path);
	if (!cap.isOpened()) {
		std::cout << "Open video file failed." << std::endl;
		return -1;
	}
	cv::Mat frame;
	do {
		cap >> frame;
		cv::imshow("frame", frame);
		if (cv::waitKey(30) >= 0)
			break;
	} while (!frame.empty());
	return 0;
}
