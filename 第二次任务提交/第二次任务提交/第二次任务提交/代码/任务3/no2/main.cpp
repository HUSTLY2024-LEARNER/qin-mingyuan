#include<opencv2/opencv.hpp>  
#include <iostream>
using namespace std;
using namespace cv;

// 颜色识别识别车牌颜色
Mat ColorFindContours(Mat srcImage,
	int iLowH, int iHighH,
	int iLowS, int iHighS,
	int iLowV, int iHighV)
{
	Mat bufImg;
	Mat imgHSV;
	//转为HSV
	cvtColor(srcImage, imgHSV, COLOR_BGR2HSV);
	inRange(imgHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), bufImg);
	return bufImg;
}

Mat ColorFindContours1(Mat srcImage)
{
	Mat des1 = ColorFindContours(srcImage,
		200/2, 248/2 ,                // 色调最小值~最大值
		int(255*0.85), int(255),   // 饱和度最小值~最大值
		(255*0.68), (255*0.90));  // 亮度最小值~最大值

	return des1;
}
Mat car(Mat img)
{
	Mat des2 = ColorFindContours1(img);
	Mat binary;
	Mat dst;
	GaussianBlur(des2, des2, Size(9, 9), 2, 2);//高斯滤波
	threshold(des2, binary, 170, 255, THRESH_BINARY | THRESH_OTSU);//二值化

	Mat kernel = getStructuringElement(MORPH_RECT, Size(22, 22), Point(-1, -1));//膨胀操作参数大小设为22*22
	dilate(binary, dst, kernel);

	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(dst, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point());
	
	Point2f ps[4];
	for (int i = 0; i < contours.size(); ++i)
	{  //遍历所有轮廓
		double areal = contourArea(contours[i]);
		if (areal > 100)
		{
			Rect rect = boundingRect(contours[i]);  //获取轮廓的外接矩形
			rectangle(img, rect, Scalar(0,255,0), 5);  //绘制轮廓矩形
			//保存轮廓
			Mat selectedRegion = img(rect);
			imwrite("jieguo.jpg", selectedRegion);
		}
	}
	return img;
}
int main()
{

    Mat img = imread("/home/qmy/task2/no2/3.png");
	Mat result = car(img);
	namedWindow("result", 0);
	imshow("result", result);//显示结果
	waitKey(0);


    return 0;
}
