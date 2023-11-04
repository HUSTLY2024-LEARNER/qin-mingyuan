#include <opencv2/opencv.hpp>  
#include<iostream>
#include <opencv2/highgui/highgui_c.h>
using namespace std;
using namespace cv;

// 颜色识别(除红色外的其他单色)
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
// 颜色识别（蓝色）
Mat ColorFindContours1(Mat srcImage)
{
    Mat des1 = ColorFindContours(srcImage,
        170 / 2, 240 / 2,                // 色调最小值~最大值
        (int)(255 * 0.4), (255 * 0.95),   // 饱和度最小值~最大值
        (int)(255 * 0.7), (255*0.95));  // 亮度最小值~最大值

    return des1;
}
void pnp_rt(Point2f ps[4])
{
    Mat img = imread("../1_raw.png");

    vector<Point3f>objPts;//3D点坐标。以灯条的尺寸，单位mm
    objPts.push_back(Point3f( -25.0f, 142.5f, 0));
    objPts.push_back(Point3f(25.0f, 142.5f, 0));
    objPts.push_back(Point3f(25.0f,-142.5f,0));
    objPts.push_back(Point3f(-25.0f,-142.5f, 0));
    vector<Point2f>imgPts ;	//2D点坐标，即图像上点的坐标
    imgPts.push_back(Point2f(ps[0].x, ps[0].y));
    imgPts.push_back(Point2f(ps[1].x, ps[1].y));
    imgPts.push_back(Point2f(ps[2].x, ps[2].y));
    imgPts.push_back(Point2f(ps[3].x, ps[3].y));
    for (int i = 0; i < 4; i++)
    {
        circle(img, imgPts[i], 3, Scalar(0, 0, 255), -1);
    }
    double fx = 1900, fy = 1900, cx = 960, cy = 540;
    Mat camera_matrix = (Mat_<double>(3, 3) << fx, 0, cx,//相机内参
        0, fy, cy,
        0, 0, 1);
    // 相机畸变系数
    Mat dist_coeffs = (Mat_<double>(5, 1) << 0, 0,
        0, 0, 0);
    Mat rvec, tvec;
    solvePnP(objPts, imgPts, camera_matrix, dist_coeffs, rvec, tvec);
    cout << "Rotation Vector " << endl << rvec << endl << endl;
    cout << "Translation Vector" << endl << tvec << endl << endl;

    Mat Rvec;
    Mat_<float> Tvec;
    rvec.convertTo(Rvec, CV_32F);  // 旋转向量转换格式
    tvec.convertTo(Tvec, CV_32F); // 平移向量转换格式 

    Mat_<float> rotMat(3, 3);
    Rodrigues(Rvec, rotMat);
    // 旋转向量转成旋转矩阵
    cout << "rotMat" << endl << rotMat << endl << endl;

    Mat P_oc;
    P_oc = -rotMat.inv() * Tvec;
    // 求解相机的世界坐标
    cout << "P_oc" << endl << P_oc << endl;


    namedWindow("Output", CV_WINDOW_NORMAL);
    imshow("Output", img);
    waitKey(0);


}

int main()
{
    /*提取灯条的轮廓*/
        Mat img = imread("../1_raw.png");
        Mat des = ColorFindContours1(img);
        Mat binary;
        GaussianBlur(des, des, Size(9, 9), 2, 2);
        threshold(des, binary, 170, 255, THRESH_BINARY | THRESH_OTSU);
        vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;
        findContours(binary, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point());
        vector<RotatedRect> minAreaRects(contours.size());
        Point2f ps[4];
        int t = 0;
        for (int i = 0; i < contours.size(); ++i)
        {  //遍历所有轮廓
                double areal = contourArea(contours[i]);
                if (areal > 470 && areal < 700)
                {
                    minAreaRects[i] = minAreaRect(contours[i]);  //获取轮廓的最小外接矩形
                    minAreaRects[i].points(ps);  //将最小外接矩形的四个端点复制给ps数组
                    if (t == 2)
                    {
                        for (int j = 0; j < 4; j++)
                        {  //绘制最小外接轮廓的四条边
                            line(img, Point(ps[j]), Point(ps[(j + 1) % 4]), Scalar(0, 255, 0), 2);
                        }

                    }
                    t++;
                }
         }
    pnp_rt(ps);//进入pnp解算函数
    return 0;
}
