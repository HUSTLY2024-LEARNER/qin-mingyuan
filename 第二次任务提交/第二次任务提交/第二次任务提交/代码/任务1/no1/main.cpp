#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;
//RGB颜色结构体
struct RGBColor
{
    int R;
    int G;
    int B;
};

//HSV颜色结构体
struct HSVColor
{
    double H;
    double S;
    double V;
};

// 将RGB颜色转换为HSV颜色
HSVColor RGBtoHSV(const RGBColor& rgb) {
    HSVColor hsv;

    double R = static_cast<double>(rgb.R) / 255.0;
    double G = static_cast<double>(rgb.G) / 255.0;
    double B = static_cast<double>(rgb.B) / 255.0;

    double Cmax = std::max(R, std::max(G, B));
    double Cmin = std::min(R, std::min(G, B));
    double delta = Cmax - Cmin;

    // 计算H（色调）
    if (delta == 0) {
        hsv.H = 0.0;  // 无色
    }
    else if (Cmax == R)
    {
        hsv.H = 60.0 * fmod((G - B) / delta, 6.0);
    }
    else if (Cmax == G)
    {
        hsv.H = 60.0 * ((B - R) / delta + 2.0);
    }
    else if (Cmax == B)
    {
        hsv.H = 60.0 * ((R - G) / delta + 4.0);
    }

    // 计算S（饱和度）
    if (Cmax == 0)
    {
        hsv.S = 0.0;
    }
    else
    {
        hsv.S = delta / Cmax;
    }

    // 计算V（亮度）
    hsv.V = Cmax;

    return hsv;
}

int main()
{
    Mat image = imread("/home/qmy/task2/no1/2.jpg");

    if (image.empty())
    {
        std::cout << "无法加载图像" << std::endl;
        return -1;
    }

    // 将RGB图像转换为HSV图像
    Mat hsv_image(image.size(), CV_8UC3);
    for (int y = 0; y < image.rows; y++)
    {
        for (int x = 0; x < image.cols; x++)
        {
            Vec3b rgb = image.at<Vec3b>(y, x);
            RGBColor rgb_color = { rgb[2], rgb[1], rgb[0] };
            HSVColor hsv_color = RGBtoHSV(rgb_color);
            hsv_image.at<Vec3b>(y, x) = Vec3b(hsv_color.H / 2, hsv_color.S * 255, hsv_color.V * 255);
        }
    }

    // 显示HSV图像
    imshow("HSV结果", hsv_image);
    waitKey(0);
    return 0;
}



