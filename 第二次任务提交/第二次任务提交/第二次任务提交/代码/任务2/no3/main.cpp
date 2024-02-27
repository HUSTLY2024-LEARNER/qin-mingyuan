#include <opencv2/opencv.hpp>
#include <vector>
using namespace std;
using namespace cv;

void zoomup(int k, void* usrdata)
{
    Mat img = *(Mat*)(usrdata);
    Mat up(img.rows * (k+1), img.cols * (k+1), img.type());//k+1是因为滑动条到0会报错退出
    // 使用at方法执行向上采样
    for (int y = 0; y < up.rows; y++)
    {
        for (int x = 0; x < up.cols; x++)
        {
            up.at<Vec3b>(y, x) = img.at<Vec3b>(y / (k + 1), x / (k + 1));//将三通道img每个像素值除以2复制到up
        }
    }
    imshow("放大图", up);
}
void zoomdown(int k, void* usrdata)
{
    Mat img = *(Mat*)(usrdata);
    Mat down(img.rows / (k + 1), img.cols / (k + 1), img.type());
    // 使用at方法执行向上采样
    for (int y = 0; y < down.rows; y++)
    {
        for (int x = 0; x < down.cols; x++)
        {
           down.at<Vec3b>(y, x) = img.at<Vec3b>(y * (k + 1), x * (k + 1));//将三通道img每个像素值除以2复制到up
        }
    }
    // 显示向上采样后的图像
    imshow("缩小图", down);
}
int main()
{
    int flag = 0;
    cout << "请选择\n1:放大\n2:缩小\n" << endl;
    cin >> flag;
    int  k = 0;
    int max = 5;
    Mat img = imread("../wuda.jpg");
    namedWindow("picture");
    imshow("picture", img);
    if (flag == 1)
        createTrackbar("zoomup", "picture", &k, max, zoomup, &img);
    else if(flag==2)
        createTrackbar("zoomdown", "picture", &k, max, zoomdown, &img);
    waitKey(0);
    destroyAllWindows();
    return 0;
}

