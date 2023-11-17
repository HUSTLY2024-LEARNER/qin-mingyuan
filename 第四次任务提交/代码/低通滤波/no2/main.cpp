#include <iostream>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>
#include "Simulator.hpp"

using namespace std;

// 低通滤波器类
class LowPassFilter 
{
public:
    LowPassFilter(double sample_rate, double cutoff_frequency) 
    {
        double dt = 1.0 / sample_rate;//设置截止频率以及采样率
        double RC = 1.0 / (cutoff_frequency * 2.0 * M_PI);
        alpha_ = dt / (dt + RC);
        prev_output_ = 0.0;
    }

    // 更新滤波器输出
    double update(double input) 
    {
        double output = alpha_ * input + (1.0 - alpha_) * prev_output_;
        prev_output_ = output;
        return output;
    }

private:
    double alpha_;
    double prev_output_;
};

int main() 
{
    // 采样率和截止频率
    double sample_rate = 100.0;
    double cutoff_frequency = 10.0;

    // 创建低通滤波器
    LowPassFilter filter(sample_rate, cutoff_frequency);
    //创建仿真器
    Simulator<double, 2>* simulator;
    simulator = new Simulator<double, 2>(Eigen::Vector2d(0, 0), 5, 0, 2);// 输入为起始点与方差，起始速度以及加速度

    // 生成随机点
    Eigen::Vector2d measurement;
    Eigen::Vector2d estimate;
    cv::Mat img(500, 500, CV_8UC3, cv::Scalar(0, 0, 0));
    int t = 0;
    while (t++<1000) 
    {
        //生成运动状态

        measurement = simulator->getMeasurement(t);

        //预测
        estimate[0] = filter.update(measurement[0]);
        estimate[1] = filter.update(measurement[1]);

        //绘制运动状态
        cv::circle(img, cv::Point((int)(measurement[0]), int(500-measurement[1] )), 2, cv::Scalar(0, 0, 255), -1);
        cv::circle(img, cv::Point((int)(estimate[0]), (int)(500-estimate[1])), 2, cv::Scalar(0, 255, 0), -1);

        cv::imshow("img", img);
        if (t != 1000)
            cv::waitKey(10);
        else
            cv::waitKey();
    }

    return 0;
}