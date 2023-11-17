#include <iostream>
#include <opencv2/opencv.hpp>
#include "KalmanFilter.hpp"
#include "Simulator.hpp"

using namespace std;

int main() 
{
    srand(114514);
    // 以一个静止滤波为例展示使用方法
    // 滤波器初始化
    KalmanFilter<double, 4, 4> *kf;
    kf = new KalmanFilter<double, 4, 4>();
    // 仿真器初始化
    Simulator<double, 4> *simulator;
    simulator = new Simulator<double, 4>(Eigen::Vector4d(0, 0,0,0), 1,10,50,3); // 输入为起始点与方差，频率,幅值,速度

    // 2. 设置状态转移矩阵
    kf->transition_matrix << 1, 0, 1, 0,
        0, 1, 0, 1,
        0, 0, 1, 0,
        0, 0, 0, 1;

    // 3. 设置测量矩阵
    kf->measurement_matrix << 1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1;

    // 4. 设置过程噪声协方差矩阵
    kf->process_noise_cov << 0.01, 0, 0, 0,
        0, 0.01, 0, 0,
        0, 0, 0.01, 0,
        0, 0, 0, 0.01;

    // 5. 设置测量噪声协方差矩阵
    kf->measurement_noise_cov << 1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1;

    // 6. 设置控制向量
    kf->control_vector << 0, 0, 0, 0;

    // 生成随机点
    Eigen::Vector4d measurement;
    cv::Mat img(500, 500, CV_8UC3, cv::Scalar(0, 0, 0));
    int t = 0;
    while (t++<100) 
    {
        measurement = simulator->getMeasurement(t);
        // 7. 预测
        kf->predict(measurement);
        // 8. 更新
        kf->update();
        // 9. 获取后验估计
        Eigen::Vector4d estimate = kf->posteriori_state_estimate;
        //测量和预测都是思维向量，其中前两个表示其x，y方向的位置
        cv::circle(img, cv::Point((int)(measurement[0] * 10), int(measurement[1] * 10+250 )), 2, cv::Scalar(0, 0, 255), -1);
        cv::circle(img, cv::Point((int)(estimate[0] * 10), (int)(estimate[1] * 10+250 )), 2, cv::Scalar(0, 255, 0), -1);
        cv::imshow("img", img);
        if (t != 99)
            cv::waitKey(100);
        else
            cv::waitKey();

    }

    return 0;
}
