#include <iostream>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>
#include "Simulator.hpp"

using namespace std;

// ��ͨ�˲�����
class LowPassFilter 
{
public:
    LowPassFilter(double sample_rate, double cutoff_frequency) 
    {
        double dt = 1.0 / sample_rate;//���ý�ֹƵ���Լ�������
        double RC = 1.0 / (cutoff_frequency * 2.0 * M_PI);
        alpha_ = dt / (dt + RC);
        prev_output_ = 0.0;
    }

    // �����˲������
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
    // �����ʺͽ�ֹƵ��
    double sample_rate = 100.0;
    double cutoff_frequency = 10.0;

    // ������ͨ�˲���
    LowPassFilter filter(sample_rate, cutoff_frequency);
    //����������
    Simulator<double, 2>* simulator;
    simulator = new Simulator<double, 2>(Eigen::Vector2d(0, 0), 5, 0, 2);// ����Ϊ��ʼ���뷽���ʼ�ٶ��Լ����ٶ�

    // ���������
    Eigen::Vector2d measurement;
    Eigen::Vector2d estimate;
    cv::Mat img(500, 500, CV_8UC3, cv::Scalar(0, 0, 0));
    int t = 0;
    while (t++<1000) 
    {
        //�����˶�״̬

        measurement = simulator->getMeasurement(t);

        //Ԥ��
        estimate[0] = filter.update(measurement[0]);
        estimate[1] = filter.update(measurement[1]);

        //�����˶�״̬
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