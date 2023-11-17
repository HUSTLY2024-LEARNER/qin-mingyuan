#ifndef KALMANFILTER_SIMULATOR_HPP
#define KALMANFILTER_SIMULATOR_HPP

#include "eigen3/Eigen/Dense"
#include <eigen3/Eigen/Core>
#include <vector>
#include <iostream>
#include <cmath>

template<typename T,int x>
class Simulator {
private:
double gaussianRandom(double mean, double _variance) {
    static double V1, V2, S;
    static int phase = 0;
    double X;

    if (phase == 0) {
        do {
            double U1 = (double)random() / RAND_MAX;
            double U2 = (double)random() / RAND_MAX;

            V1 = 2 * U1 - 1;
            V2 = 2 * U2 - 1;
            S = V1 * V1 + V2 * V2;
        } while (S >= 1 || S == 0);

        X = V1 * sqrt(-2 * log(S) / S);
    }
    else {
        X = V2 * sqrt(-2 * log(S) / S);
    }

    phase = 1 - phase;

    return X * _variance + mean;
}
Eigen::Matrix<T, x, 1> x0;
double variance;//方差
double amplitude; // 振幅
double frequency; // 频率
double velocity; //速度
public:

    Eigen::Matrix<T, x, 1> getMeasurement(double t) {
        Eigen::Matrix<T, x, 1> theoretical;
        Eigen::Matrix<T, x, 1> measurement;
        Eigen::Matrix<T, x, 1> noise;

        theoretical(0) = x0(0) + t * velocity; // x方向匀速直线运动
        theoretical(1) = x0(1) + amplitude * sin(t * frequency); // y方向正弦函数运动
        theoretical(2) = x0(2);
        theoretical(3) = x0(3);
        noise = Eigen::Matrix<T, x, 1>::Zero();
        for (int i = 0; i < x; i++) {
            noise(i, 0) = gaussianRandom(0, this->variance);
        }
        measurement = theoretical + noise;
        return measurement;
    }

    explicit Simulator(Eigen::Matrix<T, x, 1> x0, double variance, double amplitude, double frequency,double velocity) {
        this->x0 = x0;
        this->variance = variance;
        this->amplitude = amplitude;
        this->frequency = frequency;
        this->velocity = velocity;
    }
};
#endif