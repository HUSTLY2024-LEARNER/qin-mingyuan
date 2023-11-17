
#ifndef KALMANFILTER_SIMULATOR_HPP
#define KALMANFILTER_SIMULATOR_HPP

#include "eigen3/Eigen/Dense"
#include <eigen3/Eigen/Core>
#include <vector>
#include <iostream>

template<typename T, int x>
class Simulator {
private:
// 生成高斯高斯分布噪声，均值为mean，方差为variance
    double gaussianRandom(double mean, double _variance) {
        static double V1, V2, S;
        static int phase = 0;
        double X;

        if (phase == 0) {
            do {
                double U1 = (double) random() / RAND_MAX;
                double U2 = (double) random() / RAND_MAX;

                V1 = 2 * U1 - 1;
                V2 = 2 * U2 - 1;
                S = V1 * V1 + V2 * V2;
            } while (S >= 1 || S == 0);

            X = V1 * sqrt(-2 * log(S) / S);
        } else {
            X = V2 * sqrt(-2 * log(S) / S);
        }

        phase = 1 - phase;

        return X * _variance + mean;
    }
    Eigen::Matrix<T, x, 1> x0;
    double variance;
    double v;//速度
    double a;//加速度
public:

    Eigen::Matrix<T, x, 1> getMeasurement(double t ) {
        Eigen::Matrix<T, x, 1> theoretical;
        Eigen::Matrix<T, x, 1> measurement;
        Eigen::Matrix<T, x, 1> noise;

        //匀加速直线运动模型
        theoretical[0] = v * t+0.5*a*t*t;//位置
        theoretical[1] = a * t;//速度

        noise = Eigen::Matrix<T, x, 1>::Zero();
        for (int i = 0; i < x; i++) {
            noise(i, 0) = gaussianRandom(0, this->variance);
        }
        measurement = theoretical + noise;
        return measurement;
    }

    explicit Simulator(Eigen::Matrix<T, x, 1> x0, double variance,double v,double a) {
        this->x0 = x0;
        this->variance = variance;
        this->v = v;
        this->a = a;
    }
};

#endif //KALMANFILTER_SIMULATOR_HPP
