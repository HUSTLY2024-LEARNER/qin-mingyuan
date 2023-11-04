#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <iostream>
#include <opencv2/viz.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
Eigen::Matrix3d rotation_matrix3;  // 初始旋转矩阵

Eigen::Matrix3d er()
{
    cout << endl << "********** EulerAngle Turn To Rotation Matrix **********" << endl;
    //3.0 初始化欧拉角(Z-Y-X，即RPY, 先绕x轴roll,再绕y轴pitch,最后绕z轴yaw)
    double roll, pitch, yaw;
    cout << "enter roll,please" << endl;
    cin >> roll;
    cout << "enter pitch,please" << endl;
    cin >> pitch;
    cout << "enter yaw,please" << endl;
    cin >> yaw;
    Eigen::Vector3d ea(roll, pitch, yaw);

    //3.1 欧拉角转换为旋转矩阵
    Eigen::Matrix3d rotation_matrix3;
    rotation_matrix3 = Eigen::AngleAxisd(ea[0], Eigen::Vector3d::UnitZ()) *
        Eigen::AngleAxisd(ea[1], Eigen::Vector3d::UnitY()) *
        Eigen::AngleAxisd(ea[2], Eigen::Vector3d::UnitX());
    cout << "rotation matrix3 =\n" << rotation_matrix3 << endl;
    return rotation_matrix3;
}

int main(int argc, char** argv) 
{
    Eigen::Matrix3d rotation_matrix3 = er();
    viz::Viz3d myWindow("Original");
    Vec3d cam1_pose(0, 0, 0), cam1_focalPoint(0, 0, 1), cam1_y_dir(0, -1, 0); // 设置相机的朝向（光轴朝向）
    Affine3d cam_pose = viz::makeCameraPose(cam1_pose, cam1_focalPoint, cam1_y_dir); // 设置相机位置与朝向

    myWindow.showWidget("World_coordinate", viz::WCoordinateSystem(), cam_pose); // 创建相机位于世界坐标系的原点
    // 创建R\T
    myWindow.spin();
    Matx33d PoseR_0; // 旋转矩阵
    for (int i = 0; i < 3; i++) 
    {
        for (int j = 0; j < 3; j++) 
        {
            PoseR_0(i, j) = rotation_matrix3(i, j);
        }
    }
    Vec3d PoseT_0;
    PoseT_0 = Vec3d(0, 0, 0) * 10;//只旋转，偏移量为零
    Affine3d Transpose(PoseR_0, PoseT_0); // 相机变换矩阵
    viz::Viz3d myWindow1("Changed");
    myWindow1.showWidget("Cam0", viz::WCoordinateSystem(), Transpose);
    myWindow1.spin();
    return 0;
}
