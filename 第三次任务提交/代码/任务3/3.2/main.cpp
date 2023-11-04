#include <iostream>
#include "opencv2/viz.hpp"
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

int main(int argc, char** argv) 
{
    viz::Viz3d myWindow("Coordinate Frame");
    Vec3d cam1_pose(0, 0, 0), cam1_focalPoint(0, 0, 1), cam1_y_dir(0, 1, 0); // 设置相机的朝向（光轴朝向）
    Affine3d cam_3_pose = viz::makeCameraPose(cam1_pose, cam1_focalPoint, cam1_y_dir); // 设置相机位置与朝向

    myWindow.showWidget("World_coordinate", viz::WCoordinateSystem(), cam_3_pose); // 创建3号相机位于世界坐标系的原点
    // 创建R\T
    Matx33d PoseR_0; // 旋转矩阵
    Vec3d PoseT_0; // 平移向量
    PoseR_0 = Matx33d(0.99683809, -0.074948169, 0.026392058,
        -0.066030696, -0.596582, 0.79983115,
        -0.044200856, -0.79904491, -0.59964454);
    PoseT_0 = Vec3d(374.2372011478013,
    1033.054892225266,
    6285.041483288341)*0.001;

    Affine3d Transpose03(PoseR_0, PoseT_0); // 03相机变换矩阵
    myWindow.showWidget("Cam0", viz::WCoordinateSystem(), Transpose03);
    myWindow.spin();
    return 0;
}
