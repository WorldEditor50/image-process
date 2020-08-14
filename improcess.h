#ifndef IMPROCESS_H
#define IMPROCESS_H
/* c++ header */
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
/* opencv header */
#include<opencv2/opencv.hpp>
#include<opencv2/imgproc/imgproc.hpp>
using namespace std;
using namespace cv;
void Improcess_Test(const char* srcName, const char* dstName);
/* 二值化 */
void Improcess_Threshold(Mat& srcImg, Mat& dstImg);
void Improcess_AdaptiveThreshold(Mat& srcImg, Mat& dstImg);
/* 滤波 */
/* 方框滤波 */
void Improcess_BoxFilter(Mat& srcImg, Mat& dstImg);
/* 均值滤波 */
void Improcess_Blur(Mat& srcImg, Mat& dstImg);
/* 高斯滤波 */
void Improcess_GaussianBlur(Mat& srcImg, Mat& dstImg);
/* 形态学运算 */
/* 腐蚀 */
void Improcess_Erode(Mat& srcImg, Mat& dstImg);
/* 膨胀 */
void Improcess_Dilate(Mat& srcImg, Mat& dstImg);
/* 开运算 */
void Improcess_MorphOpen(Mat& srcImg, Mat& dstImg);
/* 闭运算 */
void Improcess_MorphClose(Mat& srcImg, Mat& dstImg);
/* 顶帽 */
void Improcess_MorphTopHat(Mat& srcImg, Mat& dstImg);
/* 黑帽 */
void Improcess_MorphBlackHat(Mat& srcImg, Mat& dstImg);
/* 形态学梯度 */
void Improcess_MorphGrident(Mat& srcImg, Mat& dstImg);
/* 边缘检测 */
/* canny算子 */
void Improcess_Canny(Mat& srcImg, Mat& dstImg);
/* sobel算子 */
void Improcess_Sobel(Mat& srcImg, Mat& dstImg);
/* laplace算子 */
void Improcess_Laplace(Mat& srcImg, Mat& dstImg);
/* 霍夫变换 */
/* 霍夫线变换 */
void Improcess_HoughLine(Mat& srcImg, Mat& dstImg);
void Improcess_HoughLineP(Mat& srcImg, Mat& dstImg);
/* 霍夫圆变换 */
void Improcess_HoughCircle(Mat& srcImg, Mat& dstImg);
/* 重映射与仿射 */
void Improcess_Remap(Mat& srcImg, Mat& dstImg);
void Improcess_AffineByRotate(Mat& srcImg, Mat& dstImg);
void Improcess_AffineByPoint(Mat& srcImg, Mat& dstImg);
/* 模板匹配 */
void Improcess_MatchTemplate(Mat& srcImg, Mat& dstImg);
/* 图像修复 */

/* 轮廓查找与多边形包围 */
void Improcess_FindMaxContours(Mat& srcImg, Mat& dstImg);
void Improcess_FindContours(Mat& srcImg, Mat& dstImg);
void Improcess_FindContoursByLR(Mat& srcImg, Mat& dstImg);
void Improcess_FindContoursByMeanZero(Mat& srcImg, Mat& dstImg);
/* 采样 */
void Improcess_Sampling(Mat& srcImg, Mat& dstImg);
/* 绘制点，直线，几何图形 */

/* 角点检测 */

/* 图像矫正 */

/* Mat和IplImage的像素访问 */

/* 读写xml与yml文件 */
void Improcess_Xml();
#endif // IMPROCESS_H
