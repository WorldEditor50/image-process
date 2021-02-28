#ifndef IMPROCESS_H
#define IMPROCESS_H
#include <QObject>
#include <QImage>
#include <QMap>
/* c++ header */
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <functional>
/* opencv header */
#include<opencv2/opencv.hpp>
#include<opencv2/imgproc/imgproc.hpp>
using namespace cv;
class Improcess : public QObject
{
    Q_OBJECT
    using ProcessType = std::function<void(const QString &srcPath)>;
    QMap<QString, ProcessType> methods;
public:
    explicit Improcess(QObject *parent = nullptr);
    QImage convertToQImage(const Mat& src);
signals:
    void processFinished(const QImage &img);
public slots:
    void process(const QString &method, const QString &srcPath);
public:
    /* 二值化 */
    void threshold(const QString &srcPath);
    void adaptiveThreshold(const QString &srcPath);
    /* 滤波 */
    /* 方框滤波 */
    void boxFilter(const QString &srcPath);
    /* 均值滤波 */
    void blur(const QString &srcPath);
    /* 高斯滤波 */
    void gaussianBlur(const QString &srcPath);
    /* 形态学运算 */
    /* 腐蚀 */
    void erode(const QString &srcPath);
    /* 膨胀 */
    void dilate(const QString &srcPath);
    /* 开运算 */
    void morphOpen(const QString &srcPath);
    /* 闭运算 */
    void morphClose(const QString &srcPath);
    /* 顶帽 */
    void morphTopHat(const QString &srcPath);
    /* 黑帽 */
    void morphBlackHat(const QString &srcPath);
    /* 形态学梯度 */
    void morphGrident(const QString &srcPath);
    /* 边缘检测 */
    /* canny算子 */
    void canny(const QString &srcPath);
    /* sobel算子 */
    void sobel(const QString &srcPath);
    /* laplace算子 */
    void laplace(const QString &srcPath);
    /* 霍夫变换 */
    /* 霍夫线变换 */
    void houghLine(const QString &srcPath);
    void houghLineP(const QString &srcPath);
    /* 霍夫圆变换 */
    void houghCircle(const QString &srcPath);
    /* 重映射与仿射 */
    void remap(const QString &srcPath);
    void affineByRotate(const QString &srcPath);
    void affineByPoint(const QString &srcPath);
    /* 模板匹配 */
    void matchTemplate(const QString &srcPath);
    /* 图像修复 */
    /* 轮廓查找与多边形包围 */
    Mat getMaxContours(Mat& src, Mat& edge);
    void findMaxContours(const QString &srcPath);
    void findContoursByLR(const QString &srcPath);
    void findContoursByMeanZero(const QString &srcPath);
    /* 采样 */
    /* 绘制点，直线，几何图形 */
    /* 角点检测 */

    /* 图像矫正 */

    /* Mat和IplImage的像素访问 */

    /* 读写xml与yml文件 */

};
#endif // IMPROCESS_H
