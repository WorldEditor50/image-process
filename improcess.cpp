#include "improcess.h"
#include <QDebug>
Improcess::Improcess(QObject *parent):
    QObject(parent)
{
    methods.insert("threshold", std::bind(&Improcess::threshold, this, std::placeholders::_1));
    methods.insert("adaptiveThreshold", std::bind(&Improcess::adaptiveThreshold, this, std::placeholders::_1));
    methods.insert("boxFilter", std::bind(&Improcess::boxFilter, this, std::placeholders::_1));
    methods.insert("blur", std::bind(&Improcess::blur, this, std::placeholders::_1));
    methods.insert("gaussianBlur", std::bind(&Improcess::gaussianBlur, this, std::placeholders::_1));
    methods.insert("erode", std::bind(&Improcess::erode, this, std::placeholders::_1));
    methods.insert("dilate", std::bind(&Improcess::dilate, this, std::placeholders::_1));
    methods.insert("morphOpen", std::bind(&Improcess::morphOpen, this, std::placeholders::_1));
    methods.insert("morphClose", std::bind(&Improcess::morphClose, this, std::placeholders::_1));
    methods.insert("morphTopHat", std::bind(&Improcess::morphTopHat, this, std::placeholders::_1));
    methods.insert("morphBlackHat", std::bind(&Improcess::morphBlackHat, this, std::placeholders::_1));
    methods.insert("morphGrident", std::bind(&Improcess::morphGrident, this, std::placeholders::_1));
    methods.insert("canny", std::bind(&Improcess::canny, this, std::placeholders::_1));
    methods.insert("sobel", std::bind(&Improcess::sobel, this, std::placeholders::_1));
    methods.insert("laplace", std::bind(&Improcess::laplace, this, std::placeholders::_1));
    methods.insert("houghLine", std::bind(&Improcess::houghLine, this, std::placeholders::_1));
    methods.insert("houghLineP", std::bind(&Improcess::houghLineP, this, std::placeholders::_1));
    methods.insert("houghCircle", std::bind(&Improcess::houghCircle, this, std::placeholders::_1));
    methods.insert("remap", std::bind(&Improcess::remap, this, std::placeholders::_1));
    methods.insert("affineByRotate", std::bind(&Improcess::affineByRotate, this, std::placeholders::_1));
    methods.insert("affineByPoint", std::bind(&Improcess::affineByPoint, this, std::placeholders::_1));
    methods.insert("matchTemplate", std::bind(&Improcess::matchTemplate, this, std::placeholders::_1));
    methods.insert("findMaxContours", std::bind(&Improcess::findMaxContours, this, std::placeholders::_1));
    methods.insert("findContoursByLR", std::bind(&Improcess::findContoursByLR, this, std::placeholders::_1));
    methods.insert("findContoursByMeanZero", std::bind(&Improcess::findContoursByMeanZero, this, std::placeholders::_1));

}

void Improcess::process(const QString &method, const QString &srcPath)
{
    return methods[method](srcPath);
}

QImage Improcess::convertToQImage(const Mat &src)
{
    QImage img;
    int channel = src.channels();
    switch (channel) {
    case 3:
    {
        Mat dst;
        cv::cvtColor(src, dst, COLOR_BGR2RGB);
        img = QImage(src.data, src.cols, src.rows, QImage::Format_RGB888);
        break;
    }
    case 4:
        img = QImage(src.data, src.cols, src.rows, QImage::Format_ARGB32);
        break;
    default:
        img = QImage(src.cols, src.rows, QImage::Format_Indexed8);
        uchar *data = src.data;
        for(int i = 0 ; i < src.rows ; i++){
            uchar* rowdata = img.scanLine( i );
            memcpy(rowdata, data , src.cols);
            data += src.cols;
        }
        break;
    }
    return img;
}

/* 二值化 */
void Improcess::threshold(const QString &srcPath)
{
    Mat src = cv::imread(srcPath.toStdString());
    if (src.empty()) {
        return;
    }
	/* 转化为灰度图 */
    Mat gray;
    cvtColor(src, gray, COLOR_BGR2GRAY);
	/* 二值化 */
    Mat dst;
    cv::threshold(gray, dst, 100, 255, THRESH_BINARY_INV);
    QImage img = convertToQImage(dst);
    emit processFinished(img);
	return;
}
void Improcess::adaptiveThreshold(const QString &srcPath)
{
    Mat src = cv::imread(srcPath.toStdString());
    if (src.empty()) {
        return;
    }
	/* 转化为灰度图 */
    Mat gray;
    cvtColor(src, gray, COLOR_BGR2GRAY);
	/* 二值化 */
	int blockSize = 25;
	int constValue = 10;
    Mat dst;
    cv::adaptiveThreshold(gray, dst, 255, THRESH_TOZERO, THRESH_BINARY_INV, blockSize, constValue);
    QImage img = convertToQImage(dst);
    emit processFinished(img);
    return;
}
/* 滤波 */
/* 方框滤波 */
void Improcess::boxFilter(const QString &srcPath)
{
    Mat src = cv::imread(srcPath.toStdString());
    if (src.empty()) {
        return;
    }
	/* 滤波 */
    Mat dst;
    cv::boxFilter(src, dst, -1, Size(5, 5)); // -1 specific the depth of image
    QImage img = convertToQImage(dst);
    emit processFinished(img);
	return;
}
/* 均值滤波 */
void Improcess::blur(const QString &srcPath)
{
    Mat src = cv::imread(srcPath.toStdString());
    if (src.empty()) {
        return;
    }
	/* 均值滤波是方框滤波的特殊情况，
	 * 缺点是不能很好地保护图像细节，
	 * 在图像去噪的同时也破坏了图像的部分细节。
	 * */
    Mat dst;
    cv::blur(src, dst, Size(5, 5)); // -1 specific the depth of image
    QImage img = convertToQImage(dst);
    emit processFinished(img);
    return;
}
/* 高斯滤波 */
void Improcess::gaussianBlur(const QString &srcPath)
{
    Mat src = cv::imread(srcPath.toStdString());
    if (src.empty()) {
        return;
    }
	/* 高斯滤波可以消除高斯噪声，广泛应用于減噪过程 */
    Mat dst;
    GaussianBlur(src, dst, Size(5, 5), 0, 0);
    QImage img = convertToQImage(dst);
    emit processFinished(img);
	return;
}
/* 形态学运算 */
/* 腐蚀 */
void Improcess::erode(const QString &srcPath)
{
    Mat src = cv::imread(srcPath.toStdString());
    if (src.empty()) {
        return;
    }
    /* 自定义卷积核, MORPH_RECT表示矩形卷积核 */
    Mat element = getStructuringElement(cv::MORPH_RECT, Size(15, 15));
	/* 腐蚀操作 */
    Mat dst;
    cv::erode(src, dst, element);
    QImage img = convertToQImage(dst);
    emit processFinished(img);
	return;
}
/* 膨胀 */
void Improcess::dilate(const QString &srcPath)
{
    Mat src = cv::imread(srcPath.toStdString());
    if (src.empty()) {
        return;
    }
	/* 自定义卷积核 */
    Mat element = getStructuringElement(MORPH_RECT, Size(15, 15));
	/* 膨胀操作 */
    Mat dst;
    cv::dilate(src, dst, element);
    QImage img = convertToQImage(dst);
    emit processFinished(img);
	return;
}
/* 开运算 */
void Improcess::morphOpen(const QString &srcPath)
{
    Mat src = cv::imread(srcPath.toStdString());
    if (src.empty()) {
        return;
    }
	/* 自定义卷积核 */
    Mat element = getStructuringElement(MORPH_RECT, Size(15, 15));
	/* 开运算操作 */
    Mat dst;
    morphologyEx(src, dst, MORPH_OPEN, element);
    QImage img = convertToQImage(dst);
    emit processFinished(img);
	return;
}

/* 闭运算 */
void Improcess::morphClose(const QString &srcPath)
{
    Mat src = cv::imread(srcPath.toStdString());
    if (src.empty()) {
        return;
    }
	/* 自定义卷积核 */
    Mat element = getStructuringElement(MORPH_RECT, Size(15, 15));
	/* 闭运算操作 */
    Mat dst;
    morphologyEx(src, dst, MORPH_CLOSE, element);
    QImage img = convertToQImage(dst);
    emit processFinished(img);
	return;
}

/* 顶帽 */
void Improcess::morphTopHat(const QString &srcPath)
{
    Mat src = cv::imread(srcPath.toStdString());
    if (src.empty()) {
        return;
    }
	/* 自定义卷积核 */
    Mat element = getStructuringElement(MORPH_RECT, Size(15, 15));
	/* 顶帽操作 */
    Mat dst;
    morphologyEx(src, dst, MORPH_TOPHAT, element);
    QImage img = convertToQImage(dst);
    emit processFinished(img);
	return;
}

/* 黑帽 */
void Improcess::morphBlackHat(const QString &srcPath)
{
    Mat src = cv::imread(srcPath.toStdString());
    if (src.empty()) {
        return;
    }
	/* 自定义卷积核 */
    Mat element = getStructuringElement(MORPH_RECT, Size(15, 15));
	/* 黑帽操作 */
    Mat dst;
    morphologyEx(src, dst, MORPH_BLACKHAT, element);
    QImage img = convertToQImage(dst);
    emit processFinished(img);
	return;
}

/* 形态学梯度 */
void Improcess::morphGrident(const QString &srcPath)
{
    Mat src = cv::imread(srcPath.toStdString());
    if (src.empty()) {
        return;
    }
	/* 自定义卷积核 */
    Mat element = getStructuringElement(MORPH_RECT, Size(15, 15));
	/* 形态学梯度操作 */
    Mat dst;
    morphologyEx(src, dst, MORPH_GRADIENT, element);
    QImage img = convertToQImage(dst);
    emit processFinished(img);
	return;
}

/* 边缘检测 */
/* canny算子 */
void Improcess::canny(const QString &srcPath)
{
    Mat src = cv::imread(srcPath.toStdString());
    if (src.empty()) {
        return;
    }
    /* 转化为灰度图 */
    Mat gray;
    cvtColor(src, gray, COLOR_BGR2GRAY);
    /* 滤波降噪 */
    cv::blur(gray, gray, Size(3, 3));
    /* canny算子操作 */
    Mat dst;
    Canny(gray, dst, 30, 120);
    QImage img = convertToQImage(dst);
    emit processFinished(img);
    return;
}
/* sobel算子 */
void Improcess::sobel(const QString &srcPath)
{
    Mat src = cv::imread(srcPath.toStdString());
    if (src.empty()) {
        return;
    }
    /* 转化为灰度图 */
    Mat gray;
    cvtColor(src, gray, COLOR_BGR2GRAY);
    /* 计算横向梯度 */
    Mat xGradImg;
    Sobel(gray, xGradImg, CV_16S, 1, 0);
    /* 计算纵向梯度 */
    Mat yGradImg;
    Sobel(gray, yGradImg, CV_16S, 0, 1);
    /* 合并梯度 */
    Mat dst;
    addWeighted(xGradImg, 0.5, yGradImg, 0.5, 0, dst);
    QImage img = convertToQImage(dst);
    emit processFinished(img);
    return;
}
/* laplace算子 */
void Improcess::laplace(const QString &srcPath)
{
    Mat src = cv::imread(srcPath.toStdString());
    if (src.empty()) {
        return;
    }
    /* 滤波降噪 */
    Mat gaussBlurImg;
    GaussianBlur(src, gaussBlurImg, Size(3, 3), 0);
    /* 转化为灰度图 */
    Mat gray;
    cvtColor(gaussBlurImg, gray, COLOR_BGR2GRAY);
    /* Laplace算子操作 */
    Mat filterImg;
    Laplacian(gray, filterImg, CV_16S, 3);
    /* 计算绝对值 */
    Mat dst;
    convertScaleAbs(filterImg, dst);
    QImage img = convertToQImage(dst);
    emit processFinished(img);
    return;
}
/* 霍夫变换 */
/* 霍夫线变换 */
void Improcess::houghLine(const QString &srcPath)
{
    Mat src = cv::imread(srcPath.toStdString());
    if (src.empty()) {
        return;
    }
    /* 边缘检测 */
    Mat edge;
    Canny(src, edge, 50, 200, 3);
    /* 转化为BGR图 */
    Mat dst;
    cvtColor(edge, dst, COLOR_GRAY2BGR);
    /* 获取所有直线的集合 */
    std::vector<Vec2f> lines;
    /* 第五个参数决定直线的精准度，值越大精准度越高，检测到的直线就越少，速度也越快 */
    HoughLines(edge, lines, 1, CV_PI / 180, 300, 0, 0);
    std::cout <<"lines num: "<<lines.size()<<std::endl;
    /* 画直线 */
    for (unsigned int i = 0; i < lines.size(); i++) {
        float r = lines[i][0];
        float theta = lines[i][1];
        Point p1;
        Point p2;
        p1.x = cvRound(r * cos(theta) - 1000 * sin(theta));
        p1.y = cvRound(r * sin(theta) + 1000 * cos(theta));
        p2.x = cvRound(r * cos(theta) + 1000 * sin(theta));
        p2.y = cvRound(r * sin(theta) - 1000 * cos(theta));
        line(dst, p1, p2, Scalar(55, 100, 195), 1, LINE_AA);
    }
    QImage img = convertToQImage(dst);
    emit processFinished(img);
    return;
}
void Improcess::houghLineP(const QString &srcPath)
{
    Mat src = cv::imread(srcPath.toStdString());
    if (src.empty()) {
        return;
    }
    /* 边缘检测 */
    Mat edge;
    Canny(src, edge, 50, 200);
    /* 转化为BGR图 */
    Mat dst;
    cvtColor(edge, dst, COLOR_GRAY2BGR);
    /* 直线检测 */
    std::vector<Vec4i> lines;
    HoughLinesP(edge, lines, 1, CV_PI / 180, 200);
    /* 画直线 */
    for (unsigned int i = 0; i < lines.size(); i++) {
        Vec4i lineTermin = lines[i];
        line(dst, Point(lineTermin[0], lineTermin[1]), Point(lineTermin[2], lineTermin[3]),
                Scalar(55, 100, 195), 2, LINE_AA);
    }
    QImage img = convertToQImage(dst);
    emit processFinished(img);
    return;
}

/* 霍夫圆变换 */
void Improcess::houghCircle(const QString &srcPath)
{
    Mat src = cv::imread(srcPath.toStdString());
    if (src.empty()) {
        return;
    }
    Mat dst(src);
    /* 转化为灰度图 */
    Mat gray;
    cvtColor(src, gray, COLOR_BGR2GRAY);
    /* 高斯滤波 */
    GaussianBlur(gray, gray, Size(9, 9), 2, 2);
    /* 检测圆 */
    std::vector<Vec3f> circles;
    HoughCircles(gray, circles, HOUGH_GRADIENT, 1.5, 10, 280, 150);
    /* 画圆 */
    for (unsigned int i = 0; i < circles.size(); i++) {
        Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
        int radius = cvRound(circles[i][2]);
        circle(dst, center, 4, Scalar(0, 255, 0), -1, LINE_8, 0);
        circle(dst, center, radius, Scalar(155, 50, 255), 3, LINE_8, 0);
    }
    QImage img = convertToQImage(dst);
    emit processFinished(img);
    return;
}
/* 重映射与放射 */
/* 重映射实现翻转 */
void Improcess::remap(const QString &srcPath)
{
    Mat src = cv::imread(srcPath.toStdString());
    if (src.empty()) {
        return;
    }
    /* 上下翻转180 */
    Mat dst(src.size(), src.type());
    Mat imgx(src.size(), CV_32FC1);
    Mat imgy(src.size(), CV_32FC1);
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j <src.cols; j++) {
            imgx.at<float>(i, j) = static_cast<float>(j);
            imgy.at<float>(i, j) = static_cast<float>(src.rows - i);
        }
    }
    /* 重映射 */
    cv::remap(src, dst, imgx, imgy, INTER_LINEAR, BORDER_DEFAULT, Scalar(0, 0, 0));
    QImage img = convertToQImage(dst);
    emit processFinished(img);
    return;
}
/* 仿射实现旋转与缩放 */
void Improcess::affineByRotate(const QString &srcPath)
{
    Mat src = cv::imread(srcPath.toStdString());
    if (src.empty()) {
        return;
    }
    Point2f center(src.rows / 2, src.cols/ 2);
    double angle = 45.0;
    double scale = 0.5;
    Mat dst;
    Mat transformMat = getRotationMatrix2D(center, angle, scale);
    warpAffine(src, dst, transformMat, src.size());
    QImage img = convertToQImage(dst);
    emit processFinished(img);
    return;
}
void Improcess::affineByPoint(const QString &srcPath)
{
    Mat src = cv::imread(srcPath.toStdString());
    if (src.empty()) {
        return;
    }
    /* 3点决定一个平面 */
    Point2f srcPoint[3];
    Point2f dstPoint[3];
    /* 变换前的3个点 */
    srcPoint[0] = Point2f(0, 0);
    srcPoint[1] = Point2f(0, src.rows - 1);
    srcPoint[2] = Point2f(src.cols - 1, 0);
    /* 变换后的3个点 */
    dstPoint[0] = Point2f(0, src.rows * 0.4);
    dstPoint[1] = Point2f(src.cols * 0.25, src.rows * 0.75);
    dstPoint[2] = Point2f(src.cols * 0.75, src.rows * 0.25);
    /* 变换矩阵 */
    Mat transformMat = getAffineTransform(srcPoint, dstPoint);
    /* 仿射变换 */
    Mat dst;
    warpAffine(src, dst, transformMat, src.size());
    QImage img = convertToQImage(dst);
    emit processFinished(img);
    return;
}
/* 模板匹配 */
void Improcess::matchTemplate(const QString &srcPath)
{
    Mat src = cv::imread(srcPath.toStdString());
    if (src.empty()) {
        return;
    }
    Mat dst(src);
    /* 读取模板 */
    Mat temp = imread("D:/MyPicture/tiger.jpg");
    if (temp.empty()) {
        return;
    }
    /* 匹配模板 */
    Mat result;
    int result_rows = src.rows - temp.rows + 1;
    int result_cols = src.cols - temp.cols + 1;
    result.create(result_cols, result_rows, CV_32FC1);
    cv::matchTemplate(src, temp, result, TM_SQDIFF_NORMED);
    /* 查找匹配度最高的位置 */
    double minVal = -1;
    Point minLoc;
    minMaxLoc(result, &minVal, nullptr, &minLoc, nullptr);
    /* 标注位置 */
    std::string content = "square match rate: ";
    content += std::to_string(minVal);
    putText(dst, content, Point(20, 20), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 255), 2);
    rectangle(dst, minLoc, Point(minLoc.x + temp.cols, minLoc.y + temp.rows),
              Scalar(0, 255, 0), 2, LINE_8, 0);
    QImage img = convertToQImage(dst);
    emit processFinished(img);
    return;
}
/* 图像修复 */

/* 轮廓查找与多边形包围 */
Mat Improcess::getMaxContours(Mat& src, Mat& edge)
{
    /* 获取轮廓 */
    Mat dst;
    std::vector<std::vector<Point> > contours;
    std::vector<Vec4i> hierarchy;
    cv::findContours(edge, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    if (contours.size() < 1) {
        std::cout<<"no contour"<<std::endl;
        return dst;
    }
    /* 查找面积最大轮廓 */
    int maxIndex = 0;
    double maxArea = 0;
    for (unsigned int i = 0; i < contours.size(); i++) {
        double tmpArea = contourArea(contours[i]);
        if (tmpArea > maxArea) {
            maxArea = tmpArea;
            maxIndex = i;
        }
    }
    /* 查找凸包 */
    std::vector<std::vector<Point> > convexs(1);
    convexHull(contours[maxIndex], convexs[0]);
    /* 绘制轮廓 */
    Mat tmpImg;
    tmpImg.create(src.size(), src.type());
    tmpImg.zeros(src.size(), src.type());
    drawContours(tmpImg, contours, maxIndex, Scalar(255, 255, 255), -1);
    //drawContours(tmpImg, convexs, 0, Scalar(255, 255, 255), -1);
    /* ROI提取 */
    Mat roiImg;
    roiImg.create(src.size(), src.type());
    tmpImg = ~tmpImg;
    addWeighted(src, 1, tmpImg, 1, 0, roiImg);
    /* 最小矩形包围 */
    Rect minRect;
    minRect = boundingRect(contours[maxIndex]);
    roiImg(minRect);
    return roiImg;
}

void Improcess::findMaxContours(const QString &srcPath)
{
    Mat src = cv::imread(srcPath.toStdString());
    if (src.empty()) {
        return;
    }
    /* 转灰度图 */
    Mat gray;
    cvtColor(src, gray, COLOR_BGR2GRAY);
    /* 滤波降噪 */
    Mat outImg;
    GaussianBlur(gray, outImg, Size(3, 3), 0);
    /* 边缘检测 */
    Canny(outImg, outImg, 50, 255);
    /* 获取最大轮廓 */
    Mat dst = getMaxContours(src, outImg);
    QImage img = convertToQImage(dst);
    emit processFinished(img);
    return;
}

void Improcess::findContoursByLR(const QString &srcPath)
{
    Mat src = cv::imread(srcPath.toStdString());
    if (src.empty()) {
        return;
    }
    Mat outImg;
    /* RGB过滤 */
#if 0
    double w0 = -16.0;
    double w1 = -0.4;
    double w2 = 0.5;
    double w3 = 0.0;
#else
    double w0 = -3.0;
    double w1 = -0.2;
    double w2 = 0.4;
    double w3 = -0.12;
#endif
    src.copyTo(outImg);
    for (int i = 0; i < outImg.rows; i++) {
        for (int j = 0; j < outImg.cols; j++) {
             Vec3b x = outImg.at<Vec3b>(i, j);
             double y = 0.0;
             /* logistics回归 */
             y = (w0 +  w1 * x[0] + w2 * x[1] + w3 * x[2]) / 255.0;
             y = exp(y)/(1 + exp(y));
             if (y > 0.5) {
                 outImg.at<Vec3b>(i, j)[0] = 0.0;
                 outImg.at<Vec3b>(i, j)[1] = 0.0;
                 outImg.at<Vec3b>(i, j)[2] = 0.0;
             } else {
#if 1
                 outImg.at<Vec3b>(i, j)[0] = 255.0;
                 outImg.at<Vec3b>(i, j)[1] = 255.0;
                 outImg.at<Vec3b>(i, j)[2] = 255.0;
#endif
             }
        }
    }
    Mat dst;
    outImg.copyTo(dst);
#if 1
    cvtColor(outImg, outImg, COLOR_BGR2GRAY);
    /* 腐蚀 */
    Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
    cv::dilate(outImg, outImg, element);
    cv::erode(outImg, outImg, element);
    /* 滤波降噪 */
    GaussianBlur(outImg, outImg, Size(9, 9), 0);
    /* 边缘检测 */
    Canny(outImg, outImg, 150, 150);
    cv::dilate(outImg, outImg, element);
    /* 获取最大轮廓 */
    dst = getMaxContours(src, outImg);
#endif
    QImage img = convertToQImage(dst);
    emit processFinished(img);
    return;
}
void Improcess::findContoursByMeanZero(const QString &srcPath)
{
    Mat src = cv::imread(srcPath.toStdString());
    if (src.empty()) {
        return;
    }
    Mat outImg;
    cvtColor(src, outImg, COLOR_BGR2GRAY);
    /* 计算灰度平均值 */
    double gray = 0.0;
    for (int i = 0; i < outImg.rows; i++) {
        for (int j = 0; j < outImg.cols; j++) {
            gray += outImg.at<uchar>(i, j);
        }
    }
    gray =  gray / double(outImg.rows * outImg.cols);
    std::cout<<"gray: "<<gray<<std::endl;
    /* 二值化 */
    cv::threshold(outImg, outImg, gray - 40, 255, THRESH_BINARY_INV);
    /* 腐蚀 */
    Mat element = getStructuringElement(MORPH_RECT, Size(9, 9));
    cv::erode(outImg, outImg, element);
    /* 滤波降噪 */
    GaussianBlur(outImg, outImg, Size(5, 5), 0);
    /* 边缘检测 */
    Canny(outImg, outImg, 30, 30);
    //outImg.copyTo(dst);
    element = getStructuringElement(MORPH_RECT, Size(5, 5));
    /* 膨胀操作 */
    cv::dilate(outImg, outImg, element);
    // outImg.copyTo(dst);
    /* 获取最大轮廓 */
    Mat dst = getMaxContours(src, outImg);
    QImage img = convertToQImage(dst);
    emit processFinished(img);
    return;
}
