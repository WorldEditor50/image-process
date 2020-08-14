 #include "improcess.h"
void Improcess_Test(const char* srcName, const char* dstName)
{
	/* 打开图片 */
	Mat srcImg = imread(srcName);
	if (srcImg.empty()) {
		return;
	}
	Mat dstImg;
    //Improcess_FindContoursByLR(srcImg, dstImg);
    Improcess_FindContoursByMeanZero(srcImg, dstImg);
    //Improcess_Sampling(srcImg, dstImg);
	/* 保存图片 */
    if (dstImg.empty()) {
        return;
    }
	imwrite(dstName, dstImg);
	return;
}
/* 二值化 */
void Improcess_Threshold(Mat& srcImg, Mat& dstImg)
{
	/* 转化为灰度图 */
	Mat grayImg;
	cvtColor(srcImg, grayImg, COLOR_BGR2GRAY);
	/* 二值化 */
    threshold(grayImg, dstImg, 100, 255, THRESH_BINARY_INV);
	return;
}
void Improcess_AdaptiveThreshold(Mat& srcImg, Mat& dstImg)
{
	/* 转化为灰度图 */
	Mat grayImg;
	cvtColor(srcImg, grayImg, COLOR_BGR2GRAY);
	/* 二值化 */
	int blockSize = 25;
	int constValue = 10;
    adaptiveThreshold(grayImg, dstImg, 255, THRESH_TOZERO, THRESH_BINARY_INV, blockSize, constValue);
	return;
}
/* 滤波 */
/* 方框滤波 */
void Improcess_BoxFilter(Mat& srcImg, Mat& dstImg)
{
	/* 滤波 */
	boxFilter(srcImg, dstImg, -1, Size(5, 5)); // -1 specific the depth of image
	return;
}
/* 均值滤波 */
void Improcess_Blur(Mat& srcImg, Mat& dstImg)
{
	/* 均值滤波是方框滤波的特殊情况，
	 * 缺点是不能很好地保护图像细节，
	 * 在图像去噪的同时也破坏了图像的部分细节。
	 * */
	blur(srcImg, dstImg, Size(5, 5)); // -1 specific the depth of image
	return;
}
/* 高斯滤波 */
void Improcess_GaussianBlur(Mat& srcImg, Mat& dstImg)
{
	/* 高斯滤波可以消除高斯噪声，广泛应用于減噪过程 */
	GaussianBlur(srcImg, dstImg, Size(5, 5), 0, 0);
	return;
}
/* 形态学运算 */
/* 腐蚀 */
void Improcess_Erode(Mat& srcImg, Mat& dstImg)
{
	/* 自定义卷积核, MORPH_RECT表示矩形卷积核 */
	Mat element = getStructuringElement(MORPH_RECT, Size(15, 15));
	/* 腐蚀操作 */
	erode(srcImg, dstImg, element);
	return;
}
/* 膨胀 */
void Improcess_Dilate(Mat& srcImg, Mat& dstImg)
{
	/* 自定义卷积核 */
	Mat element = getStructuringElement(MORPH_RECT, Size(15, 15));
	/* 膨胀操作 */
	dilate(srcImg, dstImg, element);
	return;
}
/* 开运算 */
void Improcess_MorphOpen(Mat& srcImg, Mat& dstImg)
{
	/* 自定义卷积核 */
	Mat element = getStructuringElement(MORPH_RECT, Size(15, 15));
	/* 开运算操作 */
	morphologyEx(srcImg, dstImg, MORPH_OPEN, element);
	return;
}

/* 闭运算 */
void Improcess_MorphClose(Mat& srcImg, Mat& dstImg)
{
	/* 自定义卷积核 */
	Mat element = getStructuringElement(MORPH_RECT, Size(15, 15));
	/* 闭运算操作 */
	morphologyEx(srcImg, dstImg, MORPH_CLOSE, element);
	return;
}

/* 顶帽 */
void Improcess_MorphTopHat(Mat& srcImg, Mat& dstImg)
{
	/* 自定义卷积核 */
	Mat element = getStructuringElement(MORPH_RECT, Size(15, 15));
	/* 顶帽操作 */
	morphologyEx(srcImg, dstImg, MORPH_TOPHAT, element);
	return;
}

/* 黑帽 */
void Improcess_MorphBlackHat(Mat& srcImg, Mat& dstImg)
{
	/* 自定义卷积核 */
	Mat element = getStructuringElement(MORPH_RECT, Size(15, 15));
	/* 黑帽操作 */
	morphologyEx(srcImg, dstImg, MORPH_BLACKHAT, element);
	return;
}

/* 形态学梯度 */
void Improcess_MorphGrident(Mat& srcImg, Mat& dstImg)
{
	/* 自定义卷积核 */
	Mat element = getStructuringElement(MORPH_RECT, Size(15, 15));
	/* 形态学梯度操作 */
	morphologyEx(srcImg, dstImg, MORPH_GRADIENT, element);
	return;
}

/* 边缘检测 */
/* canny算子 */
void Improcess_Canny(Mat& srcImg, Mat& dstImg)
{
    /* 转化为灰度图 */
    Mat grayImg;
    cvtColor(srcImg, grayImg, COLOR_BGR2GRAY);
    /* 滤波降噪 */
    blur(grayImg, grayImg, Size(3, 3));
    /* canny算子操作 */
    Canny(grayImg, dstImg, 30, 120);
    return;
}
/* sobel算子 */
void Improcess_Sobel(Mat& srcImg, Mat& dstImg)
{
    /* 转化为灰度图 */
    Mat grayImg;
    cvtColor(srcImg, grayImg, COLOR_BGR2GRAY);
    /* 计算横向梯度 */
    Mat xGradImg;
    Sobel(grayImg, xGradImg, CV_16S, 1, 0);
    /* 计算纵向梯度 */
    Mat yGradImg;
    Sobel(grayImg, yGradImg, CV_16S, 0, 1);
    /* 合并梯度 */
    addWeighted(xGradImg, 0.5, yGradImg, 0.5, 0, dstImg);
    return;
}
/* laplace算子 */
void Improcess_Laplace(Mat& srcImg, Mat& dstImg)
{
    /* 滤波降噪 */
    Mat gaussBlurImg;
    GaussianBlur(srcImg, gaussBlurImg, Size(3, 3), 0);
    /* 转化为灰度图 */
    Mat grayImg;
    cvtColor(gaussBlurImg, grayImg, COLOR_BGR2GRAY);
    /* Laplace算子操作 */
    Mat filterImg;
    Laplacian(grayImg, filterImg, CV_16S, 3);
    /* 计算绝对值 */
    convertScaleAbs(filterImg, dstImg);
    return;
}
/* 霍夫变换 */
/* 霍夫线变换 */
void Improcess_HoughLine(Mat& srcImg, Mat& dstImg)
{
    /* 边缘检测 */
    Mat edgeImg;
    Canny(srcImg, edgeImg, 50, 200, 3);
    /* 转化为BGR图 */
    Mat grayImg;
    cvtColor(edgeImg, dstImg, COLOR_GRAY2BGR);
    /* 获取所有直线的集合 */
    vector<Vec2f> lines;
    /* 第五个参数决定直线的精准度，值越大精准度越高，检测到的直线就越少，速度也越快 */
    HoughLines(edgeImg, lines, 1, CV_PI / 180, 300, 0, 0);
    cout <<"lines num: "<<lines.size()<<endl;
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
        line(dstImg, p1, p2, Scalar(55, 100, 195), 1, LINE_AA);
    }
    return;
}
void Improcess_HoughLineP(Mat& srcImg, Mat& dstImg)
{
    /* 边缘检测 */
    Mat edgeImg;
    Canny(srcImg, edgeImg, 50, 200);
    /* 转化为BGR图 */
    cvtColor(edgeImg, dstImg, COLOR_GRAY2BGR);
    /* 直线检测 */
    vector<Vec4i> lines;
    HoughLinesP(edgeImg, lines, 1, CV_PI / 180, 200);
    /* 画直线 */
    for (unsigned int i = 0; i < lines.size(); i++) {
        Vec4i lineTermin = lines[i];
        line(dstImg, Point(lineTermin[0], lineTermin[1]), Point(lineTermin[2], lineTermin[3]),
                Scalar(55, 100, 195), 2, LINE_AA);
    }
    return;
}

/* 霍夫圆变换 */
void Improcess_HoughCircle(Mat& srcImg, Mat& dstImg)
{
    srcImg.copyTo(dstImg);
    /* 转化为灰度图 */
    Mat grayImg;
    cvtColor(srcImg, grayImg, COLOR_BGR2GRAY);
    /* 高斯滤波 */
    GaussianBlur(grayImg, grayImg, Size(9, 9), 2, 2);
    /* 检测圆 */
    vector<Vec3f> circles;
    HoughCircles(grayImg, circles, HOUGH_GRADIENT, 1.5, 10, 280, 150);
    /* 画圆 */
    for (unsigned int i = 0; i < circles.size(); i++) {
        Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
        int radius = cvRound(circles[i][2]);
        circle(dstImg, center, 4, Scalar(0, 255, 0), -1, LINE_8, 0);
        circle(dstImg, center, radius, Scalar(155, 50, 255), 3, LINE_8, 0);
    }
    return;
}
/* 重映射与放射 */
/* 重映射实现翻转 */
void Improcess_Remap(Mat& srcImg, Mat& dstImg)
{
    Mat img_x;
    Mat img_y;
    /* 上下翻转180 */
    dstImg.create(srcImg.size(), srcImg.type());
    img_x.create(srcImg.size(), CV_32FC1);
    img_y.create(srcImg.size(), CV_32FC1);
    for (int i = 0; i < srcImg.rows; i++) {
        for (int j = 0; j <srcImg.cols; j++) {
            img_x.at<float>(i, j) = static_cast<float>(j);
            img_y.at<float>(i, j) = static_cast<float>(srcImg.rows - i);
        }
    }
    /* 重映射 */
    remap(srcImg, dstImg, img_x, img_y, INTER_LINEAR, BORDER_DEFAULT, Scalar(0, 0, 0));
    return;
}
/* 仿射实现旋转与缩放 */
void Improcess_AffineByRoate(Mat& srcImg, Mat& dstImg)
{
    Point2f center(srcImg.rows / 2, srcImg.cols/ 2);
    double angle = 45.0;
    double scale = 0.5;
    Mat transformMat = getRotationMatrix2D(center, angle, scale);
    warpAffine(srcImg, dstImg, transformMat, srcImg.size());
    return;
}
void Improcess_AffineByPoint(Mat& srcImg, Mat& dstImg)
{
    /* 3点决定一个平面 */
    Point2f srcPoint[3];
    Point2f dstPoint[3];
    /* 变换前的3个点 */
    srcPoint[0] = Point2f(0, 0);
    srcPoint[1] = Point2f(0, srcImg.rows - 1);
    srcPoint[2] = Point2f(srcImg.cols - 1, 0);
    /* 变换后的3个点 */
    dstPoint[0] = Point2f(0, srcImg.rows * 0.4);
    dstPoint[1] = Point2f(srcImg.cols * 0.25, srcImg.rows * 0.75);
    dstPoint[2] = Point2f(srcImg.cols * 0.75, srcImg.rows * 0.25);
    /* 变换矩阵 */
    Mat transformMat = getAffineTransform(srcPoint, dstPoint);
    /* 仿射变换 */
    warpAffine(srcImg, dstImg, transformMat, srcImg.size());
    return;
}
/* 模板匹配 */
void Improcess_MatchTemplate(Mat& srcImg, Mat& dstImg)
{
    srcImg.copyTo(dstImg);
    /* 读取模板 */
    Mat temp = imread("D:/MyPicture/tiger.jpg");
    if (temp.empty()) {
        return;
    }
    /* 匹配模板 */
    Mat result;
    int result_rows = srcImg.rows - temp.rows + 1;
    int result_cols = srcImg.cols - temp.cols + 1;
    result.create(result_cols, result_rows, CV_32FC1);
    matchTemplate(srcImg, temp, result, TM_SQDIFF_NORMED);
    /* 查找匹配度最高的位置 */
    double minVal = -1;
    Point minLoc;
    minMaxLoc(result, &minVal, nullptr, &minLoc, nullptr);
    /* 标注位置 */
    string content = "square match rate: ";
    content += std::to_string(minVal);
    putText(dstImg, content, Point(20, 20), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 255), 2);
    rectangle(dstImg, minLoc, Point(minLoc.x + temp.cols, minLoc.y + temp.rows),
              Scalar(0, 255, 0), 2, LINE_8, 0);
    return;
}
/* 图像修复 */

/* 轮廓查找与多边形包围 */
void Improcess_GetMaxContours(Mat& srcImg, Mat& edgeImg, Mat& dstImg)
{
    /* 获取轮廓 */
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    findContours(edgeImg, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    if (contours.size() < 1) {
        cout<<"no contour"<<endl;
        return;
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
    vector<vector<Point> > convexs(1);
    convexHull(contours[maxIndex], convexs[0]);
    /* 绘制轮廓 */
    Mat tmpImg;
    tmpImg.create(srcImg.size(), srcImg.type());
    tmpImg.zeros(srcImg.size(), srcImg.type());
    drawContours(tmpImg, contours, maxIndex, Scalar(255, 255, 255), -1);
    //drawContours(tmpImg, convexs, 0, Scalar(255, 255, 255), -1);
    /* ROI提取 */
    Mat roiImg;
    roiImg.create(srcImg.size(), srcImg.type());
    tmpImg = ~tmpImg;
    addWeighted(srcImg, 1, tmpImg, 1, 0, roiImg);
    /* 最小矩形包围 */
    Rect minRect;
    minRect = boundingRect(contours[maxIndex]);
    roiImg(minRect);
    roiImg.copyTo(dstImg);
    return;
}
void Improcess_FindMaxContours(Mat& srcImg, Mat& dstImg)
{
    /* 转灰度图 */
    Mat grayImg;
    cvtColor(srcImg, grayImg, COLOR_BGR2GRAY);
    /* 滤波降噪 */
    Mat outImg;
    GaussianBlur(grayImg, outImg, Size(3, 3), 0);
    /* 边缘检测 */
    Canny(outImg, outImg, 50, 255);
    /* 获取最大轮廓 */
    Improcess_GetMaxContours(srcImg, outImg, dstImg);
    return;
}

void Improcess_FindContoursByLR(Mat& srcImg, Mat& dstImg)
{
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
    srcImg.copyTo(outImg);
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
    outImg.copyTo(dstImg);
#if 1
    cvtColor(outImg, outImg, COLOR_BGR2GRAY);
    /* 腐蚀 */
    Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
    dilate(outImg, outImg, element);
    erode(outImg, outImg, element);
    /* 滤波降噪 */
    GaussianBlur(outImg, outImg, Size(9, 9), 0);
    /* 边缘检测 */
    Canny(outImg, outImg, 150, 150);
    dilate(outImg, outImg, element);
    /* 获取最大轮廓 */
    Improcess_GetMaxContours(srcImg, outImg, dstImg);
#endif
    return;
}
void Improcess_FindContoursByMeanZero(Mat& srcImg, Mat& dstImg)
{
    Mat outImg;
    cvtColor(srcImg, outImg, COLOR_BGR2GRAY);
    /* 计算灰度平均值 */
    double gray = 0.0;
    for (int i = 0; i < outImg.rows; i++) {
        for (int j = 0; j < outImg.cols; j++) {
            gray += outImg.at<uchar>(i, j);
        }
    }
    gray =  gray / double(outImg.rows * outImg.cols);
    cout<<"gray: "<<gray<<endl;
    /* 二值化 */
    threshold(outImg, outImg, gray - 40, 255, THRESH_BINARY_INV);
    /* 腐蚀 */
    Mat element = getStructuringElement(MORPH_RECT, Size(9, 9));
    erode(outImg, outImg, element);
    outImg.copyTo(dstImg);
    /* 滤波降噪 */
    GaussianBlur(outImg, outImg, Size(5, 5), 0);
    /* 边缘检测 */
    Canny(outImg, outImg, 30, 30);
    //outImg.copyTo(dstImg);
    element = getStructuringElement(MORPH_RECT, Size(5, 5));
    /* 膨胀操作 */
    dilate(outImg, outImg, element);
    // outImg.copyTo(dstImg);
    /* 获取最大轮廓 */
    Improcess_GetMaxContours(srcImg, outImg, dstImg);
    return;
}
void Improcess_FindContours(Mat& srcImg, Mat& dstImg)
{
    Mat inImg;
    srcImg.copyTo(inImg);
    cvtColor(inImg, inImg, COLOR_BGR2GRAY);
    /* 填充 */
    int w = getOptimalDFTSize(inImg.cols);
    int h = getOptimalDFTSize(inImg.rows);
    Mat padImg;
    copyMakeBorder(inImg, padImg, 0, h - inImg.rows, 0, w - inImg.cols, BORDER_CONSTANT, Scalar::all(0));
    padImg.convertTo(padImg, CV_32FC1);
    /* 中心化 */
    for (int i = 0; i < padImg.rows; i++) {
        for (int j = 0; j < padImg.cols; j++) {
            padImg.at<float>(i, j) *= pow(-1, i + j);
        }
    }
#if 0
    Mat plane[2] = {padImg, Mat::zeros(padImg.size(), CV_32F)};
    Mat complexImg;
    merge(plane, 2, complexImg);
    dft(complexImg, complexImg);
    //complexImg.copyTo(dstImg);
    Mat magImg = complexImg;
    magImg = magImg(Rect(0, 0, magImg.cols & -2, magImg.rows & -2));
    int cx = magImg.cols / 2;
    int cy = magImg.rows / 2;
    float D0 = 20;
    for (int y = 0; y < magImg.rows; y++) {
        for (int x = 0; x < magImg.cols; x++) {
            double d = sqrt(pow(y - cy, 2) + pow(x -cx, 2));
            if (d <= D0) {

            } else {
                magImg.at<float>(y, x) = 0.0;
            }
        }
    }
    //
    split(magImg, plane);
    magnitude(plane[0], plane[1], plane[0]);
    plane[0] += Scalar::all(1);
    log(plane[0], plane[0]);
    normalize(plane[0], plane[0]);
    /* dft inverse */
    idft(magImg, magImg);
    split(magImg, plane);
    magnitude(plane[0], plane[1], plane[0]);
    normalize(plane[0], plane[0]);
#endif
    padImg.copyTo(dstImg);
    return;
}
/* 采样 */
void Improcess_Sampling(Mat& srcImg, Mat& dstImg)
{

    return;
}
/* 绘制点，直线，几何图形 */

/* 角点检测 */

/* 图像矫正 */

/* Mat和IplImage的像素访问 */

/* 读写xml与yml文件 */
