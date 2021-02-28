#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    setFixedSize(1060, 700);
    /* method */
    QStringList methodList;
    methodList<<"threshold"
             <<"adaptiveThreshold"
             <<"boxFilter"
             <<"blur"
             <<"gaussianBlur"
             <<"erode"
             <<"dilate"
             <<"morphOpen"
             <<"morphClose"
             <<"morphTopHat"
             <<"morphBlackHat"
             <<"morphGrident"
             <<"canny"
             <<"sobel"
             <<"laplace"
             <<"houghLine"
             <<"houghLineP"
             <<"houghCircle"
             <<"remap"
             <<"affineByRotate"
             <<"affineByPoint"
             <<"matchTemplate"
             <<"findMaxContours"
             <<"findContoursByLR"
             <<"findContoursByMeanZero";
    ui->methodBox->addItems(methodList);
    /* open */
    connect(ui->openBtn, &QPushButton::clicked, this, [=](){
        filePath = QFileDialog::getOpenFileName(this, "Open Image", "./");
        QPixmap pix(500, 500);
        pix.load(filePath);
        pix.scaled(500, 500, Qt::KeepAspectRatio);
        ui->srcImg->setPixmap(pix);
    });

    /* process */
    process = new Improcess;
    process->moveToThread(&worker);
    connect(&worker, &QThread::finished, process, &Improcess::deleteLater, Qt::QueuedConnection);
    connect(process, &Improcess::processFinished, this, [=](const QImage img){
        QPixmap pix(500, 500);
        pix = QPixmap::fromImage(img);
        pix.scaled(500, 500, Qt::KeepAspectRatio);
        ui->dstImg->setPixmap(pix);
    }, Qt::QueuedConnection);
    /* convert */
    connect(this, &MainWindow::sigProcessImage, process, &Improcess::process, Qt::QueuedConnection);
    connect(ui->convertBtn, &QPushButton::clicked, this, [=]{
        if (filePath.isEmpty()) {
            return;
        }
        QString method = ui->methodBox->currentText();
        if (method.isEmpty()) {
            return;
        }
        emit sigProcessImage(method, filePath);
    });
    worker.start();
}

MainWindow::~MainWindow()
{
    worker.quit();
    worker.wait();
    process->deleteLater();
    delete ui;
}

