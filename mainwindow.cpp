#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    this->setGeometry(100, 100, 1800, 800);
    /* create label */
    m_srcLabel = new QLabel(this);
    m_dstLabel = new QLabel(this);
    /* add title */
    m_srcLabel->setText(QString("souce image"));
    m_dstLabel->setText(QString("destinating image"));
    /* set position */
    m_srcLabel->setGeometry(0, 0, this->width() / 2, this->height());
    m_dstLabel->setGeometry(this->width() / 2, 0, this->width() / 2, this->height());
    /* set picture */
    QPixmap srcPixmap;
    srcPixmap.load("/home/eigen/Pictures/1234.jpg");
    m_srcLabel->setPixmap(srcPixmap);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::updateImage()
{
    QPixmap dstPixmap;
    dstPixmap.load("/home/eigen/Pictures/tmp.jpg");
    m_dstLabel->setPixmap(dstPixmap);
    return;
}

