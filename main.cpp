#include "mainwindow.h"
#include "improcess.h"
#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    MainWindow w;
    Improcess_Test("/home/eigen/Pictures/1234.jpg", "/home/eigen/Pictures/tmp.jpg");
    w.updateImage();
    w.show();
    return a.exec();
}
