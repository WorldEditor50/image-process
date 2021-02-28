#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QLabel>
#include <QPixmap>
#include <QThread>
#include <QPushButton>
#include <QFileDialog>
#include <QImage>
#include <QPixmap>
#include <QPicture>
#include "improcess.h"

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE
class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();
signals:
    void sigProcessImage(const QString &method, const QString &srcPath);
public slots:

private:
    Ui::MainWindow *ui;
    Improcess *process;
    QThread worker;
    QString filePath;
};
#endif // MAINWINDOW_H
