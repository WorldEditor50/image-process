QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++11

# The following define makes your compiler emit warnings if you use
# any Qt feature that has been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    improcess.cpp \
    main.cpp \
    mainwindow.cpp

HEADERS += \
    improcess.h \
    mainwindow.h

FORMS += \
    mainwindow.ui

msvc{
    INCLUDEPATH += D:/opencv-4.1.2-MSVC-X64/include \
                   D:/opencv-4.1.2-MSVC-X64/include/opencv2
    LIBS += -LD:/opencv-4.1.2-MSVC-X64/x64/vc16/lib/ -lopencv_stitching412
    LIBS += -LD:/opencv-4.1.2-MSVC-X64/x64/vc16/lib/ -lopencv_video412
    #LIBS += -LD:/opencv-4.1.2-MSVC-X64/x64/vc16/lib/ -lopencv_videoio_ffmpeg412_64
    LIBS += -LD:/opencv-4.1.2-MSVC-X64/x64/vc16/lib/ -lopencv_videoio412
    LIBS += -LD:/opencv-4.1.2-MSVC-X64/x64/vc16/lib/ -lopencv_calib3d412
    LIBS += -LD:/opencv-4.1.2-MSVC-X64/x64/vc16/lib/ -lopencv_core412
    LIBS += -LD:/opencv-4.1.2-MSVC-X64/x64/vc16/lib/ -lopencv_dnn412
    LIBS += -LD:/opencv-4.1.2-MSVC-X64/x64/vc16/lib/ -lopencv_features2d412
    LIBS += -LD:/opencv-4.1.2-MSVC-X64/x64/vc16/lib/ -lopencv_flann412
    LIBS += -LD:/opencv-4.1.2-MSVC-X64/x64/vc16/lib/ -lopencv_gapi412
    LIBS += -LD:/opencv-4.1.2-MSVC-X64/x64/vc16/lib/ -lopencv_highgui412
    LIBS += -LD:/opencv-4.1.2-MSVC-X64/x64/vc16/lib/ -lopencv_imgcodecs412
    LIBS += -LD:/opencv-4.1.2-MSVC-X64/x64/vc16/lib/ -lopencv_imgproc412
    LIBS += -LD:/opencv-4.1.2-MSVC-X64/x64/vc16/lib/ -lopencv_ml412
    LIBS += -LD:/opencv-4.1.2-MSVC-X64/x64/vc16/lib/ -lopencv_objdetect412
    LIBS += -LD:/opencv-4.1.2-MSVC-X64/x64/vc16/lib/ -lopencv_photo412
}
# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target
