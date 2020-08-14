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
#INCLUDEPATH += D:/opencv-4.1.2-x86/include \
#               D:/opencv-4.1.2-x86/include/opencv \
#               D:/opencv-4.1.2-x86/include/opencv2
#LIBS += D:/opencv-4.1.2-x86/x86/mingw/lib/libopencv_*

INCLUDEPATH += /usr/local/include/opencv4 \
               /usr/local/include/opencv4/opencv \
               /usr/local/include/opencv4/opencv2
LIBS += /usr/local/lib/libopencv_*
# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target