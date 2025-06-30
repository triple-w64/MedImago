#include <QApplication>
#include "../include/MainWindow.h"

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    
    // 设置应用程序信息
    QCoreApplication::setOrganizationName("MedImago");
    QCoreApplication::setApplicationName("MedImago");
    QCoreApplication::setApplicationVersion("1.0.0");
    
    // 创建并显示主窗口
    MainWindow mainWindow;
    mainWindow.show();
    
    return app.exec();
} 