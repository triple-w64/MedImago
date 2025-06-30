#include "../include/DicomLoaderThread.h"
#include <vtkSmartVolumeMapper.h>
#include <QDebug>

DicomLoaderThread::DicomLoaderThread(const QString &directory, QObject *parent)
    : QThread(parent), directory(directory)
{
}

void DicomLoaderThread::run()
{
    try {
        // 创建DICOM读取器
        vtkSmartPointer<vtkDICOMImageReader> reader = vtkSmartPointer<vtkDICOMImageReader>::New();
        reader->SetDirectoryName(directory.toStdString().c_str());
        reader->Update();
        
        // 检查是否成功读取
        if (reader->GetOutput()->GetNumberOfPoints() <= 0) {
            emit loadingError("无法从DICOM目录读取有效数据");
            return;
        }
        
        // 创建体积映射器
        vtkSmartPointer<vtkSmartVolumeMapper> volumeMapper = vtkSmartPointer<vtkSmartVolumeMapper>::New();
        volumeMapper->SetInputConnection(reader->GetOutputPort());
        
        // 发送完成信号
        emit loadingFinished(volumeMapper, directory);
        
    } catch (const std::exception &e) {
        emit loadingError(QString("加载DICOM出错: %1").arg(e.what()));
    } catch (...) {
        emit loadingError("加载DICOM时出现未知错误");
    }
} 