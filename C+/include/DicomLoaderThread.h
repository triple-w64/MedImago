#ifndef DICOMLOADERTHREAD_H
#define DICOMLOADERTHREAD_H

#include <QThread>
#include <QString>
#include <vtkSmartPointer.h>
#include <vtkDICOMImageReader.h>
#include <vtkSmartVolumeMapper.h>

class DicomLoaderThread : public QThread
{
    Q_OBJECT
public:
    explicit DicomLoaderThread(const QString &directory, QObject *parent = nullptr);
    void run() override;

signals:
    void loadingFinished(vtkSmartVolumeMapper *mapper, QString directory);
    void loadingError(QString errorMessage);

private:
    QString directory;
};

#endif // DICOMLOADERTHREAD_H 