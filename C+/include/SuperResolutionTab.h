#ifndef SUPERRESOLUTIONTAB_H
#define SUPERRESOLUTIONTAB_H

#include <QWidget>
#include <QLabel>
#include <QPushButton>
#include <QComboBox>
#include <QSpinBox>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QFileDialog>
#include <QStatusBar>
#include <QImageReader>
#include <QImage>
#include <QPixmap>

class SuperResolutionTab : public QWidget
{
    Q_OBJECT

public:
    explicit SuperResolutionTab(QWidget *parent = nullptr, QStatusBar *statusBar = nullptr);
    ~SuperResolutionTab();

private slots:
    void loadImage();
    void applySuperResolution();
    void saveResult();

private:
    QStatusBar *statusBar;
    
    // 原始图像显示区域
    QGroupBox *originalImageGroup;
    QLabel *originalImageLabel;
    QPushButton *loadImageButton;
    QImage originalImage;
    
    // 超分辨率控制区域
    QGroupBox *srControlGroup;
    QComboBox *algorithmComboBox;
    QSpinBox *scaleFactorSpinBox;
    QPushButton *applyButton;
    
    // 结果显示区域
    QGroupBox *resultGroup;
    QLabel *resultImageLabel;
    QPushButton *saveResultButton;
    QImage resultImage;
};

#endif // SUPERRESOLUTIONTAB_H 