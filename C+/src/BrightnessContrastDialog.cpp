#include "../include/BrightnessContrastDialog.h"
#include "../include/SuperResolutionTab.h"

BrightnessContrastDialog::BrightnessContrastDialog(SuperResolutionTab *parent, int brightness, int contrast)
    : QDialog(parent), parent(parent)
{
    setWindowTitle("亮度/对比度调整");
    setMinimumWidth(400);
    
    // 保存原始图像
    if (parent) {
        // 假设parent有一个获取当前图像的方法
        // 此处需要根据SuperResolutionTab类的实际设计进行调整
        // originalImage = parent->getCurrentImage().clone();
    }
    
    QGridLayout *layout = new QGridLayout(this);
    
    // 亮度滑块
    layout->addWidget(new QLabel("亮度:"), 0, 0);
    brightnessSlider = new QSlider(Qt::Horizontal);
    brightnessSlider->setMinimum(-100);
    brightnessSlider->setMaximum(100);
    brightnessSlider->setValue(brightness);
    layout->addWidget(brightnessSlider, 0, 1);
    brightnessValue = new QLabel(QString::number(brightness));
    layout->addWidget(brightnessValue, 0, 2);
    
    // 对比度滑块
    layout->addWidget(new QLabel("对比度:"), 1, 0);
    contrastSlider = new QSlider(Qt::Horizontal);
    contrastSlider->setMinimum(-100);
    contrastSlider->setMaximum(100);
    contrastSlider->setValue(contrast);
    layout->addWidget(contrastSlider, 1, 1);
    contrastValue = new QLabel(QString::number(contrast));
    layout->addWidget(contrastValue, 1, 2);
    
    // 添加按钮
    buttonBox = new QDialogButtonBox();
    applyButton = buttonBox->addButton("应用", QDialogButtonBox::ApplyRole);
    buttonBox->addButton(QDialogButtonBox::Ok);
    buttonBox->addButton(QDialogButtonBox::Cancel);
    layout->addWidget(buttonBox, 2, 0, 1, 3);
    
    // 连接信号
    connect(brightnessSlider, &QSlider::valueChanged, this, &BrightnessContrastDialog::updateBrightnessValue);
    connect(brightnessSlider, &QSlider::valueChanged, this, &BrightnessContrastDialog::previewChanges);
    connect(contrastSlider, &QSlider::valueChanged, this, &BrightnessContrastDialog::updateContrastValue);
    connect(contrastSlider, &QSlider::valueChanged, this, &BrightnessContrastDialog::previewChanges);
    connect(applyButton, &QPushButton::clicked, this, &BrightnessContrastDialog::applyChanges);
    connect(buttonBox, &QDialogButtonBox::accepted, this, &BrightnessContrastDialog::accept);
    connect(buttonBox, &QDialogButtonBox::rejected, this, &BrightnessContrastDialog::reject);
    
    // 初始预览
    previewChanges();
}

void BrightnessContrastDialog::updateBrightnessValue(int value)
{
    brightnessValue->setText(QString::number(value));
}

void BrightnessContrastDialog::updateContrastValue(int value)
{
    contrastValue->setText(QString::number(value));
}

std::pair<int, int> BrightnessContrastDialog::getValues() const
{
    return std::make_pair(brightnessSlider->value(), contrastSlider->value());
}

void BrightnessContrastDialog::previewChanges()
{
    if (originalImage.empty()) {
        return;
    }
    
    try {
        int brightness = brightnessSlider->value();
        int contrast = contrastSlider->value();
        
        // 复制原图以保留原始数据
        cv::Mat adjustedImg = originalImage.clone();
        
        // 计算亮度和对比度参数
        double alpha = (contrast + 100) / 100.0;  // 对比度因子
        int beta = brightness;  // 亮度因子
        
        // 应用变换: g(x) = alpha * f(x) + beta
        cv::convertScaleAbs(adjustedImg, adjustedImg, alpha, beta);
        
        // 显示结果 - 需要调用父类的显示方法
        // parent->displayImage(adjustedImg);
        
    } catch (const std::exception &e) {
        // 错误处理
    }
}

void BrightnessContrastDialog::applyChanges()
{
    if (originalImage.empty()) {
        return;
    }
    
    try {
        int brightness = brightnessSlider->value();
        int contrast = contrastSlider->value();
        
        // 复制原图以保留原始数据
        cv::Mat adjustedImg = originalImage.clone();
        
        // 计算亮度和对比度参数
        double alpha = (contrast + 100) / 100.0;  // 对比度因子
        int beta = brightness;  // 亮度因子
        
        // 应用变换: g(x) = alpha * f(x) + beta
        cv::convertScaleAbs(adjustedImg, adjustedImg, alpha, beta);
        
        // 应用更改到父类
        // parent->setImage(adjustedImg);
        // parent->updateStatusBar(QString("亮度对比度调整已应用: 亮度=%1, 对比度=%2").arg(brightness).arg(contrast));
        
        // 更新对话框中的原始图像
        originalImage = adjustedImg.clone();  // 更新对话框中的原始图像
        
    } catch (const std::exception &e) {
        // 错误处理
    }
} 