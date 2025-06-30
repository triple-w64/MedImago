#include "../include/WindowLevelDialog.h"
#include "../include/SuperResolutionTab.h"

WindowLevelDialog::WindowLevelDialog(SuperResolutionTab *parent, int windowLevel, int windowWidth)
    : QDialog(parent), parent(parent)
{
    setWindowTitle("窗位窗宽调整");
    setMinimumWidth(400);
    
    // 保存原始图像
    if (parent) {
        // 假设parent有一个获取当前图像的方法
        // 此处需要根据SuperResolutionTab类的实际设计进行调整
        // originalImage = parent->getCurrentImage().clone();
    }
    
    QGridLayout *layout = new QGridLayout(this);
    
    // 窗位滑块
    layout->addWidget(new QLabel("窗位:"), 0, 0);
    levelSlider = new QSlider(Qt::Horizontal);
    levelSlider->setMinimum(0);
    levelSlider->setMaximum(255);
    levelSlider->setValue(windowLevel);
    layout->addWidget(levelSlider, 0, 1);
    levelValue = new QLabel(QString::number(windowLevel));
    layout->addWidget(levelValue, 0, 2);
    
    // 窗宽滑块
    layout->addWidget(new QLabel("窗宽:"), 1, 0);
    widthSlider = new QSlider(Qt::Horizontal);
    widthSlider->setMinimum(1);
    widthSlider->setMaximum(255);
    widthSlider->setValue(windowWidth);
    layout->addWidget(widthSlider, 1, 1);
    widthValue = new QLabel(QString::number(windowWidth));
    layout->addWidget(widthValue, 1, 2);
    
    // 添加按钮
    buttonBox = new QDialogButtonBox();
    applyButton = buttonBox->addButton("应用", QDialogButtonBox::ApplyRole);
    buttonBox->addButton(QDialogButtonBox::Ok);
    buttonBox->addButton(QDialogButtonBox::Cancel);
    layout->addWidget(buttonBox, 2, 0, 1, 3);
    
    // 连接信号
    connect(levelSlider, &QSlider::valueChanged, this, &WindowLevelDialog::updateLevelValue);
    connect(levelSlider, &QSlider::valueChanged, this, &WindowLevelDialog::previewChanges);
    connect(widthSlider, &QSlider::valueChanged, this, &WindowLevelDialog::updateWidthValue);
    connect(widthSlider, &QSlider::valueChanged, this, &WindowLevelDialog::previewChanges);
    connect(applyButton, &QPushButton::clicked, this, &WindowLevelDialog::applyChanges);
    connect(buttonBox, &QDialogButtonBox::accepted, this, &WindowLevelDialog::accept);
    connect(buttonBox, &QDialogButtonBox::rejected, this, &WindowLevelDialog::reject);
    
    // 初始预览
    previewChanges();
}

void WindowLevelDialog::updateLevelValue(int value)
{
    levelValue->setText(QString::number(value));
}

void WindowLevelDialog::updateWidthValue(int value)
{
    widthValue->setText(QString::number(value));
}

std::pair<int, int> WindowLevelDialog::getValues() const
{
    return std::make_pair(levelSlider->value(), widthSlider->value());
}

void WindowLevelDialog::previewChanges()
{
    if (originalImage.empty()) {
        return;
    }
    
    try {
        int level = levelSlider->value();
        int width = widthSlider->value();
        
        // 计算上下阈值
        int minVal = std::max(0, level - width / 2);
        int maxVal = std::min(255, level + width / 2);
        
        // 复制原图以保留原始数据
        cv::Mat adjustedImg = originalImage.clone();
        
        // 如果是彩色图像，分别处理每个通道
        if (adjustedImg.channels() == 3) {
            std::vector<cv::Mat> channels;
            cv::split(adjustedImg, channels);
            
            for (int i = 0; i < 3; i++) {  // 处理BGR三个通道
                // 限制在阈值范围内
                cv::threshold(channels[i], channels[i], maxVal, maxVal, cv::THRESH_TRUNC);
                cv::threshold(channels[i], channels[i], minVal, minVal, cv::THRESH_TOZERO);
                
                // 重新映射到0-255
                if (maxVal > minVal) {  // 避免除零错误
                    channels[i].convertTo(channels[i], CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));
                }
            }
            
            cv::merge(channels, adjustedImg);
        } else {
            // 灰度图像处理
            cv::threshold(adjustedImg, adjustedImg, maxVal, maxVal, cv::THRESH_TRUNC);
            cv::threshold(adjustedImg, adjustedImg, minVal, minVal, cv::THRESH_TOZERO);
            
            if (maxVal > minVal) {  // 避免除零错误
                adjustedImg.convertTo(adjustedImg, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));
            }
        }
        
        // 显示结果 - 需要调用父类的显示方法
        // parent->displayImage(adjustedImg);
        
    } catch (const std::exception &e) {
        // 错误处理
    }
}

void WindowLevelDialog::applyChanges()
{
    if (originalImage.empty()) {
        return;
    }
    
    try {
        int level = levelSlider->value();
        int width = widthSlider->value();
        
        // 计算调整后的图像（与previewChanges相同的处理）
        
        // 应用更改到父类
        // parent->setImage(adjustedImg);
        // parent->updateStatusBar(QString("窗位窗宽调整已应用: 窗位=%1, 窗宽=%2").arg(level).arg(width));
        
        // 更新对话框中的原始图像
        // originalImage = adjustedImg.clone();
        
    } catch (const std::exception &e) {
        // 错误处理
    }
} 