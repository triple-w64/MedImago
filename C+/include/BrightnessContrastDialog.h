#ifndef BRIGHTNESSCONTRASTDIALOG_H
#define BRIGHTNESSCONTRASTDIALOG_H

#include <QDialog>
#include <QSlider>
#include <QLabel>
#include <QPushButton>
#include <QGridLayout>
#include <QDialogButtonBox>
#include <opencv2/opencv.hpp>

class SuperResolutionTab;

class BrightnessContrastDialog : public QDialog
{
    Q_OBJECT
public:
    explicit BrightnessContrastDialog(SuperResolutionTab *parent = nullptr, int brightness = 0, int contrast = 0);
    std::pair<int, int> getValues() const;

private slots:
    void updateBrightnessValue(int value);
    void updateContrastValue(int value);
    void previewChanges();
    void applyChanges();

private:
    SuperResolutionTab *parent;
    cv::Mat originalImage;
    
    QSlider *brightnessSlider;
    QSlider *contrastSlider;
    QLabel *brightnessValue;
    QLabel *contrastValue;
    QDialogButtonBox *buttonBox;
    QPushButton *applyButton;
};

#endif // BRIGHTNESSCONTRASTDIALOG_H 