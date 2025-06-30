#ifndef WINDOWLEVELDIALOG_H
#define WINDOWLEVELDIALOG_H

#include <QDialog>
#include <QSlider>
#include <QLabel>
#include <QPushButton>
#include <QGridLayout>
#include <QDialogButtonBox>
#include <opencv2/opencv.hpp>

class SuperResolutionTab;

class WindowLevelDialog : public QDialog
{
    Q_OBJECT
public:
    explicit WindowLevelDialog(SuperResolutionTab *parent = nullptr, int windowLevel = 128, int windowWidth = 255);
    std::pair<int, int> getValues() const;

private slots:
    void updateLevelValue(int value);
    void updateWidthValue(int value);
    void previewChanges();
    void applyChanges();

private:
    SuperResolutionTab *parent;
    cv::Mat originalImage;
    
    QSlider *levelSlider;
    QSlider *widthSlider;
    QLabel *levelValue;
    QLabel *widthValue;
    QDialogButtonBox *buttonBox;
    QPushButton *applyButton;
};

#endif // WINDOWLEVELDIALOG_H 