#ifndef RECONSTRUCTIONTAB_H
#define RECONSTRUCTIONTAB_H

#include <QWidget>
#include <QLabel>
#include <QPushButton>
#include <QComboBox>
#include <QSlider>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QFileDialog>
#include <QStatusBar>

class ReconstructionTab : public QWidget
{
    Q_OBJECT

public:
    explicit ReconstructionTab(QWidget *parent = nullptr, QStatusBar *statusBar = nullptr);
    ~ReconstructionTab();

private slots:
    void loadDicomDirectory();
    void startReconstruction();
    void saveModel();

private:
    QStatusBar *statusBar;
    
    // 控制区域
    QGroupBox *controlGroup;
    QPushButton *loadDicomButton;
    QComboBox *reconstructionMethodComboBox;
    QSlider *thresholdSlider;
    QPushButton *startButton;
    QPushButton *saveModelButton;
    
    // 显示区域
    QGroupBox *displayGroup;
    QLabel *viewportLabel;
    
    // 状态变量
    QString dicomDirectory;
    bool hasLoadedData;
};

#endif // RECONSTRUCTIONTAB_H 