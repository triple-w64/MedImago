#include "../include/SuperResolutionTab.h"
#include <QMessageBox>

SuperResolutionTab::SuperResolutionTab(QWidget *parent, QStatusBar *statusBar)
    : QWidget(parent), statusBar(statusBar)
{
    // Create main layout
    QHBoxLayout *mainLayout = new QHBoxLayout(this);
    
    // Create left side original image area
    originalImageGroup = new QGroupBox("Original Image", this);
    QVBoxLayout *originalLayout = new QVBoxLayout(originalImageGroup);
    
    originalImageLabel = new QLabel(this);
    originalImageLabel->setMinimumSize(400, 400);
    originalImageLabel->setAlignment(Qt::AlignCenter);
    originalImageLabel->setScaledContents(true);
    originalImageLabel->setText("Please load an image");
    
    loadImageButton = new QPushButton("Load Image", this);
    connect(loadImageButton, &QPushButton::clicked, this, &SuperResolutionTab::loadImage);
    
    originalLayout->addWidget(originalImageLabel);
    originalLayout->addWidget(loadImageButton);
    
    // Create middle control area
    srControlGroup = new QGroupBox("Super Resolution Controls", this);
    QVBoxLayout *controlLayout = new QVBoxLayout(srControlGroup);
    
    QLabel *algorithmLabel = new QLabel("Algorithm:", this);
    algorithmComboBox = new QComboBox(this);
    algorithmComboBox->addItems({"Bicubic", "EDSR", "FSRCNN", "ESPCN"});
    
    QLabel *scaleFactorLabel = new QLabel("Scale Factor:", this);
    scaleFactorSpinBox = new QSpinBox(this);
    scaleFactorSpinBox->setRange(2, 4);
    scaleFactorSpinBox->setValue(2);
    
    applyButton = new QPushButton("Apply Super Resolution", this);
    connect(applyButton, &QPushButton::clicked, this, &SuperResolutionTab::applySuperResolution);
    
    controlLayout->addWidget(algorithmLabel);
    controlLayout->addWidget(algorithmComboBox);
    controlLayout->addWidget(scaleFactorLabel);
    controlLayout->addWidget(scaleFactorSpinBox);
    controlLayout->addWidget(applyButton);
    controlLayout->addStretch();
    
    // Create right side result area
    resultGroup = new QGroupBox("Super Resolution Result", this);
    QVBoxLayout *resultLayout = new QVBoxLayout(resultGroup);
    
    resultImageLabel = new QLabel(this);
    resultImageLabel->setMinimumSize(400, 400);
    resultImageLabel->setAlignment(Qt::AlignCenter);
    resultImageLabel->setScaledContents(true);
    resultImageLabel->setText("Waiting for processing");
    
    saveResultButton = new QPushButton("Save Result", this);
    saveResultButton->setEnabled(false);
    connect(saveResultButton, &QPushButton::clicked, this, &SuperResolutionTab::saveResult);
    
    resultLayout->addWidget(resultImageLabel);
    resultLayout->addWidget(saveResultButton);
    
    // Add to main layout
    mainLayout->addWidget(originalImageGroup, 1);
    mainLayout->addWidget(srControlGroup, 0);
    mainLayout->addWidget(resultGroup, 1);
}

SuperResolutionTab::~SuperResolutionTab()
{
}

void SuperResolutionTab::loadImage()
{
    QString filePath = QFileDialog::getOpenFileName(
        this,
        "Open Image File",
        QString(),
        "Image Files (*.png *.jpg *.bmp);;All Files (*.*)"
    );
    
    if (filePath.isEmpty()) {
        return;
    }
    
    if (originalImage.load(filePath)) {
        originalImageLabel->setPixmap(QPixmap::fromImage(originalImage));
        statusBar->showMessage("Image loaded: " + filePath);
    } else {
        QMessageBox::warning(this, "Error", "Failed to load image: " + filePath);
        statusBar->showMessage("Image loading failed", 3000);
    }
}

void SuperResolutionTab::applySuperResolution()
{
    if (originalImage.isNull()) {
        QMessageBox::warning(this, "Warning", "Please load an original image first");
        return;
    }
    
    // Get user-selected parameters
    QString algorithm = algorithmComboBox->currentText();
    int scaleFactor = scaleFactorSpinBox->value();
    
    statusBar->showMessage("Processing image...");
    
    // Simple scaling of original image for demonstration
    resultImage = originalImage.scaled(
        originalImage.width() * scaleFactor,
        originalImage.height() * scaleFactor,
        Qt::IgnoreAspectRatio,
        Qt::SmoothTransformation
    );
    
    // Display result
    resultImageLabel->setPixmap(QPixmap::fromImage(resultImage));
    saveResultButton->setEnabled(true);
    
    statusBar->showMessage("Super resolution processing complete, scale factor: " + QString::number(scaleFactor));
}

void SuperResolutionTab::saveResult()
{
    if (resultImage.isNull()) {
        QMessageBox::warning(this, "Warning", "No result to save");
        return;
    }
    
    QString filePath = QFileDialog::getSaveFileName(
        this,
        "Save Image File",
        QString(),
        "PNG Image (*.png);;JPEG Image (*.jpg);;BMP Image (*.bmp)"
    );
    
    if (filePath.isEmpty()) {
        return;
    }
    
    if (resultImage.save(filePath)) {
        statusBar->showMessage("Result saved to: " + filePath);
    } else {
        QMessageBox::warning(this, "Error", "Failed to save image: " + filePath);
        statusBar->showMessage("Save failed", 3000);
    }
}
