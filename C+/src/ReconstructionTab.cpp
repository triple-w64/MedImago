#include "../include/ReconstructionTab.h"
#include <QMessageBox>

ReconstructionTab::ReconstructionTab(QWidget *parent, QStatusBar *statusBar)
    : QWidget(parent), statusBar(statusBar), hasLoadedData(false)
{
    // Create main layout
    QHBoxLayout *mainLayout = new QHBoxLayout(this);
    
    // Create left control area
    controlGroup = new QGroupBox("3D Reconstruction Controls", this);
    QVBoxLayout *controlLayout = new QVBoxLayout(controlGroup);
    
    // Add DICOM load button
    loadDicomButton = new QPushButton("Load DICOM Directory", this);
    connect(loadDicomButton, &QPushButton::clicked, this, &ReconstructionTab::loadDicomDirectory);
    controlLayout->addWidget(loadDicomButton);
    
    // Add reconstruction method selection
    QLabel *methodLabel = new QLabel("Reconstruction Method:", this);
    reconstructionMethodComboBox = new QComboBox(this);
    reconstructionMethodComboBox->addItems({"Volume Rendering", "Surface Rendering", "MIP", "MinIP"});
    controlLayout->addWidget(methodLabel);
    controlLayout->addWidget(reconstructionMethodComboBox);
    
    // Add threshold slider
    QLabel *thresholdLabel = new QLabel("Threshold:", this);
    thresholdSlider = new QSlider(Qt::Horizontal, this);
    thresholdSlider->setRange(0, 255);
    thresholdSlider->setValue(128);
    controlLayout->addWidget(thresholdLabel);
    controlLayout->addWidget(thresholdSlider);
    
    // Add start reconstruction button
    startButton = new QPushButton("Start Reconstruction", this);
    startButton->setEnabled(false);
    connect(startButton, &QPushButton::clicked, this, &ReconstructionTab::startReconstruction);
    controlLayout->addWidget(startButton);
    
    // Add save model button
    saveModelButton = new QPushButton("Save Model", this);
    saveModelButton->setEnabled(false);
    connect(saveModelButton, &QPushButton::clicked, this, &ReconstructionTab::saveModel);
    controlLayout->addWidget(saveModelButton);
    
    // Add stretch space
    controlLayout->addStretch();
    
    // Create right display area
    displayGroup = new QGroupBox("3D Display", this);
    QVBoxLayout *displayLayout = new QVBoxLayout(displayGroup);
    
    viewportLabel = new QLabel(this);
    viewportLabel->setMinimumSize(600, 600);
    viewportLabel->setAlignment(Qt::AlignCenter);
    viewportLabel->setText("3D reconstruction result will be displayed after loading DICOM data");
    
    displayLayout->addWidget(viewportLabel);
    
    // Add to main layout
    mainLayout->addWidget(controlGroup, 0);
    mainLayout->addWidget(displayGroup, 1);
}

ReconstructionTab::~ReconstructionTab()
{
}

void ReconstructionTab::loadDicomDirectory()
{
    QString dirPath = QFileDialog::getExistingDirectory(
        this,
        "Select DICOM Directory"
    );
    
    if (dirPath.isEmpty()) {
        return;
    }
    
    // In a real application, DICOM data would be loaded here
    // Simplified for demonstration
    dicomDirectory = dirPath;
    hasLoadedData = true;
    startButton->setEnabled(true);
    
    statusBar->showMessage("DICOM directory loaded: " + dirPath);
    viewportLabel->setText("DICOM data loaded, click \"Start Reconstruction\" button to proceed");
}

void ReconstructionTab::startReconstruction()
{
    if (!hasLoadedData) {
        QMessageBox::warning(this, "Warning", "Please load DICOM data first");
        return;
    }
    
    QString method = reconstructionMethodComboBox->currentText();
    int threshold = thresholdSlider->value();
    
    statusBar->showMessage("Performing 3D reconstruction...");
    
    // In a real application, actual 3D reconstruction would happen here
    // Simplified for demonstration
    
    // Show a message indicating reconstruction completed
    QMessageBox::information(this, "3D Reconstruction", QString("Reconstruction completed using %1 method, threshold: %2").arg(method).arg(threshold));
    saveModelButton->setEnabled(true);
    statusBar->showMessage("3D reconstruction completed");
    
    // Update display area
    viewportLabel->setText(QString("Reconstruction completed using %1 method\nThreshold: %2").arg(method).arg(threshold));
}

void ReconstructionTab::saveModel()
{
    if (!hasLoadedData) {
        QMessageBox::warning(this, "Warning", "No model to save");
        return;
    }
    
    QString filePath = QFileDialog::getSaveFileName(
        this,
        "Save 3D Model",
        QString(),
        "STL Files (*.stl);;OBJ Files (*.obj)"
    );
    
    if (filePath.isEmpty()) {
        return;
    }
    
    statusBar->showMessage("Saving model...");
    
    // In a real application, the 3D model would be saved here
    // Simplified for demonstration
    QMessageBox::information(this, "Save Model", "Model saved to: " + filePath);
    statusBar->showMessage("Model saved to: " + filePath);
}
