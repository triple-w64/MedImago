#include "../include/MainWindow.h"
#include "../include/SuperResolutionTab.h"
#include "../include/ReconstructionTab.h"
#include <QMessageBox>
#include <QFileDialog>
#include <QStandardPaths>
#include <QDebug>
#include <QApplication>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent),
      settings("MedImago", "MedImago")
{
    // Set window title and size
    setWindowTitle("MedImago");
    resize(1024, 768);
    
    // Create status bar
    statusBar = new QStatusBar(this);
    setStatusBar(statusBar);
    statusBar->showMessage("Ready");
    
    // Create tab widget
    tabWidget = new QTabWidget(this);
    setCentralWidget(tabWidget);
    
    // Add feature tabs
    srTab = new SuperResolutionTab(this, statusBar);
    reconsTab = new ReconstructionTab(this, statusBar);
    
    tabWidget->addTab(srTab, "Super Resolution");
    tabWidget->addTab(reconsTab, "3D Reconstruction");
    
    // Connect tab change signal
    connect(tabWidget, &QTabWidget::currentChanged, this, &MainWindow::onTabChanged);
    
    // Create menus and toolbar
    createActions();
    createMenus();
    createToolBar();
    
    // Load saved settings
    loadSettings();
    
    // Set initial state
    onTabChanged(0);  // Default to first tab
}

MainWindow::~MainWindow()
{
    saveSettings();
}

void MainWindow::closeEvent(QCloseEvent *event)
{
    saveSettings();
    event->accept();
}

void MainWindow::createActions()
{
    // File menu actions
    openAction = new QAction("Open", this);
    openAction->setShortcut(QKeySequence::Open);
    connect(openAction, &QAction::triggered, this, &MainWindow::openFile);
    
    exitAction = new QAction("Exit", this);
    exitAction->setShortcut(QKeySequence::Quit);
    connect(exitAction, &QAction::triggered, qApp, &QApplication::quit);
    
    // Help menu actions
    aboutAction = new QAction("About", this);
    connect(aboutAction, &QAction::triggered, this, &MainWindow::about);
}

void MainWindow::createMenus()
{
    QMenuBar *menuBar = this->menuBar();
    
    // File menu
    QMenu *fileMenu = menuBar->addMenu("File");
    fileMenu->addAction(openAction);
    fileMenu->addSeparator();
    fileMenu->addAction(exitAction);
    
    // Help menu
    QMenu *helpMenu = menuBar->addMenu("Help");
    helpMenu->addAction(aboutAction);
}

void MainWindow::createToolBar()
{
    QToolBar *toolBar = addToolBar("Main Toolbar");
    toolBar->addAction(openAction);
}

void MainWindow::loadSettings()
{
    settings.beginGroup("MainWindow");
    resize(settings.value("size", QSize(1280, 800)).toSize());
    move(settings.value("pos", QPoint(100, 100)).toPoint());
    tabWidget->setCurrentIndex(settings.value("currentTab", 0).toInt());
    settings.endGroup();
}

void MainWindow::saveSettings()
{
    settings.beginGroup("MainWindow");
    settings.setValue("size", size());
    settings.setValue("pos", pos());
    settings.setValue("currentTab", tabWidget->currentIndex());
    settings.endGroup();
}

void MainWindow::openFile()
{
    // Choose file type based on active tab
    int currentIndex = tabWidget->currentIndex();
    
    if (currentIndex == 0) {
        // Super resolution tab
        QString filePath = QFileDialog::getOpenFileName(
            this,
            "Open Image File",
            QString(),
            "Image Files (*.png *.jpg *.bmp);;All Files (*.*)"
        );
        
        if (!filePath.isEmpty()) {
            statusBar->showMessage("Opened image file: " + filePath);
        }
    } else if (currentIndex == 1) {
        // 3D reconstruction tab
        QString dirPath = QFileDialog::getExistingDirectory(
            this,
            "Select DICOM Directory"
        );
        
        if (!dirPath.isEmpty()) {
            statusBar->showMessage("Selected DICOM directory: " + dirPath);
        }
    }
}

void MainWindow::about()
{
    QMessageBox::about(this, "About MedImago",
        "<h2>MedImago</h2>"
        "<p>Version 1.0.0</p>"
        "<p>Medical imaging platform with super-resolution and 3D reconstruction features.</p>"
        "<p>Copyright &copy; 2023</p>");
}

void MainWindow::onTabChanged(int index)
{
    // Update status bar based on current tab
    if (index == 0) {
        statusBar->showMessage("Super Resolution Mode");
    } else if (index == 1) {
        statusBar->showMessage("3D Reconstruction Mode");
    }
} 