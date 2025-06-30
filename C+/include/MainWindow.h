#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QTabWidget>
#include <QMenuBar>
#include <QToolBar>
#include <QStatusBar>
#include <QAction>
#include <QSettings>
#include <QCloseEvent>

#include "SuperResolutionTab.h"
#include "ReconstructionTab.h"

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

protected:
    void closeEvent(QCloseEvent *event) override;

private slots:
    void openFile();
    void about();
    void onTabChanged(int index);

private:
    void createActions();
    void createMenus();
    void createToolBar();
    void loadSettings();
    void saveSettings();
    
    QTabWidget *tabWidget;
    SuperResolutionTab *srTab;
    ReconstructionTab *reconsTab;
    
    QAction *openAction;
    QAction *exitAction;
    QAction *aboutAction;
    QStatusBar *statusBar;
    
    QSettings settings;
};

#endif // MAINWINDOW_H 