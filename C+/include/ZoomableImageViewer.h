#ifndef ZOOMABLEIMAGEVIEWER_H
#define ZOOMABLEIMAGEVIEWER_H

#include <QGraphicsView>
#include <QGraphicsScene>
#include <QGraphicsPixmapItem>
#include <QWheelEvent>
#include <QMouseEvent>
#include <QPainter>
#include <QKeyEvent>

class ZoomableImageViewer : public QGraphicsView
{
    Q_OBJECT
public:
    explicit ZoomableImageViewer(QWidget *parent = nullptr);
    void setImage(const QPixmap &pixmap);
    void setTool(const QString &toolName);
    void fitInView();

protected:
    void wheelEvent(QWheelEvent *event) override;
    void mousePressEvent(QMouseEvent *event) override;
    void mouseMoveEvent(QMouseEvent *event) override;
    void mouseReleaseEvent(QMouseEvent *event) override;
    void resizeEvent(QResizeEvent *event) override;

private:
    QGraphicsScene *scene;
    QGraphicsPixmapItem *pixmapItem;
    double zoomLevel;
    double minZoom;
    double maxZoom;
    bool shouldFitInView;
    bool isDragging;
    QPoint dragStartPos;
    QString currentTool;
};

#endif // ZOOMABLEIMAGEVIEWER_H 