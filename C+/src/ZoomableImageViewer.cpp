#include "../include/ZoomableImageViewer.h"
#include <QScrollBar>
#include <QApplication>

ZoomableImageViewer::ZoomableImageViewer(QWidget *parent)
    : QGraphicsView(parent),
      zoomLevel(1.0),
      minZoom(0.1),
      maxZoom(10.0),
      shouldFitInView(true),
      isDragging(false),
      currentTool("pointer")
{
    scene = new QGraphicsScene(this);
    setScene(scene);
    setRenderHint(QPainter::Antialiasing);
    setTransformationAnchor(QGraphicsView::AnchorUnderMouse);
    setResizeAnchor(QGraphicsView::AnchorUnderMouse);
    setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    setBackgroundBrush(QBrush(QColor(240, 240, 240)));
    setFrameShape(QFrame::NoFrame);
    setDragMode(QGraphicsView::NoDrag); // 默认为指针模式
    pixmapItem = nullptr;
}

void ZoomableImageViewer::setImage(const QPixmap &pixmap)
{
    scene->clear();
    pixmapItem = scene->addPixmap(pixmap);
    setSceneRect(QRectF(pixmap.rect()));
    
    // 重置缩放状态
    zoomLevel = 1.0;
    shouldFitInView = true;
    
    // 仅在初始设置图像时适应视图
    fitInView(pixmapItem, Qt::KeepAspectRatio);
}

void ZoomableImageViewer::setTool(const QString &toolName)
{
    currentTool = toolName;
    
    if (toolName == "pointer") {
        setDragMode(QGraphicsView::NoDrag);
        setCursor(Qt::ArrowCursor);
    } else if (toolName == "hand") {
        setDragMode(QGraphicsView::ScrollHandDrag);
        setCursor(Qt::OpenHandCursor);
    }
}

void ZoomableImageViewer::fitInView()
{
    if (pixmapItem && shouldFitInView) {
        QGraphicsView::fitInView(pixmapItem, Qt::KeepAspectRatio);
        zoomLevel = 1.0;
    }
}

void ZoomableImageViewer::wheelEvent(QWheelEvent *event)
{
    // 获取滚轮事件信息
    int delta = event->angleDelta().y();
    
    // 一旦用户开始缩放，就禁用自动适应视图
    shouldFitInView = false;
    
    // 设置缩放因子和方向
    double factor = 1.15; // 稍微增加缩放因子，使缩放更明显
    
    // 确保缩放方向直观：向上滚动放大，向下滚动缩小
    double newZoom;
    if (delta > 0) { // 向上滚动
        // 放大
        newZoom = zoomLevel * factor;
    } else { // 向下滚动
        // 缩小
        newZoom = zoomLevel / factor;
    }
    
    // 应用缩放限制
    if (newZoom < minZoom) {
        newZoom = minZoom;
    } else if (newZoom > maxZoom) {
        newZoom = maxZoom;
    }
    
    // 计算实际缩放因子
    double actualFactor = newZoom / zoomLevel;
    
    // 应用缩放，确保缩放量有效
    if (std::abs(actualFactor - 1.0) > 0.001) {
        scale(actualFactor, actualFactor);
        zoomLevel = newZoom;
    }
    
    // 阻止事件传递，避免滚动条同时响应
    event->accept();
}

void ZoomableImageViewer::mousePressEvent(QMouseEvent *event)
{
    if (event->button() == Qt::MiddleButton) {
        // 中键点击重置缩放
        if (pixmapItem) {
            shouldFitInView = true;
            fitInView(pixmapItem, Qt::KeepAspectRatio);
            zoomLevel = 1.0;
            event->accept();
            return;
        }
    } else if (event->button() == Qt::LeftButton) {
        // 左键按下，可能开始拖动
        isDragging = true;
        dragStartPos = event->pos();
        if (currentTool == "hand") {
            setCursor(Qt::ClosedHandCursor); // 显示抓取手势
        }
    }
    
    // 调用父类方法
    QGraphicsView::mousePressEvent(event);
}

void ZoomableImageViewer::mouseMoveEvent(QMouseEvent *event)
{
    // 如果处于拖动状态，并且是手型工具模式
    if (isDragging && currentTool == "hand" && !dragStartPos.isNull()) {
        // 实现拖动逻辑
        QPoint delta = event->pos() - dragStartPos;
        dragStartPos = event->pos();
        
        // 移动视图
        horizontalScrollBar()->setValue(horizontalScrollBar()->value() - delta.x());
        verticalScrollBar()->setValue(verticalScrollBar()->value() - delta.y());
        
        event->accept();
        return;
    }
    
    QGraphicsView::mouseMoveEvent(event);
}

void ZoomableImageViewer::mouseReleaseEvent(QMouseEvent *event)
{
    if (event->button() == Qt::LeftButton) {
        // 左键释放，结束拖动
        isDragging = false;
        if (currentTool == "hand") {
            setCursor(Qt::OpenHandCursor); // 恢复普通手势
        }
    }
    
    // 调用父类方法
    QGraphicsView::mouseReleaseEvent(event);
}

void ZoomableImageViewer::resizeEvent(QResizeEvent *event)
{
    QGraphicsView::resizeEvent(event);
    
    // 只有在初始状态或明确请求时才适应视图
    if (pixmapItem && shouldFitInView) {
        fitInView(pixmapItem, Qt::KeepAspectRatio);
    }
} 