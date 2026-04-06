import sys
import multiprocessing
from PySide6.QtWidgets import (QApplication, QDialog, QVBoxLayout, QHBoxLayout,
                               QLabel, QProgressBar, QMessageBox)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap

# 延迟导入，以便先显示进度条

class CustomProgressDialog(QDialog):
    def __init__(self, title="初始化中", label_text="正在启动程序，请稍候...", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.resize(400, 90)  # 减小高度从100到90
        
        layout = QVBoxLayout(self)
        layout.setSpacing(5)  # 设置控件之间的间距为5像素
        layout.setContentsMargins(10, 10, 10, 10)  # 设置外边距（左，上，右，下）
        
        # 文本标签
        self.text_label = QLabel(label_text)
        layout.addWidget(self.text_label)
        
        # 水平布局：箭头图标 + 进度条
        h_layout = QHBoxLayout()
        h_layout.setSpacing(8)  # 箭头和进度条之间的间距
        
        # 创建图标标签
        self.icon_label = QLabel()
        # 尝试加载自定义图片，如果不存在则使用系统默认箭头
        try:
            pixmap = QPixmap("arrow_right.png")
            if not pixmap.isNull():
                scaled_pixmap = pixmap.scaled(24, 24, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.icon_label.setPixmap(scaled_pixmap)
            else:
                # 图片不存在，使用文本箭头
                self.icon_label.setText("→")
                self.icon_label.setStyleSheet("font-size: 20px; font-weight: bold;")
        except:
            # 加载失败，使用文本箭头
            self.icon_label.setText("→")
            self.icon_label.setStyleSheet("font-size: 20px; font-weight: bold;")
        
        h_layout.addWidget(self.icon_label)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p%")
        h_layout.addWidget(self.progress_bar)
        
        layout.addLayout(h_layout)
        
    def setValue(self, value):
        """设置进度值 (0-100)"""
        self.progress_bar.setValue(value)
        QApplication.processEvents()
        
    def setLabelText(self, text):
        """设置提示文本"""
        self.text_label.setText(text)
        QApplication.processEvents()

def dynamic_imports(progress_callback):
    """动态导入所需库，并更新进度"""
    # 步骤1: 导入 OpenCV (10% -> 30%)
    progress_callback(10, "正在导入 OpenCV...")
    import cv2
    progress_callback(30, "OpenCV 导入完成")

    # 步骤2: 其他标准库导入 (30% -> 50%)
    progress_callback(40, "正在准备其他模块...")
    import os
    from multiprocessing import Process, Queue, Event, Manager
    progress_callback(50, "基础模块准备完成")

    return cv2, os, Process, Queue, Event, Manager

def main():
    multiprocessing.freeze_support()
    
    # 必须先创建 QApplication 才能显示界面
    app = QApplication(sys.argv)
    
    # 创建自定义进度条对话框
    progress = CustomProgressDialog("初始化中", "正在启动程序，请稍候...")
    progress.show()
    
    def update_progress(value, text):
        progress.setValue(value)
        progress.setLabelText(text)
    
    try:
        # ---------- 动态导入所有库 ----------
        cv2, os, Process, Queue, Event, Manager = dynamic_imports(update_progress)
        
        # ---------- 检查模型文件 ----------
        update_progress(55, "检查模型文件夹...")
        MODEL_PATH = "yolo26n-face_openvino_model"
        if not os.path.exists(MODEL_PATH):
            progress.close()
            QMessageBox.critical(None, "错误", f"模型文件夹 {MODEL_PATH} 不存在，请放入程序目录。")
            sys.exit(1)
        
        # ---------- 扫描摄像头 ----------
        update_progress(60, "正在扫描摄像头...")
        available_cams = []
        for i in range(10):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_V4L2)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    available_cams.append(i)
                cap.release()
        if not available_cams:
            progress.close()
            QMessageBox.critical(None, "错误", "未检测到可用摄像头，程序将退出。")
            sys.exit(1)
        selected_cam = available_cams[0]  # 自动选择第一个可用摄像头
        update_progress(70, f"已选择摄像头 {selected_cam}")
        
        # ---------- 创建多进程通信对象 ----------
        update_progress(75, "准备多进程通信...")
        frame_queue = Queue(maxsize=1)
        raw_queue = Queue(maxsize=1)
        init_queue = Queue(maxsize=1)
        stop_event = Event()
        manager = Manager()
        shared_dict = manager.dict()
        shared_dict['latest_detections'] = []
        
        # ---------- 启动子进程（摄像头+推理） ----------
        update_progress(80, "启动人脸检测引擎...")
        from core import worker_process_v2
        worker = Process(
            target=worker_process_v2,
            args=(selected_cam, MODEL_PATH, frame_queue,
                  raw_queue, stop_event, shared_dict, init_queue)
        )
        worker.start()
        
        # ---------- 等待子进程初始化完成 ----------
        update_progress(85, "等待摄像头和模型加载...")
        import time
        timeout = 60
        start_time = time.time()
        while init_queue.empty():
            if time.time() - start_time > timeout:
                progress.close()
                QMessageBox.critical(None, "错误", "初始化超时，请检查摄像头或模型文件。")
                worker.terminate()
                sys.exit(1)
            QApplication.processEvents()
            time.sleep(0.05)
        init_success = init_queue.get()
        if not init_success:
            progress.close()
            QMessageBox.critical(None, "错误", "子进程初始化失败（摄像头打开或模型加载错误）。")
            worker.terminate()
            sys.exit(1)
        
        update_progress(95, "准备界面...")
        
        # ---------- 导入 UI 并创建主窗口 ----------
        from ui import FaceRollCallApp
        window = FaceRollCallApp(
            frame_queue=frame_queue,
            raw_queue=raw_queue,
            shared_dict=shared_dict,
            stop_event=stop_event,
            worker_process=worker,
            selected_cam_index=selected_cam,
            available_cams=available_cams
        )
        
        update_progress(100, "启动完成")
        progress.close()
        
        # 显示主窗口
        window.show()
        sys.exit(app.exec())
        
    except Exception as e:
        progress.close()
        QMessageBox.critical(None, "初始化失败", f"发生错误：{str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()