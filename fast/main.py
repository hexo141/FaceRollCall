import sys
import multiprocessing
from PySide6.QtWidgets import (QApplication, QDialog, QVBoxLayout, QHBoxLayout,
                               QLabel, QProgressBar, QMessageBox)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap

class CustomProgressDialog(QDialog):
    def __init__(self, title="初始化中", label_text="正在启动程序，请稍候...", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.resize(400, 90)

        layout = QVBoxLayout(self)
        layout.setSpacing(5)
        layout.setContentsMargins(10, 10, 10, 10)

        self.text_label = QLabel(label_text)
        layout.addWidget(self.text_label)

        h_layout = QHBoxLayout()
        h_layout.setSpacing(8)

        self.icon_label = QLabel()
        try:
            pixmap = QPixmap("arrow_right.png")
            if not pixmap.isNull():
                scaled_pixmap = pixmap.scaled(24, 24, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.icon_label.setPixmap(scaled_pixmap)
            else:
                self.icon_label.setText("→")
                self.icon_label.setStyleSheet("font-size: 20px; font-weight: bold;")
        except:
            self.icon_label.setText("→")
            self.icon_label.setStyleSheet("font-size: 20px; font-weight: bold;")

        h_layout.addWidget(self.icon_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p%")
        h_layout.addWidget(self.progress_bar)

        layout.addLayout(h_layout)

    def setValue(self, value):
        self.progress_bar.setValue(value)
        QApplication.processEvents()

    def setLabelText(self, text):
        self.text_label.setText(text)
        QApplication.processEvents()

def dynamic_imports(progress_callback):
    progress_callback(10, "正在导入 OpenCV...")
    import cv2
    progress_callback(30, "OpenCV 导入完成")
    progress_callback(40, "正在准备其他模块...")
    import os
    from multiprocessing import Process, Queue, Event, Manager
    progress_callback(50, "基础模块准备完成")
    return cv2, os, Process, Queue, Event, Manager

def main():
    multiprocessing.freeze_support()
    app = QApplication(sys.argv)

    progress = CustomProgressDialog("初始化中", "正在启动程序，请稍候...")
    progress.show()

    def update_progress(value, text):
        progress.setValue(value)
        progress.setLabelText(text)

    try:
        cv2, os, Process, Queue, Event, Manager = dynamic_imports(update_progress)

        update_progress(55, "检查模型文件夹...")
        MODEL_PATH = "yolo26n-face_openvino_model"
        if not os.path.exists(MODEL_PATH):
            progress.close()
            QMessageBox.critical(None, "错误", f"模型文件夹 {MODEL_PATH} 不存在，请放入程序目录。")
            sys.exit(1)

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
        selected_cam = available_cams[0]
        update_progress(70, f"已选择摄像头 {selected_cam}")

        update_progress(75, "准备多进程通信...")
        raw_queue = Queue(maxsize=1)          # 用于实时显示原始帧
        command_queue = Queue()               # 用于发送点名命令
        result_queue = Queue()                # 用于接收点名结果（带框帧+检测结果）
        init_queue = Queue(maxsize=1)
        stop_event = Event()
        manager = Manager()
        shared_dict = manager.dict()
        shared_dict['latest_detections'] = []

        update_progress(80, "启动人脸检测引擎...")
        from core import worker_process_v3   # 注意函数名改为 worker_process_v3
        worker = Process(
            target=worker_process_v3,
            args=(selected_cam, MODEL_PATH, raw_queue, stop_event,
                  shared_dict, init_queue, command_queue, result_queue)
        )
        worker.start()

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

        from ui import FaceRollCallApp
        window = FaceRollCallApp(
            raw_queue=raw_queue,
            shared_dict=shared_dict,
            stop_event=stop_event,
            worker_process=worker,
            selected_cam_index=selected_cam,
            available_cams=available_cams,
            command_queue=command_queue,
            result_queue=result_queue
        )

        update_progress(100, "启动完成")
        progress.close()

        window.show()
        sys.exit(app.exec())

    except Exception as e:
        progress.close()
        QMessageBox.critical(None, "初始化失败", f"发生错误：{str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()