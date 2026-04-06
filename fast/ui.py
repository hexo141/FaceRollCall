import sys
import cv2
import time
from PySide6.QtWidgets import (QMainWindow, QLabel, QPushButton,
                               QVBoxLayout, QWidget, QHBoxLayout, QSpinBox,
                               QComboBox, QMessageBox, QDialog, QScrollArea, QStatusBar)
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import QTimer, Qt, Slot

# 尝试导入 psutil
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("提示：安装 psutil 可显示 CPU 占用率 (pip install psutil)")

# ==============================================================================
# 带标记图像显示弹窗
# ==============================================================================
class MarkedImageDialog(QDialog):
    def __init__(self, marked_image_np, selected_count, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"点名结果 - 已标记 {selected_count} 位同学")
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
        self.resize(800, 600)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        h, w, ch = marked_image_np.shape
        bytes_per_line = ch * w
        q_img = QImage(marked_image_np.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        scaled_pixmap = pixmap.scaled(800, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        label = QLabel()
        label.setPixmap(scaled_pixmap)
        label.setAlignment(Qt.AlignCenter)

        scroll.setWidget(label)

        layout = QVBoxLayout()
        layout.addWidget(scroll)
        self.setLayout(layout)

# ==============================================================================
# 主窗口类
# ==============================================================================
class FaceRollCallApp(QMainWindow):
    def __init__(self, raw_queue, shared_dict, stop_event, worker_process,
                 selected_cam_index, available_cams, command_queue, result_queue):
        super().__init__()
        self.setWindowTitle("学生人脸点名器 FaceRollCall")
        self.showMaximized()

        # 接收外部传入的资源
        self.raw_queue = raw_queue
        self.shared_dict = shared_dict
        self.stop_event = stop_event
        self.worker_process = worker_process
        self.is_running = True
        self.selected_cam_index = selected_cam_index
        self.command_queue = command_queue
        self.result_queue = result_queue

        # 点名进行中标志，用于禁止重复点击
        self.rollcall_in_progress = False

        # 定时器用于更新实时画面（无检测框）
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        # 状态栏信息定时器（每秒更新一次）
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start(1000)

        # 点名结果检查定时器（仅点名时启用）
        self.result_check_timer = QTimer()
        self.result_check_timer.timeout.connect(self.check_rollcall_result)

        # 初始化 UI
        self.init_ui(available_cams)

    def init_ui(self, available_cams):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # 视频显示区域
        self.video_label = QLabel("等待视频流...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("QLabel { background-color: #000; color: #fff; border: 1px solid #555; }")
        self.video_label.setMinimumSize(640, 480)
        main_layout.addWidget(self.video_label, 1)

        # 控制区域
        control_layout = QHBoxLayout()

        # 摄像头选择
        self.cam_combo = QComboBox()
        self.cam_combo.setEditable(False)
        for idx in available_cams:
            self.cam_combo.addItem(f"Camera {idx}", idx)

        if self.selected_cam_index in available_cams:
            self.cam_combo.setCurrentIndex(available_cams.index(self.selected_cam_index))

        self.cam_combo.setEnabled(True)
        self.cam_combo.currentIndexChanged.connect(self.on_camera_changed)

        control_layout.addWidget(QLabel("摄像头: "))
        control_layout.addWidget(self.cam_combo)

        # 选取人数
        self.count_spin = QSpinBox()
        self.count_spin.setRange(1, 50)
        self.count_spin.setValue(5)
        control_layout.addWidget(QLabel("选取人数: "))
        control_layout.addWidget(self.count_spin)

        # 点名按钮
        self.roll_call_btn = QPushButton("开始点名")
        self.roll_call_btn.clicked.connect(self.start_roll_call)
        self.roll_call_btn.setEnabled(True)
        self.roll_call_btn.setStyleSheet("background-color: #3498db; color: white; font-weight: bold; padding: 5px; ")
        control_layout.addWidget(self.roll_call_btn)

        main_layout.addLayout(control_layout)

        # 底部状态栏
        status_bar = QStatusBar()
        self.setStatusBar(status_bar)
        self.status_label = QLabel("初始化...")
        status_bar.addWidget(self.status_label)

    @Slot(int)
    def on_camera_changed(self, index):
        if index < 0:
            return
        cam_id = self.cam_combo.itemData(index)
        print(f"UI: 请求切换到摄像头 {cam_id}")
        self.shared_dict['cam_index'] = cam_id
        self.video_label.clear()
        self.video_label.setText("切换摄像头中...")

    def update_status(self):
        """每秒更新状态栏：点名人数、CPU占用"""
        detections = self.shared_dict.get('latest_detections', [])
        face_count = len(detections)

        if PSUTIL_AVAILABLE:
            cpu_percent = psutil.cpu_percent(interval=None)
            cpu_str = f"CPU: {cpu_percent:.1f}%"
        else:
            cpu_str = "CPU: N/A"

        # 修改文字：“可抽取人数” -> “点名人数”
        status_text = f"点名人数: {face_count}  |  {cpu_str}"
        self.status_label.setText(status_text)

    @Slot()
    def update_frame(self):
        """从 raw_queue 获取原始帧并显示（无检测框）"""
        if self.is_running and not self.raw_queue.empty():
            try:
                frame_bgr = self.raw_queue.get_nowait()
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                h, w, ch = frame_rgb.shape
                bytes_per_line = ch * w
                q_img = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_img)
                scaled_pixmap = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.video_label.setPixmap(scaled_pixmap)
            except Exception:
                pass

    def start_roll_call(self):
        """点名：发送命令，禁止重复点击，等待子进程返回结果"""
        if not self.is_running or self.rollcall_in_progress:
            return

        # 禁用点名按钮和摄像头切换
        self.roll_call_btn.setEnabled(False)
        self.cam_combo.setEnabled(False)
        self.rollcall_in_progress = True

        # 发送点名命令
        self.command_queue.put("rollcall")

        # 启动结果检查定时器
        self.result_check_timer.start(50)  # 每50ms检查一次

    def check_rollcall_result(self):
        """定时检查结果队列，收到后处理并恢复界面"""
        if self.result_queue.empty():
            return

        # 停止定时器
        self.result_check_timer.stop()

        # 获取结果 (marked_frame_rgb, detections)
        result = self.result_queue.get()
        marked_frame_rgb, detections = result

        if marked_frame_rgb is None:
            QMessageBox.warning(self, "提示", "无法获取摄像头画面，请检查摄像头连接。")
        elif len(detections) == 0:
            QMessageBox.information(self, "提示", "当前画面未检测到人脸，请调整摄像头角度或光线。")
        else:
            # 按置信度排序，选取前 N 个
            num_to_select = self.count_spin.value()
            actual_count = min(len(detections), num_to_select)
            sorted_detections = sorted(detections, key=lambda k: k['conf'], reverse=True)
            selected_detections = sorted_detections[:actual_count]

            # marked_frame_rgb 已经是 RGB 格式，直接复制使用
            marked_frame = marked_frame_rgb.copy()
            for det in selected_detections:
                x1, y1, x2, y2 = int(det['x1']), int(det['y1']), int(det['x2']), int(det['y2'])
                # 在 RGB 图像上绘制红色框，使用 (255, 0, 0) 表示红色
                cv2.rectangle(marked_frame, (x1, y1), (x2, y2), (255, 0, 0), 3)

            # 直接使用 marked_frame（已经是 RGB），不需要再转换颜色
            dialog = MarkedImageDialog(marked_frame, actual_count, self)
            dialog.exec()
            QMessageBox.information(self, "点名完成", f"已从当前画面标记 {actual_count} 位同学")

        # 恢复界面
        self.roll_call_btn.setEnabled(True)
        self.cam_combo.setEnabled(True)
        self.rollcall_in_progress = False