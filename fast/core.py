import sys
import cv2
import os
import time
import threading
import queue
import numpy as np
from multiprocessing import Queue, Event

try:
    import openvino as ov
    OV_AVAILABLE = True
except ImportError:
    OV_AVAILABLE = False
    print("错误：请安装 openvino (pip install openvino)")

DEFAULT_IMG_SZ = 640
CONF_THRESHOLD = 0.30

def preprocess_frame(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (DEFAULT_IMG_SZ, DEFAULT_IMG_SZ))
    img = img / 255.0
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, 0).astype(np.float32)
    return img

def postprocess_output(output, frame_shape, conf_threshold=CONF_THRESHOLD):
    output = np.squeeze(output)
    h, w = frame_shape[:2]
    scale_x = w / DEFAULT_IMG_SZ
    scale_y = h / DEFAULT_IMG_SZ
    detections = []

    if len(output.shape) == 1:
        output = output.reshape(1, -1)

    for det in output:
        if len(det) < 5:
            continue
        val1, val2, val3, val4, conf = det[0], det[1], det[2], det[3], det[4]
        if conf < conf_threshold:
            continue

        x1, y1, x2, y2 = 0, 0, 0, 0
        if val3 < 2.0:
            cx, cy, bw, bh = val1, val2, val3, val4
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
        else:
            if val3 < val1:
                cx, cy, bw, bh = val1, val2, val3, val4
                x1 = int((cx - bw / 2) * scale_x)
                y1 = int((cy - bh / 2) * scale_y)
                x2 = int((cx + bw / 2) * scale_x)
                y2 = int((cy + bh / 2) * scale_y)
            else:
                x1 = int(val1 * scale_x)
                y1 = int(val2 * scale_y)
                x2 = int(val3 * scale_x)
                y2 = int(val4 * scale_y)

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 > x1 and y2 > y1:
            detections.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'conf': float(conf)})
    return detections

def worker_process_v3(camera_index, model_path, raw_queue, stop_event,
                      manager_dict, init_queue, command_queue, result_queue):
    """
    子进程：持续捕获原始帧用于显示，监听点名命令，收到后执行单次推理。
    """
    # 加载模型
    try:
        core = ov.Core()
        ov_model = core.read_model(os.path.join(model_path, "yolo26n-face.xml"))
        device = "GPU" if "GPU" in core.available_devices else "CPU"
        print(f"OpenVINO 可用设备: {core.available_devices}, 选择使用: {device}")
        compiled_model = core.compile_model(ov_model, device)
        input_layer = compiled_model.input(0)
        output_layer = compiled_model.output(0)
        print("OpenVINO 模型加载成功")
    except Exception as e:
        print(f"模型加载失败：{e}")
        init_queue.put(False)
        return

    init_queue.put(True)

    # 辅助函数：打开摄像头（使用 DSHOW 后端以增加稳定性）
    def open_camera(idx):
        if os.name == 'nt':
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            cap.set(cv2.CAP_PROP_FPS, 60)
        return cap

    # 捕获线程函数（持续读取，自动重连）
    def capture_loop(cap, cam_idx, stop_flag):
        while not stop_flag.is_set() and not stop_event.is_set():
            if cap is None or not cap.isOpened():
                # 尝试重新打开摄像头
                time.sleep(0.5)
                cap = open_camera(cam_idx)
                if cap is None or not cap.isOpened():
                    continue
            ret, frame = cap.read()
            if not ret:
                # 读取失败，标记 cap 无效，下一轮重新打开
                cap.release()
                cap = None
                continue
            # 成功读取，放入队列
            try:
                if raw_queue.full():
                    raw_queue.get_nowait()
                raw_queue.put(frame)
            except:
                pass
        if cap is not None:
            cap.release()
        print("捕获线程退出")

    # 主循环：管理摄像头切换和命令响应
    current_cam_idx = camera_index
    stop_capture = threading.Event()
    capture_thread = None
    cap = None

    while not stop_event.is_set():
        # 检查切换请求
        target_cam_idx = manager_dict.get('cam_index', current_cam_idx)
        if target_cam_idx != current_cam_idx:
            print(f"切换摄像头: {current_cam_idx} -> {target_cam_idx}")
            # 停止当前捕获线程
            if capture_thread is not None:
                stop_capture.set()
                capture_thread.join(timeout=1)
                stop_capture.clear()
            # 释放当前摄像头
            if cap is not None:
                cap.release()
                cap = None
            current_cam_idx = target_cam_idx

        # 确保捕获线程运行
        if capture_thread is None or not capture_thread.is_alive():
            cap = open_camera(current_cam_idx)
            if cap is None or not cap.isOpened():
                print(f"无法打开摄像头 {current_cam_idx}，1秒后重试")
                time.sleep(1)
                continue
            stop_capture.clear()
            capture_thread = threading.Thread(target=capture_loop, args=(cap, current_cam_idx, stop_capture), daemon=True)
            capture_thread.start()
            print(f"摄像头 {current_cam_idx} 捕获线程已启动")

        # 检查命令队列（点名）
        try:
            cmd = command_queue.get(timeout=0.1)
        except:
            # 没有命令，继续循环
            continue

        if cmd == "rollcall":
            # 点名：直接从 cap 读取最新帧（避免队列延迟）
            if cap is None or not cap.isOpened():
                # 尝试重新打开
                cap = open_camera(current_cam_idx)
                if cap is None or not cap.isOpened():
                    result_queue.put((None, []))
                    continue
            ret, frame = cap.read()
            if not ret:
                # 如果读取失败，尝试从 raw_queue 取最后一帧
                try:
                    frame = raw_queue.get_nowait()
                except:
                    frame = None
            if frame is None:
                result_queue.put((None, []))
                continue

            # 推理
            input_data = preprocess_frame(frame)
            infer_request = compiled_model.create_infer_request()
            infer_request.infer({input_layer.any_name: input_data})
            output = infer_request.get_output_tensor(output_layer.index).data
            detections = postprocess_output(output, frame.shape)

            # 绘制绿色细框（所有检测到的人脸）
            marked_frame = frame.copy()
            for det in detections:
                x1, y1, x2, y2 = int(det['x1']), int(det['y1']), int(det['x2']), int(det['y2'])
                cv2.rectangle(marked_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

            # 转为 RGB 并返回
            marked_frame_rgb = cv2.cvtColor(marked_frame, cv2.COLOR_BGR2RGB)
            result_queue.put((marked_frame_rgb, detections))
            manager_dict['latest_detections'] = detections

    # 清理
    if capture_thread is not None:
        stop_capture.set()
        capture_thread.join(timeout=1)
    if cap is not None:
        cap.release()
    print("子进程退出")
    sys.exit(0)