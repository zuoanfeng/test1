import datetime
import os
import cv2
import sys
import torch
import time
import numpy as np
import math

from PyQt6.QtCore import (QTimer, Qt, QPropertyAnimation, QRect, QAbstractAnimation, QParallelAnimationGroup,
                          QThread, pyqtSignal)

from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, \
    QWidget, QHBoxLayout, QVBoxLayout, QGridLayout, QSizePolicy, QComboBox, QDoubleSpinBox, QSlider, QSpinBox, \
    QProgressBar
from PyQt6.QtGui import QPixmap, QImage, QIcon, QGuiApplication

from ultralytics import YOLO

# 线程类，PyQt6中有两个QLabel分别显示原始图片和检测后图片，这里采样多线程来分别显示（减少检测视频时的延迟）
class WorkerThread(QThread):
    # 线程信号，用于控制视频的逐帧播放
    finished = pyqtSignal()

    def __init__(self, label, image):
        super().__init__()
        self.label = label
        self.image = image

    def run(self):
        # 将原始图片、检测图片进行  numpy格式-》QImage格式转换，以便可以在QLabel显示
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        height, width, ch = self.image.shape
        label_width = int(self.label.width())
        label_height = int(self.label.height())
        if width > height:
            self.image = cv2.resize(self.image, (label_width, int(height * label_width * 1.0 / width)))
            bytes_per_line = ch * label_width
        else:
            self.image = cv2.resize(self.image, (int(width * label_height * 1.0 / height), label_height))
            bytes_per_line = ch * int(width * label_height * 1.0 / height)
        self.image = QImage(self.image.data, self.image.shape[1], self.image.shape[0], bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(self.image)


        # 视频流置于label中间部分播放
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # 启用图像的等比例缩放
        # self.label.setScaledContents(True)
        self.label.setPixmap(pixmap)

        # 发射信号，即在QLabel上显示检测结果
        self.finished.emit()

class FallDetector:
    def __init__(self):
        # COCO关键点索引 (0-16)
        self.NOSE = 0
        self.LEFT_SHOULDER = 5
        self.RIGHT_SHOULDER = 6
        self.LEFT_HIP = 11
        self.RIGHT_HIP = 12
        self.LEFT_ANKLE = 15
        self.RIGHT_ANKLE = 16

    def is_person_fallen(self, keypoints, image_height, image_width):
        """
        使用COCO关键点(17个点)判断人是否跌倒或躺下
        :param keypoints: 包含17个(x,y)元组的列表，顺序与COCO数据集一致
        :param image_height: 图像高度（用于归一化判断）
        :param image_width: 图像宽度（用于归一化判断）
        :return: (bool是否跌倒, dict诊断信息)
        """
        # 1. 获取所需关键点（直接按索引获取）
        try:
            head = keypoints[self.NOSE]
            left_ankle = keypoints[self.LEFT_ANKLE]
            right_ankle = keypoints[self.RIGHT_ANKLE]
            left_shoulder = keypoints[self.LEFT_SHOULDER]
            right_shoulder = keypoints[self.RIGHT_SHOULDER]
            left_hip = keypoints[self.LEFT_HIP]
            right_hip = keypoints[self.RIGHT_HIP]
        except IndexError:
            return False, {'error': '关键点数量不足17个'}

        # 2. 垂直方向分析 - 头部和脚部的Y坐标差
        ankle_y = max(left_ankle[1], right_ankle[1])
        head_ankle_diff = abs(head[1] - ankle_y)

        # 3. 身体角度分析 - 肩膀和臀部形成的线与水平面的角度
        # 计算肩膀和臀部中点
        shoulder_center = (
            (left_shoulder[0] + right_shoulder[0]) // 2,
            (left_shoulder[1] + right_shoulder[1]) // 2
        )
        hip_center = (
            (left_hip[0] + right_hip[0]) // 2,
            (left_hip[1] + right_hip[1]) // 2
        )

        # 计算躯干角度（相对于水平面）
        if hip_center[0] != shoulder_center[0]:  # 避免除以零
            torso_angle = math.degrees(math.atan2(hip_center[1] - shoulder_center[1],
                                                  hip_center[0] - shoulder_center[0]))
            torso_angle = abs(torso_angle)
        else:
            torso_angle = 90  # 完全垂直

        # 4. 关键点高度比 - 肩膀是否低于臀部
        shoulder_below_hip = shoulder_center[1] > hip_center[1]

        # 5. 综合判断条件
        condition1 = head_ankle_diff < 0.3 * image_height  # 头部和脚部垂直距离小
        condition2 = torso_angle < 45 or torso_angle > 135  # 躯干接近水平
        condition3 = shoulder_below_hip  # 肩膀低于臀部

        # 判断是否跌倒（至少满足1个条件）
        fallen = sum([condition1, condition2, condition3]) >= 1

        return fallen, {
            'head_ankle_diff': head_ankle_diff,
            'torso_angle': torso_angle,
            'shoulder_below_hip': shoulder_below_hip,
            'conditions_met': [condition1, condition2, condition3]
        }

class DetectThread(QThread):
    # 线程信号，用于控制视频的逐帧播放
    finished = pyqtSignal()

    def __init__(self):
        super().__init__()

        self.img = None
        self.det_model = None
        self.conf = None
        self.iou = None
        self.results = []

        self.fall_det = FallDetector()

    def run(self):
        if self.img is None:
            print("请重新上传")
            return
        img_ = np.copy(self.img)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        start_time = time.time()
        self.det_model.eval()
        results = self.det_model.predict(
            source=img_,
            conf=self.conf,
            iou=self.iou,
            verbose=False,
            device=device,
            save=False
        )[0]
        boxes = results.boxes.xyxy.detach().cpu().numpy().tolist()
        conf = results.boxes.conf.detach().cpu().numpy().tolist()
        keypoints = results.keypoints.data.detach().cpu().numpy().tolist()

        if len(boxes) < 1:
            self.results = [img_, 0, ["未检测到人"]]
            self.finished.emit()
            return

        target = []
        for id, box in enumerate(boxes):
            box = [int(x) for x in box]
            x1, y1, x2, y2 = box

            keypoint = keypoints[id]
            keypoint = [(int(x), int(y)) for x, y, _ in keypoint]
            score = conf[id]

            img_ = self.connect_keypoints(img_, keypoint)
            # 这里可以添加人体行为分类的代码
            # 例如，根据关键点坐标计算特征，然后使用分类器进行分类
            # 检测是否跌倒
            # ···············
            fallen, _ = self.fall_det.is_person_fallen(keypoint, img_.shape[0], img_.shape[1])

            # 准备要显示的文本 这里显示 names[cls_id] （people）当作为跌倒识别，可自行更改为fall或者no fall
            det_cls = "fallen" if fallen else "no fallen"
            color = (255, 192, 203) if not fallen else (0, 0, 255)
            text = det_cls + f" {score:.2f}"
            target.append(det_cls)
            fontFace = cv2.FONT_HERSHEY_SIMPLEX
            # 在图像上绘制文本
            cv2.putText(img_, text, (x1, y1 - 10), fontFace=fontFace, fontScale=0.5,
                        lineType=cv2.LINE_AA, color=color, thickness=2
                        )
            if not fallen:
                cv2.rectangle(img_, (x1, y1), (x2, y2), color, 2)  # 对图片上检测到的人进行画框框
            else:
                cv2.rectangle(img_, (x1, y1), (x2, y2), color, 2)  # 对图片上检测到的人进行画框框

        end_time = time.time()
        fps = 1 / (end_time - start_time)

        self.results = [img_, fps, target]
        self.finished.emit()

    def set_param(self, img, iou, conf):
        self.img = img
        self.conf = conf
        self.iou = iou

    def connect_keypoints(self, img, keypoints):
        # 定义需要连接的关键点对
        green = (0, 255, 0)
        blue = (255, 0, 0)
        orange = (0, 165, 255)

        connections = [
            (0, 1, blue), (0, 2, blue), (1, 2, blue), (2, 4, blue), (1, 3, blue), (4, 6, blue), (3, 5, blue),

            (6, 5, green), (6, 8, green), (5, 7, green), (8, 10, green), (7, 9, green), (6, 12, green), (5, 11, green),
            (11, 12, green),

            (12, 14, orange), (11, 13, orange), (14, 16, orange), (13, 15, orange)
        ]

        for keypoint in keypoints:
            x, y = keypoint
            x = int(x)
            y = int(y)
            cv2.circle(img, (x, y), 4, (0, 0, 255), -1)

        for connection in connections:
            start_idx, end_idx, color = connection
            start_point = tuple(keypoints[start_idx])
            end_point = tuple(keypoints[end_idx])
            if start_point == (0, 0) or end_point == (0, 0):
                continue
            cv2.line(img, start_point, end_point, color, 2)  # 绘制绿色线条连接关键点

        return img


class VideoSave(QThread):
    finished = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.img_list = None
        self.save_path = None
        self.fps = None

    def run(self):
        self.images_to_video(self.img_list, self.save_path, self.fps)

    def set_params(self, img_list, save_path, fps):
        self.img_list = img_list
        self.save_path = save_path
        self.fps = fps

    def images_to_video(self, image_list, save_path, fps=30):
        """
        将图片列表保存为视频
        :param image_list: 包含图片的列表，图片为 numpy 数组形式
        :param save_path: 视频保存的路径
        :param fps: 视频的帧率，默认为 30
        """
        # 检查图片列表是否为空
        if not image_list:
            print("图片列表为空，无法生成视频。")
            return

        # 获取图片的尺寸
        height, width, _ = image_list[0].shape

        # 定义视频编码器
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        # 创建 VideoWriter 对象
        video_writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

        # 逐帧写入图片
        for image in image_list:
            video_writer.write(image)

        # 释放资源
        video_writer.release()

        print(f"视频已保存到 {save_path}")


# 窗口，即整个PyQt6的界面编写，相关QWidget和
class MainWindow(QMainWindow):
    # 整个PyQt6界面由各种小组件组成，这部分是对所需要的小组件进行初始化
    def __init__(self):
        super(MainWindow, self).__init__()
        # 获取桌面的大小，并设置主窗口高和宽的最大值为桌面的大小（不设置的话，对窗口进行放大时可能出错）
        screen = QGuiApplication.primaryScreen()
        screen_geometry = screen.geometry()
        self.max_width = screen_geometry.width()  # 桌面宽
        self.max_height = screen_geometry.height()  # 桌面高
        self.setMaximumSize(self.max_width, self.max_height)

        # 设置主窗口和底层QWidget的大小为880*520
        self.WIDTH = 1100
        self.HEIGHT = 650
        self.resize(self.WIDTH, self.HEIGHT)
        self.draggable = True
        self.dragging_position = None
        # 记录窗口是否处于最大化状态
        self.is_maximized = False

        # 设置窗口图标为logo图片
        self.setWindowIcon(QIcon("qss_imgs/logo.png"))

        self.center_widget = QWidget(self)  # 底层QWidget的初始化
        self.center_widget.resize(self.WIDTH, self.HEIGHT)  # 设置底层QWidget的宽、高

        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)  # 设置窗口标志，隐藏标题栏和图标
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)  # 表示窗口具有透明效果

        self.c1_widget = QWidget(self.center_widget)  # 左QWidget的初始化，主要用于放置菜单栏
        self.c2_widget = QWidget(self.center_widget)  # 右QWidget的初始化，主要用于放置各种显示结果、画面的组件
        self.anim = QPropertyAnimation(self.c2_widget, b"geometry")  # 菜单栏的过渡动画初始化

        # 放置logo图标
        self.logo_label = QLabel(self.center_widget)
        self.logo_left_label = QLabel(self.center_widget)
        self.logo_right_label = QLabel(self.center_widget)

        # 隐藏菜单栏按钮
        self.hide_account = 1
        self.hide_button = QPushButton(self.center_widget)
        self.hide_left_label = QLabel(self.center_widget)
        self.hide_right_label = QLabel(self.center_widget)

        # 上传文件按钮
        self.file_button = QPushButton(self.center_widget)
        self.file_left_label = QLabel(self.center_widget)
        self.file_right_label = QLabel(self.center_widget)

        # 摄像头按钮
        self.camera_button = QPushButton(self.center_widget)
        self.camera_left_label = QLabel(self.center_widget)
        self.camera_right_label = QLabel(self.center_widget)

        # rtsp按钮，该功能只是待定，并未实现，可以不管，起个装饰作用
        self.rtsp_button = QPushButton(self.center_widget)
        self.rtsp_left_label = QLabel(self.center_widget)
        self.rtsp_right_label = QLabel(self.center_widget)

        # 显示设计的系统为2.0版本，起个装饰作用
        self.version_label = QLabel(self.center_widget)

        # 右上部分的QWidget初始化，用于放置名称、缩小、放大、关闭按钮
        self.c2_top_widget = QWidget(self.center_widget)
        self.c2_top_re = QLabel(self.center_widget)
        self.title = QLabel(self.center_widget)  # 用于放置名称
        self.zoom = QLabel(self.center_widget)
        self.zoom_set = QPushButton(self.center_widget)  # 隐藏按钮
        self.zoom_in = QPushButton(self.center_widget)  # 缩小按钮
        self.zoom_out = QPushButton(self.center_widget)  # 放大按钮
        self.zoom_close = QPushButton(self.center_widget)  # 关闭按钮

        # 右中部分得QWidget用于放置显示结果、画面
        self.c2_center_widget = QWidget(self.center_widget)
        self.c2_center_1 = QWidget(self.center_widget)
        self.detection = QLabel(self.center_widget)

        # 显示目标个数
        self.target_license_fps_model = QWidget(self.center_widget)
        self.target = QWidget(self.center_widget)
        self.target_top = QLabel(self.center_widget)
        self.target_bottom = QLabel(self.center_widget)

        # 显示车牌号码
        self.license = QWidget(self.center_widget)
        self.license_top = QLabel(self.center_widget)
        self.license_bottom = QLabel(self.center_widget)

        # 显示实时检测速度
        self.fps = QWidget(self.center_widget)
        self.fps_top = QLabel(self.center_widget)
        self.fps_bottom = QLabel(self.center_widget)

        # 显示所采用得模型
        self.model = QWidget(self.center_widget)
        self.model_top = QLabel(self.center_widget)
        self.model_bottom = QLabel(self.center_widget)

        # 显示原始图片和检测图片
        self.plays = QWidget(self.center_widget)
        self.plays_left = QLabel(self.center_widget)
        self.plays_right = QLabel(self.center_widget)

        # 显示进度条
        self.bar = QWidget(self.center_widget)
        self.bar_start = QPushButton(self.center_widget)
        self.bar_progress = QProgressBar(self.center_widget)
        self.bar_stop = QLabel(self.center_widget)

        # 初始化两个过渡动画
        self.c2_center_2 = QWidget(self.center_widget)
        self.set_anim_1 = QPropertyAnimation(self.c2_center_1, b"geometry")
        self.set_anim_2 = QPropertyAnimation(self.c2_center_2, b"geometry")
        self.set_account = 1
        # 创建并行动画组
        self.parallel_group = QParallelAnimationGroup(self)
        self.parallel_group.addAnimation(self.set_anim_1)
        self.parallel_group.addAnimation(self.set_anim_2)

        # 显示各种调节参数
        self.set_1 = QLabel(self.center_widget)
        self.set_2 = QWidget(self.center_widget)

        self.set_2_logo = QLabel(self.center_widget)  # 用于模型选择
        self.set_2_name = QLabel(self.center_widget)
        self.set_2_box = QComboBox(self.center_widget)

        self.set_3 = QWidget(self.center_widget)  # 用于IOU调节
        self.set_3_logo = QLabel(self.center_widget)
        self.set_3_name = QLabel(self.center_widget)
        self.set_3_box = QDoubleSpinBox(self.center_widget)
        self.set_3_slider = QSlider(self.center_widget)

        self.set_4 = QWidget(self.center_widget)  # 用于Conf调节
        self.set_4_logo = QLabel(self.center_widget)
        self.set_4_name = QLabel(self.center_widget)
        self.set_4_box = QDoubleSpinBox(self.center_widget)
        self.set_4_slider = QSlider(self.center_widget)

        self.set_5 = QWidget(self.center_widget)  # 用于播放视频的延迟调节
        self.set_5_logo = QLabel(self.center_widget)
        self.set_5_name = QLabel(self.center_widget)
        self.set_5_box = QSpinBox(self.center_widget)
        self.set_5_slider = QSlider(self.center_widget)

        self.set_6 = QWidget(self.center_widget)  # 用于是否保存图片
        self.set_6_logo = QLabel(self.center_widget)
        self.set_6_name = QLabel(self.center_widget)
        self.set_6_save_img = QPushButton(self.center_widget)
        self.set_6_img = QLabel(self.center_widget)
        self.set_6_save_video = QPushButton(self.center_widget)
        self.set_6_video = QLabel(self.center_widget)

        self.c2_bottom_widget = QWidget(self.center_widget)
        self.welcome = QLabel(self.center_widget)

        self.initLayout()  # 组件之间的布局初始化
        self.initSet()  # 组件的设置初始化，如大小、放置的图标、位置等
        self.initUi()  # 组件的样式初始化，该函数功能为给组件添加上各种颜色、外表形状等

        # 以下为各种中间参数
        self.file_path = None  # 保存上传文件的路径

        # 用于播放视频
        self.timer = None
        self.cap = None
        self.delay = None
        self.frame_count = None

        # 保存iou和conf
        self.iou = None
        self.conf = None
        # 判断是否启用摄像头
        self.camera_flag = None

        # 判断进度条是否显示
        self.bar_start_account = 1

        # 保存加载好的模型
        self.Yolo = None  # yolov8
        self.LPRNet = None  # LPRNet

        # 两个播放线程
        self.thread_1 = None
        self.thread_2 = None

        # 保存文件路径
        self.save_imgVideo_flag = False
        self.save_label_flag = False
        self.video_writer = None
        self.camera_path = None

        # 检测线程
        self.detect_thread = DetectThread()
        self.detect_thread.finished.connect(self.detect_finished)

        self.video_save_thread = VideoSave()
        self.video_img_list = []

        self.ret = None

        #未开发，隐藏
        self.rtsp_button.setEnabled(False)
        self.rtsp_button.hide()

        self.set_6_save_video.setEnabled(False)
        self.set_6_save_video.hide()
        self.set_6_video.hide()
    # 该函数即初始化中所提到的布局初始化，主要是方便控制整个界面上的组件的放置位置，使其摆放整齐
    ### 1、QGridLayout为网格布局，QHBoxLayout为横向布局， QVBoxLayout为垂直布局
    ### 2、addWidget用于指明各种布局所作用的组件
    ### 3、setColumnStretch用于控制伸缩量，即组件所占用的区域
    ### 4、setContentsMargins用于控制组件距离QWidget的左、上、右、下的边距
    ### 5、setSpacing用于控制组件之间的距离

    def initLayout(self):
        # 底层QWidget的布局
        center_layout = QGridLayout(self.center_widget)
        center_layout.addWidget(self.c1_widget, 0, 0, 1, 2)
        center_layout.addWidget(self.c2_widget, 0, 1, 1, 2)
        center_layout.setColumnStretch(0, 1)
        center_layout.setColumnStretch(1, 1)
        center_layout.setColumnStretch(2, 12)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(0)

        # 左QWidget的布局
        c1_layout = QGridLayout(self.c1_widget)
        c1_layout.addWidget(self.logo_label, 0, 0, 1, 1)
        c1_layout.addWidget(self.hide_button, 1, 0, 1, 1)
        c1_layout.addWidget(self.file_button, 2, 0, 1, 1)
        c1_layout.addWidget(self.camera_button, 3, 0, 1, 1)
        c1_layout.addWidget(self.rtsp_button, 4, 0, 1, 1)
        c1_layout.addWidget(self.version_label, 6, 0, 1, 1)
        c1_layout.setRowStretch(0, 2)
        c1_layout.setRowStretch(1, 1)
        c1_layout.setRowStretch(2, 1)
        c1_layout.setRowStretch(3, 1)
        c1_layout.setRowStretch(4, 1)
        c1_layout.setRowStretch(5, 3)
        c1_layout.setRowStretch(6, 1)
        c1_layout.setContentsMargins(0, 0, 0, -1)
        c1_layout.setSpacing(0)

        # logo图标以及作者署名的布局
        logo_center_layout = QHBoxLayout(self.logo_label)
        logo_center_layout.addWidget(self.logo_left_label)
        logo_center_layout.addWidget(self.logo_right_label)
        logo_center_layout.setStretch(0, 2)
        logo_center_layout.setStretch(1, 1)
        self.logo_right_label.setText("Yolo Qt\nby Su  ")  # 设置文字
        self.logo_right_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)  # 设置成中间
        self.logo_right_label.setIndent(13)
        logo_center_layout.setContentsMargins(5, 25, 0, 25)
        logo_center_layout.setSpacing(20)

        # 隐藏菜单栏的布局
        hide_layout = QHBoxLayout(self.hide_button)
        hide_layout.addWidget(self.hide_left_label)
        hide_layout.addWidget(self.hide_right_label)
        hide_layout.setStretch(0, 1)
        hide_layout.setStretch(1, 1)
        hide_layout.setContentsMargins(13, 16, 13, 16)
        hide_layout.setSpacing(41)
        self.hide_right_label.setText("Hide")

        # 文件按钮布局
        file_layout = QHBoxLayout(self.file_button)
        file_layout.addWidget(self.file_left_label)
        file_layout.addWidget(self.file_right_label)
        file_layout.setStretch(0, 1)
        file_layout.setStretch(1, 1)
        file_layout.setContentsMargins(12, 13, 0, 13)
        file_layout.setSpacing(23)
        self.file_right_label.setText("Local File")

        # 摄像头按钮布局
        camera_layout = QHBoxLayout(self.camera_button)
        camera_layout.addWidget(self.camera_left_label)
        camera_layout.addWidget(self.camera_right_label)
        camera_layout.setStretch(0, 1)
        camera_layout.setStretch(1, 1)
        camera_layout.setContentsMargins(13, 11, 5, 11)
        camera_layout.setSpacing(23)
        self.camera_right_label.setText("Camera")

        # rtsp布局
        detect_layout = QHBoxLayout(self.rtsp_button)
        detect_layout.addWidget(self.rtsp_left_label)
        detect_layout.addWidget(self.rtsp_right_label)
        detect_layout.setStretch(0, 1)
        detect_layout.setStretch(1, 1)
        detect_layout.setContentsMargins(13, 16, 8, 16)
        detect_layout.setSpacing(28)
        self.rtsp_right_label.setText("RTSP")

        # 版本号布局
        self.version_label.setText("Version:2.0")
        self.version_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignCenter)
        self.version_label.setIndent(4)

        # 右QWidget部分布局
        c2_layout = QVBoxLayout(self.c2_widget)
        c2_layout.addWidget(self.c2_top_widget)
        c2_layout.addWidget(self.c2_center_widget)
        c2_layout.addWidget(self.c2_bottom_widget)
        c2_layout.setStretch(0, 1)
        c2_layout.setStretch(1, 20)
        c2_layout.setStretch(2, 1)
        c2_layout.setContentsMargins(0, 0, 0, 0)
        c2_layout.setSpacing(0)

        # 右上部分布局
        c2_top_layout = QHBoxLayout(self.c2_top_widget)
        c2_top_layout.addWidget(self.c2_top_re)
        c2_top_layout.addWidget(self.title)
        c2_top_layout.addWidget(self.zoom)
        c2_top_layout.setStretch(0, 1)
        c2_top_layout.setStretch(1, 3)
        c2_top_layout.setStretch(2, 1)
        c2_top_layout.setContentsMargins(0, 0, 0, 0)
        c2_top_layout.setSpacing(0)
        self.title.setText("YoloQt APP - A Person Fall Interface For YoloV8")
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # 放大、缩小、关闭按钮布局
        zoom_layout = QHBoxLayout(self.zoom)
        zoom_layout.addWidget(self.zoom_set)
        zoom_layout.addWidget(self.zoom_in)
        zoom_layout.addWidget(self.zoom_out)
        zoom_layout.addWidget(self.zoom_close)
        zoom_layout.setStretch(0, 1)
        zoom_layout.setStretch(1, 1)
        zoom_layout.setStretch(2, 1)
        zoom_layout.setStretch(3, 1)
        zoom_layout.setContentsMargins(75, 8, 18, 8)
        zoom_layout.setSpacing(5)

        # 右中部分QWidget布局
        c2_center_layout = QHBoxLayout(self.c2_center_widget)
        c2_center_layout.addWidget(self.c2_center_1)
        c2_center_layout.addWidget(self.c2_center_2)
        c2_center_layout.setStretch(0, 8)
        c2_center_layout.setStretch(1, 2)
        c2_center_layout.setContentsMargins(0, 0, 0, 0)
        c2_center_layout.setSpacing(0)

        detection_target_license_fps_model_plays_layout = QVBoxLayout(self.c2_center_1)
        detection_target_license_fps_model_plays_layout.addWidget(self.detection)
        detection_target_license_fps_model_plays_layout.addWidget(self.target_license_fps_model)
        detection_target_license_fps_model_plays_layout.addWidget(self.plays)
        detection_target_license_fps_model_plays_layout.addWidget(self.bar)
        detection_target_license_fps_model_plays_layout.setStretch(0, 1)
        detection_target_license_fps_model_plays_layout.setStretch(1, 20)
        detection_target_license_fps_model_plays_layout.setStretch(2, 56)
        detection_target_license_fps_model_plays_layout.setStretch(3, 4)
        detection_target_license_fps_model_plays_layout.setSpacing(3)
        detection_target_license_fps_model_plays_layout.setContentsMargins(5, -1, 2, 0)

        # 结果显示布局
        target_license_fps_model_plays_layout = QHBoxLayout(self.target_license_fps_model)
        target_license_fps_model_plays_layout.addWidget(self.target)
        target_license_fps_model_plays_layout.addWidget(self.license)
        target_license_fps_model_plays_layout.addWidget(self.fps)
        target_license_fps_model_plays_layout.addWidget(self.model)

        # 目标个数布局
        target_layout = QVBoxLayout(self.target)
        target_layout.addWidget(self.target_top)
        target_layout.addWidget(self.target_bottom)
        target_layout.setStretch(0, 2)
        target_layout.setStretch(1, 3)
        target_layout.setContentsMargins(0, 0, 0, 0)
        target_layout.setSpacing(0)

        # 车牌号布局
        license_layout = QVBoxLayout(self.license)
        license_layout.addWidget(self.license_top)
        license_layout.addWidget(self.license_bottom)
        license_layout.setStretch(0, 2)
        license_layout.setStretch(1, 3)
        license_layout.setContentsMargins(0, 0, 0, 0)
        license_layout.setSpacing(0)

        # 检测速度布局
        fps_layout = QVBoxLayout(self.fps)
        fps_layout.addWidget(self.fps_top)
        fps_layout.addWidget(self.fps_bottom)
        fps_layout.setStretch(0, 2)
        fps_layout.setStretch(1, 3)
        fps_layout.setContentsMargins(0, 0, 0, 0)
        fps_layout.setSpacing(0)

        # 模型布局
        model_layout = QVBoxLayout(self.model)
        model_layout.addWidget(self.model_top)
        model_layout.addWidget(self.model_bottom)
        model_layout.setStretch(0, 2)
        model_layout.setStretch(1, 3)
        model_layout.setContentsMargins(0, 0, 0, 0)
        model_layout.setSpacing(0)

        # 显示画面布局
        plays_layout = QHBoxLayout(self.plays)
        plays_layout.addWidget(self.plays_left)
        plays_layout.addWidget(self.plays_right)
        plays_layout.setStretch(0, 1)
        plays_layout.setStretch(1, 1)
        plays_layout.setSpacing(3)
        plays_layout.setContentsMargins(0, 0, 0, 0)

        # 进度条布局
        bar_layout = QHBoxLayout(self.bar)
        bar_layout.addWidget(self.bar_start)
        bar_layout.addWidget(self.bar_progress)
        bar_layout.addWidget(self.bar_stop)
        bar_layout.setStretch(0, 1)
        bar_layout.setStretch(1, 23)
        bar_layout.setStretch(2, 1)
        bar_layout.setContentsMargins(0, 0, 0, 0)

        welcome_layout = QHBoxLayout(self.c2_bottom_widget)
        welcome_layout.addWidget(self.welcome)
        welcome_layout.setContentsMargins(10, 0, -1, 0)

        # 调节参数布局
        set_layout = QVBoxLayout(self.c2_center_2)
        set_layout.addWidget(self.set_1)
        set_layout.addWidget(self.set_2)
        set_layout.addWidget(self.set_3)
        set_layout.addWidget(self.set_4)
        set_layout.addWidget(self.set_5)
        set_layout.addWidget(self.set_6)
        set_layout.setStretch(0, 1)
        set_layout.setStretch(1, 2)
        set_layout.setStretch(2, 2)
        set_layout.setStretch(3, 2)
        set_layout.setStretch(4, 2)
        set_layout.setStretch(5, 3)
        set_layout.setContentsMargins(-1, -1, -1, 30)
        set_layout.setSpacing(5)

        set_2_layout = QGridLayout(self.set_2)
        set_2_layout.addWidget(self.set_2_logo, 0, 0, 1, 1)
        set_2_layout.addWidget(self.set_2_name, 0, 1, 1, 1)
        set_2_layout.addWidget(self.set_2_box, 1, 0, 1, 2)
        set_2_layout.setColumnStretch(0, 1)
        set_2_layout.setColumnStretch(1, 2)
        set_2_layout.setHorizontalSpacing(25)

        set_3_layout = QGridLayout(self.set_3)
        set_3_layout.addWidget(self.set_3_logo, 0, 0, 1, 1)
        set_3_layout.addWidget(self.set_3_name, 0, 1, 1, 1)
        set_3_layout.addWidget(self.set_3_box, 1, 0, 1, 1)
        set_3_layout.addWidget(self.set_3_slider, 1, 1, 1, 1)
        set_3_layout.setColumnStretch(0, 1)
        set_3_layout.setColumnStretch(1, 2)

        set_4_layout = QGridLayout(self.set_4)
        set_4_layout.addWidget(self.set_4_logo, 0, 0, 1, 1)
        set_4_layout.addWidget(self.set_4_name, 0, 1, 1, 1)
        set_4_layout.addWidget(self.set_4_box, 1, 0, 1, 1)
        set_4_layout.addWidget(self.set_4_slider, 1, 1, 1, 1)

        set_5_layout = QGridLayout(self.set_5)
        set_5_layout.addWidget(self.set_5_logo, 0, 0, 1, 1)
        set_5_layout.addWidget(self.set_5_name, 0, 1, 1, 1)
        set_5_layout.addWidget(self.set_5_box, 1, 0, 1, 1)
        set_5_layout.addWidget(self.set_5_slider, 1, 1, 1, 1)

        set_6_layout = QGridLayout(self.set_6)
        set_6_layout.addWidget(self.set_6_logo, 0, 0, 1, 1)
        set_6_layout.addWidget(self.set_6_name, 0, 1, 1, 2)
        set_6_layout.addWidget(self.set_6_save_img, 1, 0, 1, 1)
        set_6_layout.addWidget(self.set_6_img, 1, 1, 1, 2)
        set_6_layout.addWidget(self.set_6_save_video, 2, 0, 1, 1)
        set_6_layout.addWidget(self.set_6_video, 2, 1, 1, 2)

    # 该函数为初始化过程中所提到的组件设置函数，主要功能如下：
    ### 1、给按钮赋上图标
    ### 2、使得按钮的大小可改变（默认是不允许改变）
    ### 3、连接按钮的信号与执行函数
    def initSet(self):
        pixmap = QPixmap("qss_imgs/logo.png").scaled(55, 55)
        self.logo_left_label.setFixedSize(pixmap.size())
        self.logo_left_label.setPixmap(pixmap)

        self.anim.setDuration(200)
        self.anim.setLoopCount(1)
        self.set_anim_1.setDuration(300)
        self.set_anim_1.setLoopCount(1)
        self.set_anim_2.setDuration(300)
        self.set_anim_2.setLoopCount(1)

        self.hide_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.hide_button.clicked.connect(self.hide_menu)
        pixmap = QPixmap("qss_imgs/menu.png")
        self.hide_left_label.setFixedSize(pixmap.size())
        self.hide_left_label.setPixmap(pixmap)
        self.hide_right_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.file_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        pixmap = QPixmap("qss_imgs/file.png")
        self.file_left_label.setFixedSize(pixmap.size())
        self.file_left_label.setPixmap(pixmap)
        self.file_right_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.file_button.clicked.connect(self.upload_file)
        self.camera_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        pixmap = QPixmap("qss_imgs/cam.png")
        self.camera_left_label.setFixedSize(pixmap.size())
        self.camera_left_label.setPixmap(pixmap)
        self.camera_right_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_button.clicked.connect(self.camera_detection)
        self.rtsp_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        pixmap = QPixmap("qss_imgs/RTSP.png")
        self.rtsp_left_label.setFixedSize(pixmap.size())
        self.rtsp_left_label.setPixmap(pixmap)
        self.rtsp_right_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.zoom_set.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.zoom_set.clicked.connect(self.set_menu)
        self.zoom_in.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.zoom_in.clicked.connect(self.zoom_in_window)
        self.zoom_out.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.zoom_out.clicked.connect(self.zoom_out_window)
        self.zoom_close.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.zoom_close.clicked.connect(self.close)

        self.detection.setText("Detection")

        self.target_top.setText("Target")
        self.target_top.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.target_bottom.setText("--")
        self.target_bottom.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.license_top.setText("Action")
        self.license_top.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.license_bottom.setText("--")
        self.license_bottom.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.fps_top.setText("FPS")
        self.fps_top.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.fps_bottom.setText("--")
        self.fps_bottom.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.model_top.setText("Model")
        self.model_top.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.bar_start.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.bar_start.setCheckable(True)
        self.bar_start.setChecked(False)
        self.bar_progress.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        # 设置进度条的最小值和最大值
        self.bar_progress.setMinimum(0)
        self.bar_progress.setMaximum(100)
        self.bar_stop.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.welcome.setText("Welcome")

        self.set_1.setText("Settings")
        self.set_1.setAlignment(Qt.AlignmentFlag.AlignCenter)

        pixmap = QPixmap("qss_imgs/model.png").scaled(20, 20)
        self.set_2_logo.setFixedSize(pixmap.size())
        self.set_2_logo.setPixmap(pixmap)
        self.set_2_name.setText("Model")
        self.set_2_name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        for item in os.listdir("weights"):
            self.set_2_box.addItem(item)
        self.model_bottom.setText(self.set_2_box.currentText())
        self.model_bottom.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.set_2_box.currentIndexChanged.connect(self.set_2_box_to_model)

        pixmap = QPixmap("qss_imgs/IOU.png").scaled(20, 20)
        self.set_3_logo.setFixedSize(pixmap.size())
        self.set_3_logo.setPixmap(pixmap)
        self.set_3_name.setText("IOU")
        self.set_3_name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # 设置最小值、最大值和步进值
        self.set_3_box.setMinimum(0.01)
        self.set_3_box.setMaximum(1.00)
        self.set_3_box.setSingleStep(0.01)
        self.set_3_box.setValue(0.45)
        self.set_3_box.valueChanged.connect(self.set_3_box_to_slider_values)
        self.set_3_slider.setMinimum(1)
        self.set_3_slider.setMaximum(100)
        self.set_3_slider.setValue(45)
        self.set_3_slider.setSingleStep(1)
        self.set_3_slider.valueChanged.connect(self.set_3_slider_to_box_values)
        self.set_3_slider.setOrientation(Qt.Orientation.Horizontal)

        pixmap = QPixmap("qss_imgs/conf.png").scaled(20, 20)
        self.set_4_logo.setFixedSize(pixmap.size())
        self.set_4_logo.setPixmap(pixmap)
        self.set_4_name.setText("Conf")
        self.set_4_name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # 设置最小值、最大值和步进值
        self.set_4_box.setMinimum(0.01)
        self.set_4_box.setMaximum(1.00)
        self.set_4_box.setSingleStep(0.01)
        self.set_4_box.setValue(0.25)
        self.set_4_box.valueChanged.connect(self.set_4_box_to_slider_values)
        self.set_4_slider.setMinimum(1)
        self.set_4_slider.setMaximum(100)
        self.set_4_slider.setValue(25)
        self.set_4_slider.setSingleStep(1)
        self.set_4_slider.valueChanged.connect(self.set_4_slider_to_box_values)
        self.set_4_slider.setOrientation(Qt.Orientation.Horizontal)

        pixmap = QPixmap("qss_imgs/delay.png").scaled(20, 20)
        self.set_5_logo.setFixedSize(pixmap.size())
        self.set_5_logo.setPixmap(pixmap)
        self.set_5_name.setText("Delay(ms)")
        self.set_5_name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # 设置最小值、最大值和步进值
        self.set_5_box.setMinimum(0)
        self.set_5_box.setMaximum(60)
        self.set_5_box.setSingleStep(1)
        self.set_5_box.setValue(10)
        self.set_5_box.valueChanged.connect(self.set_5_box_to_slider_values)
        self.set_5_slider.setMinimum(0)
        self.set_5_slider.setMaximum(60)
        self.set_5_slider.setValue(10)
        self.set_5_slider.setSingleStep(1)
        self.set_5_slider.valueChanged.connect(self.set_5_slider_to_box_values)
        self.set_5_slider.setOrientation(Qt.Orientation.Horizontal)

        pixmap = QPixmap("qss_imgs/save.png").scaled(20, 20)
        self.set_6_logo.setFixedSize(pixmap.size())
        self.set_6_logo.setPixmap(pixmap)
        self.set_6_name.setText("Save")
        self.set_6_name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.set_6_save_img.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.set_6_save_img.setCheckable(True)
        self.set_6_save_img.setChecked(False)
        self.set_6_save_img.clicked.connect(self.save_file_img)
        self.set_6_img.setText("Save MP4/IPG")
        self.set_6_save_video.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.set_6_save_video.setCheckable(True)
        self.set_6_save_video.setChecked(False)
        self.set_6_save_video.clicked.connect(self.save_file_video)
        self.set_6_video.setText("Save labels(.txt)")

        self.bar_start.clicked.connect(self.detection_file)

    ### 该函数为初始化中所提到的美化界面函数
    def initUi(self):
        self.center_widget.setStyleSheet(
            """
            background:  qlineargradient(x0:0, y0:1, x1:1, y1:1,stop:0.4  rgb(107, 128, 210), stop:1 rgb(180, 140, 255));
            border-top-left-radius:{0}px;
            border-bottom-left-radius:{0}px;
            border-top-right-radius:{0}px;
            border-bottom-right-radius:{0}px;
            """.format(30)
        )

        self.c1_widget.setStyleSheet(
            """
            background-color: transparent;
            border-top-left-radius:{0}px;
            border-bottom-left-radius:{0}px;
            border-top-right-radius:{0}px;
            border-bottom-right-radius:{0}px;
            """.format(30)
        )
        self.logo_label.setStyleSheet(
            """
            background-color: transparent;
            """
        )
        self.logo_left_label.setStyleSheet(
            """

            border: 2px solid white;
            background-color: transparent;
            border-top-left-radius:{0}px;
            border-bottom-left-radius:{0}px;
            border-top-right-radius:{0}px;
            border-bottom-right-radius:{0}px;
            """.format(20)
        )
        self.logo_right_label.setStyleSheet(
            """
            background-color: transparent;
            font-size: 10px;
            font-style: italic;
            color: white;
            """
        )
        self.hide_button.setStyleSheet(
            """
            QPushButton {
                background-color: transparent;
            }
            QPushButton:hover {
                background-color: rgba(114, 129, 214, 59);
            }
            """
        )
        self.hide_left_label.setStyleSheet(
            """
            background-color: transparent;
            """
        )
        self.hide_right_label.setStyleSheet(
            """
            background-color: transparent;
            font-size: 14px;
            font-style: italic;
            color: white;
            """
        )
        self.file_button.setStyleSheet(
            """
            QPushButton {
                background-color: transparent;
            }
            QPushButton:hover {
                background-color: rgba(114, 129, 214, 59);
            }
            """
        )
        self.file_left_label.setStyleSheet(
            """
            background-color: transparent;
            """
        )
        self.file_right_label.setStyleSheet(
            """
            background-color: transparent;
            font-size: 14px;
            font-style: italic;
            color: white;
            """
        )
        self.camera_button.setStyleSheet(
            """
            QPushButton {
                background-color: transparent;
            }
            QPushButton:hover {
                background-color: rgba(114, 129, 214, 59);
            }
            """
        )
        self.camera_left_label.setStyleSheet(
            """
            background-color: transparent;
            """
        )
        self.camera_right_label.setStyleSheet(
            """
            background-color: transparent;
            font-size: 14px;
            font-style: italic;
            color: white;
            """
        )
        self.rtsp_button.setStyleSheet(
            """
            QPushButton {
                background-color: transparent;
            }
            QPushButton:hover {
                background-color: rgba(114, 129, 214, 59);
            }
            """
        )
        self.rtsp_left_label.setStyleSheet(
            """
            background-color: transparent;
            """
        )
        self.rtsp_right_label.setStyleSheet(
            """
            background-color: transparent;
            font-size: 14px;
            font-style: italic;
            color: white;
            """
        )
        self.version_label.setStyleSheet(
            """
            background-color: transparent;
            font-size: 12px;
            font-style: italic;
            color: white;
            """
        )
        self.c2_widget.setStyleSheet(
            """
            background: rgb(255, 255, 255);
            border-top-left-radius:{0}px;
            border-bottom-left-radius:{0}px;
            border-top-right-radius:{0}px;
            border-bottom-right-radius:{0}px;
            """.format(30)
        )
        self.c2_top_widget.setStyleSheet(
            """
            background: transparent;
            border-top-right-radius:{0}px;
            border-bottom-right-radius:{0}px;
            """.format(5)
        )
        self.title.setStyleSheet(
            """
            background: transparent;
            font-size: 14px;
            font-style: italic;
            font-weight: bold;
            color: black;
            """
        )
        self.zoom.setStyleSheet(
            """
            background: transparent;
            """
        )
        self.zoom_set.setStyleSheet(
            """
            border-image: url(qss_imgs/set.png);
            background-color: transparent;
            """
        )
        self.zoom_in.setStyleSheet(
            """
            border-image: url(qss_imgs/绿圆.png);
            background-color: transparent;
            """
        )
        self.zoom_out.setStyleSheet(
            """
            border-image: url(qss_imgs/黄圆.png);
            background-color: transparent;
            """
        )
        self.zoom_close.setStyleSheet(
            """
            border-image: url(qss_imgs/红圆.png);
            background-color: transparent;
            """
        )
        self.c2_center_widget.setStyleSheet(
            """
            background: transparent;;
            border-top-left-radius:{0}px;
            border-bottom-left-radius:{0}px;
            border-top-right-radius:{0}px;
            border-bottom-right-radius:{0}px;
            """.format(0)
        )
        self.c2_center_1.setStyleSheet(
            """
            background: transparent;;
            border-top-left-radius:{0}px;
            border-bottom-left-radius:{0}px;
            border-top-right-radius:{0}px;
            border-bottom-right-radius:{0}px;
            """.format(30)
        )
        self.detection.setStyleSheet(
            """
            background: transparent;
            font-size: 14px;
            font-style: italic;
            color: black;
            """
        )
        self.target_license_fps_model.setStyleSheet(
            """
            background: rgb(238, 242, 255);
            border-top-left-radius:{0}px;
            border-bottom-left-radius:{0}px;
            border-top-right-radius:{0}px;
            border-bottom-right-radius:{0}px;
            """.format(15)
        )
        self.target.setStyleSheet(
            """
            background-color: qradialgradient(cx:0, cy:0, radius:1, fx:0.1, fy:0.1, stop:0 rgb(162, 129, 247),  stop:1 rgb(119, 111, 252));
            order-top-left-radius:{0}px;
            border-bottom-left-radius:{0}px;
            border-top-right-radius:{0}px;
            border-bottom-right-radius:{0}px;
            """.format(10)
        )
        self.target_top.setStyleSheet(
            """
            background-color: transparent;
            font-size: 14px;
            font-style: italic;
            color: white;
            border-bottom: 2px solid white;
            border-bottom-left-radius:{0}px;;
            border-bottom-right-radius:{0}px;
            """.format(0)
        )
        self.target_bottom.setStyleSheet(
            """
            background-color: transparent;
            font-size: 13px;
            font-style: italic;
            color: white;
            """
        )
        self.license.setStyleSheet(
            """
            background-color: qradialgradient(cx:0, cy:0, radius:1, fx:0.1, fy:0.1, stop:0 rgb(253, 139, 133),  stop:1 rgb(248, 194, 152));
            order-top-left-radius:{0}px;
            border-bottom-left-radius:{0}px;
            border-top-right-radius:{0}px;
            border-bottom-right-radius:{0}px;
            """.format(10)
        )
        self.license_top.setStyleSheet(
            """
            background-color: transparent;
            font-size: 14px;
            font-style: italic;
            color: white;
            border-bottom: 2px solid white;
            border-bottom-left-radius:{0}px;;
            border-bottom-right-radius:{0}px;
            """.format(0)
        )
        self.license_bottom.setStyleSheet(
            """
            background-color: transparent;
            font-size: 13px;
            font-style: italic;
            color: white;
            """
        )
        self.fps.setStyleSheet(
            """
            background-color: qradialgradient(cx:0, cy:0, radius:1, fx:0.1, fy:0.1, stop:0 rgb(243, 175, 189),  stop:1 rgb(155, 118, 218));
            order-top-left-radius:{0}px;
            border-bottom-left-radius:{0}px;
            border-top-right-radius:{0}px;
            border-bottom-right-radius:{0}px;
            """.format(10)
        )
        self.fps_top.setStyleSheet(
            """
            background-color: transparent;
            font-size: 14px;
            font-style: italic;
            color: white;
            border-bottom: 2px solid white;
            border-bottom-left-radius:{0}px;;
            border-bottom-right-radius:{0}px;
            """.format(0)
        )
        self.fps_bottom.setStyleSheet(
            """
            background-color: transparent;
            font-size: 13px;
            font-style: italic;
            color: white;
            """
        )
        self.model.setStyleSheet(
            """
            background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #59969b, stop:1 #04e7fa);
            order-top-left-radius:{0}px;
            border-bottom-left-radius:{0}px;
            border-top-right-radius:{0}px;
            border-bottom-right-radius:{0}px;
            """.format(10)
        )
        self.model_top.setStyleSheet(
            """
            background-color: transparent;
            font-size: 14px;
            font-style: italic;
            color: white;
            border-bottom: 2px solid white;
            border-bottom-left-radius:{0}px;;
            border-bottom-right-radius:{0}px;
            """.format(0)
        )
        self.model_bottom.setStyleSheet(
            """
            background-color: transparent;
            font-size: 13px;
            font-style: italic;
            color: white;
            """
        )
        self.plays.setStyleSheet(
            """
            background-color: transparent;
            border-top-left-radius:{0}px;
            border-bottom-left-radius:{0}px;
            border-top-right-radius:{0}px;
            border-bottom-right-radius:{0}px;
            """.format(15)
        )
        self.plays_left.setStyleSheet(
            """
            background-color: rgb(238, 242, 255);
            border-top-left-radius:{0}px;
            border-bottom-left-radius:{0}px;
            border-top-right-radius:{0}px;
            border-bottom-right-radius:{0}px;
            """.format(10)
        )
        self.plays_right.setStyleSheet(
            """
            background-color: rgb(238, 242, 255);
            border-top-left-radius:{0}px;
            border-bottom-left-radius:{0}px;
            border-top-right-radius:{0}px;
            border-bottom-right-radius:{0}px;
            """.format(10)
        )
        self.bar.setStyleSheet(
            """
            background-color: transparent;
            """
        )
        self.bar_start.setStyleSheet(
            """
            background-color: transparent;
            background-image: url("qss_imgs/begin.png"); 
            """
        )
        self.bar_progress.setStyleSheet(
            """
            QProgressBar {
            border: 2px solid gray;
            border-radius: 5px;
            text-align: center;
            }
            QProgressBar::chunk {
            background-color: orange;
            border-radius: 5px;
            }
            """
        )
        self.bar_stop.setStyleSheet(
            """
            background-color: transparent;
            background-image: url("qss_imgs/stop.png"); 
            """
        )
        self.c2_center_2.setStyleSheet(
            """
            background: qradialgradient(cx:0, cy:0, radius:1, fx:0.1, fy:0.1, stop:0 rgb(243, 175, 189),  stop:1 rgb(155, 118, 218));
            border-top-left-radius:{0}px;
            border-bottom-left-radius:{0}px;
            """.format(20)
        )
        self.set_1.setStyleSheet(
            """
            background: transparent;
            font-size: 14px;
            font-style: italic;
            font-weight: bold;
            color: white;
            """
        )
        self.set_2.setStyleSheet(
            """
            background: transparent;
            border: 2px solid white;
            border-top-left-radius:{0}px;
            border-bottom-left-radius:{0}px;
            border-top-right-radius:{0}px;
            border-bottom-right-radius:{0}px;
            """.format(10)
        )
        self.set_2_logo.setStyleSheet(
            """
            background: transparent;
            border: 0px;
            """
        )
        self.set_2_name.setStyleSheet(
            """
            background: transparent;
            border: 0px;
            font-size: 14px;
            font-weight: bold;
            color: white;
            """
        )
        self.set_2_box.setStyleSheet(
            """
            background-color: rgba(255, 255, 255, 90);
            color: rgba(0, 0, 0, 140);
            border: 0px;
            """
        )
        self.set_3.setStyleSheet(
            """
            background: transparent;
            border: 2px solid white;
            border-top-left-radius:{0}px;
            border-bottom-left-radius:{0}px;
            border-top-right-radius:{0}px;
            border-bottom-right-radius:{0}px;
            """.format(10)
        )
        self.set_3_logo.setStyleSheet(
            """
            background: transparent;
            border: 0px;
            """
        )
        self.set_3_name.setStyleSheet(
            """
            background: transparent;
            border: 0px;
            font-size: 14px;
            font-weight: bold;
            color: white;
            """
        )
        self.set_3_box.setStyleSheet(
            """
            QDoubleSpinBox {
            background-color: rgba(255,255,255,90);
            border: 0px solid lightgray;
            border-radius: 2px;
            font: 600 9pt \"Segoe UI\";
            }
            QDoubleSpinBox::up-button{
            width: 10px;
            height: 9px;
            margin: 0px 3px 0px 0px;
            border-image: url(qss_imgs/box_up.png);
            }
            QDoubleSpinBox::up-button:pressed{
            margin-top: 1px;
            }
            QDoubleSpinBox::down-button{
            width: 10px;
            height: 9px;
            margin: 0px 3px 0px 0px;
            border-image: url(qss_imgs/box_down.png);
            }
            QDoubleSpinBox::down-button:pressed{
            margin-bottom: 1px;
            }
            """
        )
        self.set_3_slider.setStyleSheet(
            """
            QSlider{
            border: 0px;
            }
            QSlider::groove:horizontal{
            background-color: rgba(255,255,255,90);
            border-radius: 5px;
            border: none;
            height: 10px;
            }
            QSlider::handle:horizontal{
            background-color: white;
            width: 10px;
            border-radius: 3px;
            margin: -1px 0px -1px 0px;
            }
            QSlider::sub-page:horizontal{
            background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #59969b, stop:1 #04e7fa);
            border-radius: 5px;
            }
            """
        )
        self.set_4.setStyleSheet(
            """
            background: transparent;
            border: 2px solid white;
            border-top-left-radius:{0}px;
            border-bottom-left-radius:{0}px;
            border-top-right-radius:{0}px;
            border-bottom-right-radius:{0}px;
            """.format(10)
        )
        self.set_4_logo.setStyleSheet(
            """
            background: transparent;
            border: 0px;
            """
        )
        self.set_4_name.setStyleSheet(
            """
            background: transparent;
            border: 0px;
            font-size: 14px;
            font-weight: bold;
            color: white;
            """
        )
        self.set_4_box.setStyleSheet(
            """
            QDoubleSpinBox {
            background-color: rgba(255,255,255,90);
            border: 0px solid lightgray;
            border-radius: 2px;
            font: 600 9pt \"Segoe UI\";
            }
            QDoubleSpinBox::up-button{
            width: 10px;
            height: 9px;
            margin: 0px 3px 0px 0px;
            border-image: url(qss_imgs/box_up.png);
            }
            QDoubleSpinBox::up-button:pressed{
            margin-top: 1px;
            }
            QDoubleSpinBox::down-button{
            width: 10px;
            height: 9px;
            margin: 0px 3px 0px 0px;
            border-image: url(qss_imgs/box_down.png);
            }
            QDoubleSpinBox::down-button:pressed{
            margin-bottom: 1px;
            }
            """
        )
        self.set_4_slider.setStyleSheet(
            """
            QSlider{
            border: 0px;
            }
            QSlider::groove:horizontal{
            background-color: rgba(255,255,255,90);
            border-radius: 5px;
            border: none;
            height: 10px;
            }
            QSlider::handle:horizontal{
            background-color: white;
            width: 10px;
            border-radius: 3px;
            margin: -1px 0px -1px 0px;
            }
            QSlider::sub-page:horizontal{
            background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #59969b, stop:1 #04e7fa);
            border-radius: 5px;
            }
            """
        )
        self.set_5.setStyleSheet(
            """
            background: transparent;
            border: 2px solid white;
            border-top-left-radius:{0}px;
            border-bottom-left-radius:{0}px;
            border-top-right-radius:{0}px;
            border-bottom-right-radius:{0}px;
            """.format(10)
        )
        self.set_5_logo.setStyleSheet(
            """
            background: transparent;
            border: 0px;
            """
        )
        self.set_5_name.setStyleSheet(
            """
            background: transparent;
            border: 0px;
            font-size: 14px;
            font-weight: bold;
            color: white;
            """
        )
        self.set_5_box.setStyleSheet(
            """
            QSpinBox {
            background-color: rgba(255,255,255,90);
            border: 0px solid lightgray;
            border-radius: 2px;
            font: 600 9pt \"Segoe UI\";
            width: 25px;
            }
            QSpinBox::up-button{
            width: 10px;
            height: 9px;
            margin: 0px 3px 0px 0px;
            border-image: url(qss_imgs/box_up.png);
            }
            QSpinBox::up-button:pressed{
            margin-top: 1px;
            }
            QSpinBox::down-button{
            width: 10px;
            height: 9px;
            margin: 0px 3px 0px 0px;
            border-image: url(qss_imgs/box_down.png);
            }
            QSpinBox::down-button:pressed{
            margin-bottom: 1px;
            }
            """
        )
        self.set_5_slider.setStyleSheet(
            """
            QSlider{
            border: 0px;
            }
            QSlider::groove:horizontal{
            background-color: rgba(255,255,255,90);
            border-radius: 5px;
            border: none;
            height: 10px;
            }
            QSlider::handle:horizontal{
            background-color: white;
            width: 10px;
            border-radius: 3px;
            margin: -1px 0px -1px 0px;
            }
            QSlider::sub-page:horizontal{
            background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #59969b, stop:1 #04e7fa);
            border-radius: 5px;
            }
            """
        )
        self.set_6.setStyleSheet(
            """
            background: transparent;
            border: 2px solid white;
            border-top-left-radius:{0}px;
            border-bottom-left-radius:{0}px;
            border-top-right-radius:{0}px;
            border-bottom-right-radius:{0}px;
            """.format(10)
        )
        self.set_6_logo.setStyleSheet(
            """
            background: transparent;
            border: 0px;
            """
        )
        self.set_6_name.setStyleSheet(
            """
            background: transparent;
            border: 0px;
            font-size: 14px;
            font-weight: bold;
            color: white;
            """
        )
        self.set_6_save_img.setStyleSheet(
            """
            background: transparent;
            border-image: url(qss_imgs/check_no.png);
            border: 0px;
            """
        )
        self.set_6_img.setStyleSheet(
            """
            background: transparent;
            border: 0px;
            font-size: 12px;
            color: white;
            """
        )
        self.set_6_save_video.setStyleSheet(
            """
            background: transparent;
            border-image: url(qss_imgs/check_no.png);
            border: 0px;
            """
        )
        self.set_6_video.setStyleSheet(
            """
            background: transparent;
            border: 0px;
            font-size: 12px;
            color: white;
            """
        )
        self.c2_bottom_widget.setStyleSheet(
            """
            background: transparent;
            """
        )
        self.welcome.setStyleSheet(
            """
            background: transparent;
            border: 0px;
            font-size: 12px;
            font-style: italic;
            """
        )

    # 实现缩小界面功能，与缩小按钮连接
    def zoom_in_window(self):
        self.showMinimized()

    def zoom_out_window(self):
        try:
            if not self.is_maximized:
                # 将窗口大小设置为屏幕大小
                self.showMaximized()
                self.center_widget.resize(self.width(), self.height())

                pixmap = QPixmap("qss_imgs/pause.png").scaled(30, 30)
                self.bar_stop.setFixedSize(pixmap.size())
                self.bar_stop.setPixmap(pixmap)

                self.bar_start.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
                self.bar_start.resize(25, 5)
                pixmap = QPixmap("qss_imgs/begin.png").scaled(25, 5)
                icon = QIcon(pixmap)
                self.bar_start.setIcon(icon)
                self.bar_start.setIconSize(pixmap.size())

            else:
                # 恢复到初始大小
                self.showNormal()
                self.resize(self.WIDTH, self.HEIGHT)
                self.center_widget.resize(self.WIDTH, self.HEIGHT)
            self.is_maximized = not self.is_maximized

        except Exception as e:
            print(f"放大窗口时出现错误: {e}")

    # 隐藏菜单栏的过渡动画
    def hide_menu(self):
        num = 64  # 设置过渡动画的移动距离
        if self.anim.state() == QAbstractAnimation.State.Running:  # 判断动画是否结束
            return
        if self.hide_account % 2 == 1:  # 判断动画是该向左还是向右
            # 动画是向右
            final = QRect(self.c2_widget.geometry().x() + num, self.c2_widget.geometry().y(),
                          self.c2_widget.width() - num, self.c2_widget.height())
            self.anim.setEndValue(final)
            self.anim.start()  # 开始动画
            while True:
                if self.anim.state() == QAbstractAnimation.State.Running:
                    break
        else:
            # 动画是向左
            final = QRect(self.c2_widget.geometry().x() - num, self.c2_widget.geometry().y(),
                          self.c2_widget.width() + num, self.c2_widget.height())
            self.anim.setEndValue(final)
            self.anim.start()  # 动画开始
            while True:
                if self.anim.state() == QAbstractAnimation.State.Running:
                    break
        self.hide_account += 1

    # 调节参数的过渡动画
    def set_menu(self):
        num = 183  # 设置过渡动画的移动距离
        if self.parallel_group.state() == QAbstractAnimation.State.Running:
            return
        if self.set_account % 2 == 1:  # 判断动画是否结束
            # 动画是向右
            final_1 = QRect(self.c2_center_1.geometry().x(), self.c2_center_1.geometry().y(),
                            self.c2_center_1.width() + num, self.c2_center_1.height())

            final_2 = QRect(self.c2_center_2.geometry().x() + num, self.c2_center_2.geometry().y(),
                            self.c2_center_2.width() - num, self.c2_center_2.height())
            self.set_anim_1.setEndValue(final_1)
            self.set_anim_2.setEndValue(final_2)
            self.parallel_group.start()
            while True:
                if self.parallel_group.state() == QAbstractAnimation.State.Running:
                    break
        else:
            # 动画是向左
            final_1 = QRect(self.c2_center_1.geometry().x(), self.c2_center_1.geometry().y(),
                            self.c2_center_1.width() - num, self.c2_center_1.height())
            final_2 = QRect(self.c2_center_2.geometry().x() - num, self.c2_center_2.geometry().y(),
                            self.c2_center_2.width() + num, self.c2_center_2.height())
            self.set_anim_1.setEndValue(final_1)
            self.set_anim_2.setEndValue(final_2)
            self.parallel_group.start()
            while True:
                if self.parallel_group.state() == QAbstractAnimation.State.Running:
                    break
        self.set_account += 1

    # 加载模型
    def set_2_box_to_model(self):
        self.model_bottom.setText(self.set_2_box.currentText())
        self.clear()
        self.loadModel()  ## 加载
        self.file_path = None

    # 实现iou的实时控制
    def set_3_box_to_slider_values(self):
        value = self.set_3_box.value()
        self.set_3_slider.setValue(int(value * 100))  # 将浮点数转换为整数进行设置
        self.time_reset()

    def set_3_slider_to_box_values(self):
        value = self.set_3_slider.value()
        self.set_3_box.setValue(value * 1.0 / 100)  # 将浮点数转换为整数进行设置
        self.time_reset()

    # 实现conf的实时控制
    def set_4_box_to_slider_values(self):
        value = self.set_4_box.value()
        self.set_4_slider.setValue(int(value * 100))  # 将浮点数转换为整数进行设置
        self.time_reset()

    def set_4_slider_to_box_values(self):
        value = self.set_4_slider.value()
        self.set_4_box.setValue(value * 1.0 / 100)  # 将浮点数转换为整数进行设置
        self.time_reset()

    # 实现delay的实时控制
    def set_5_box_to_slider_values(self):
        value = self.set_5_box.value()
        self.set_5_slider.setValue(int(value))  # 将浮点数转换为整数进行设置
        self.time_reset()

    def set_5_slider_to_box_values(self):
        value = self.set_5_slider.value()
        self.set_5_box.setValue(int(value))  # 将浮点数转换为整数进行设置
        self.time_reset()

    def get_file_type(self, file_path):
        """
        获取文件路径的类型
        :param file_path: 文件路径
        :return: 文件类型，'image' 表示图片，'video' 表示视频，'unknown' 表示未知类型
        """
        if file_path is None:
            return 'file_path is None'
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif']  # 图片文件的扩展名
        video_extensions = ['.mp4', '.avi', '.mov', '.wmv', '.mkv', '.MP4']  # 视频文件的扩展名
        file_extension = file_path[file_path.rfind('.'):].lower()  # 获取文件路径的扩展名，并转换为小写

        if os.path.isdir(file_path):
            return "dir"
        elif file_extension in image_extensions:
            return 'image'
        elif file_extension in video_extensions:
            return 'video'
        else:
            return 'unknown'

    # # 实现保存按钮的开始、关闭的实时显示
    def save_file_img(self):
        if self.set_6_save_img.isChecked():
            self.set_6_save_img.setStyleSheet(
                """
                background: transparent;
                border-image: url(qss_imgs/check_yes.png);
                border: 0px;
                """
            )
            self.save_imgVideo_flag = True
        else:
            self.set_6_save_img.setStyleSheet(
                """
                background: transparent;
                border-image: url(qss_imgs/check_no.png);
                border: 0px;
                """
            )
            self.save_imgVideo_flag = False

    # 用于判断是否需要保存视频
    def save_file_video(self):
        if self.set_6_save_video.isChecked():
            self.set_6_save_video.setStyleSheet(
                """
                background: transparent;
                border-image: url(qss_imgs/check_yes.png);
                border: 0px;
                """
            )
            self.save_label_flag = True
        else:
            self.set_6_save_video.setStyleSheet(
                """
                background: transparent;
                border-image: url(qss_imgs/check_no.png);
                border: 0px;
                """
            )
            self.save_label_flag = False

    # 用于暂停视频播放，方便调整iou、conf
    def time_reset(self):
        if self.timer is not None:
            self.timer.stop()  # 暂停播放
            self.iou = self.set_3_box.value()  # 调整iou
            self.conf = self.set_4_box.value()  # 调整conf
            self.delay = int(self.set_5_box.value())  # 调整delay
            if self.camera_flag:
                self.delay = 100
            self.timer.start(self.delay)
        else:
            self.iou = self.set_3_box.value()  # 调整iou
            self.conf = self.set_4_box.value()  # 调整conf
            self.delay = int(self.set_5_box.value())  # 调整delay
            if self.camera_flag:
                self.delay = 100
            return

    # 模型加载函数
    def loadModel(self):
        model_name = self.set_2_box.currentText()
        self.detect_thread.det_model = YOLO("weights/" + model_name)  # 加载yolov8检测模型,best.pt是训练好的，你也可以自己训练一个，train.py

    ### 检测、识别函数（只要功能为检查按钮的组件样式变换）
    def detection_file(self):
        if self.file_path is None and not self.camera_flag:
            return
        if self.detect_thread.det_model is None:  # 若模型为None则加载模型
            self.welcome.setText("Loading Model")
            self.loadModel()
        if self.iou is None or self.conf is None:  # 若参数为None则加载参数
            self.iou = self.set_3_box.value()
            self.conf = self.set_4_box.value()
        if self.camera_flag:  # 判断是否用摄像头识别
            if self.bar_start_account % 2 == 1:
                self.bar_start.setStyleSheet(
                    """
                    background-color: transparent;
                    background-image: url("qss_imgs/begin.png"); 
                    """
                )
                self.timer.stop()
                self.welcome.setText("Detection stop")
                if self.video_writer is not None:
                    self.video_writer.release()
                    self.video_writer = None
                    print("文件保存在save/videos/{}".format(self.camera_path))
                    # 视频保存路径和文件名
                    now = datetime.datetime.now()
                    datetime_string = now.strftime("%Y-%m-%d-%H-%M-%S")
                    self.camera_path = "save/videos/" + datetime_string + ".mp4"
            else:
                self.bar_start.setStyleSheet(
                    """
                    background-color: transparent;
                    background-image: url("qss_imgs/pause.png"); 
                    """
                )
                if self.timer is not None:
                    self.timer.start(self.delay)
                else:
                    print("error")
                self.welcome.setText("Detection ing")
            self.bar_start_account += 1

        if not self.camera_flag and self.get_file_type(self.file_path) == "image":
            image = cv2.imread(self.file_path)
            self.welcome.setText("Detection ing")
            self.bar_start.setStyleSheet(
                """
                background-color: transparent;
                background-image: url("qss_imgs/pause.png"); 
                """
            )
            self.show_image(image)

        elif not self.camera_flag and self.get_file_type(self.file_path) == "video":
            if self.bar_start_account % 2 == 1:
                if self.timer is None:
                    self.cap = cv2.VideoCapture(self.file_path)
                    self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    self.delay = int(self.set_5_box.value())  # 计算帧间延迟时间
                    self.timer = QTimer()
                    self.timer.timeout.connect(self.display_next_frame)
                self.bar_start.setStyleSheet(
                    """
                    background-color: transparent;
                    background-image: url("qss_imgs/pause.png"); 
                    """
                )
                self.timer.start(self.delay)
                self.welcome.setText("Detection ing")
            else:
                self.timer.stop()
                self.bar_start.setStyleSheet(
                    """
                    background-color: transparent;
                    background-image: url("qss_imgs/begin.png"); 
                    """
                )
                self.welcome.setText("Detection stop")
            self.bar_start_account += 1

    # 摄像头的实时检测
    def camera_detection(self):
        self.bar_progress.reset()  # 初始化进度条（因为摄像头不要进度条）
        if self.timer is not None:  # 要是之前有视频检测，清除之前的记录
            self.timer.stop()
            self.cap.release()
            self.timer = None
            self.cap = None
            self.plays_left.clear()
            self.plays_right.clear()
        if self.detect_thread.det_model is None:
            self.welcome.setText("Loading Model")
            self.loadModel()
        if self.iou is None or self.conf is None:
            self.iou = self.set_3_box.value()
            self.conf = self.set_4_box.value()

        # 获取选择的设备名称
        self.cap = cv2.VideoCapture()
        self.camera_flag = self.cap.open(0)
        # 视频保存路径和文件名
        now = datetime.datetime.now()
        datetime_string = now.strftime("%Y-%m-%d-%H-%M-%S")
        self.camera_path = "save/videos/" + datetime_string + ".mp4"

        if self.camera_flag:
            self.timer = QTimer()
            self.timer.timeout.connect(self.display_next_frame)
            self.delay = 100
            self.timer.start(self.delay)
            self.bar_start.setStyleSheet(
                """
                background-color: transparent;
                background-image: url("qss_imgs/pause.png"); 
                """
            )
            self.welcome.setText("Detection ing")

    # 在QLabel显示图片
    def show_image(self, img):
        self.detect_thread.set_param(img, self.iou, self.conf)
        self.detect_thread.start()

    def detect_finished(self):
        deImage, fps, license_names = self.detect_thread.results
        fps = float(fps)
        # 线程初始化
        if self.thread_1 is not None:  #
            self.thread_1.deleteLater()
            self.thread_2.deleteLater()
        self.thread_1 = WorkerThread(self.plays_left, self.detect_thread.img)
        self.thread_2 = WorkerThread(self.plays_right, deImage)
        self.thread_1.start()
        self.thread_2.start()
        self.target_bottom.setText(f"{len(license_names)}")
        if len(license_names) > 0:
            self.license_bottom.setText(license_names[0])
        else:
            self.license_bottom.setText("--")
        # 显示检测fps
        self.fps_bottom.setText(f"FPS: {fps:.2f}")
        if not self.camera_flag:
            if self.get_file_type(self.file_path) == "image":
                if self.save_imgVideo_flag:
                    if not os.path.exists("./save/images"):
                        os.makedirs("./save/images")
                    cv2.imwrite("./save/images/save_{}".format(os.path.basename(self.file_path)), deImage)
                    print("图片已保存在save/images/文件下 save_{}".format(os.path.basename(self.file_path)))
                self.welcome.setText("Detection Completed")
                self.bar_start.setStyleSheet(
                    """
                    background-color: transparent;
                    background-image: url("qss_imgs/begin.png");
                    """
                )
                self.bar_progress.setValue(100)
            elif self.get_file_type(self.file_path) == "video":
                if self.save_imgVideo_flag:
                    self.video_img_list.append(deImage)

    # 将image格式转化为QImage，用于在QLabel显示
    def convert_frame_to_image(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, ch = frame.shape
        bytes_per_line = ch * width
        image = QImage(frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        return image

    # 用于不断将视频的帧图片显示，以达到视频播放的效果
    def display_next_frame(self):
        ret, frame = self.cap.read()
        self.ret = ret
        # 获取原始视频的帧率
        origin_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.camera_flag:
            current_frame = -1
        else:
            current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.bar_progress.setValue(int(current_frame * 1.0 / self.frame_count * 100))  # 进度条实现显示进度
        if not ret:  # 判断视频是否播放、检测完毕
            # self.cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 0)
            self.welcome.setText("Detection Completed")  # 显示检测完成
            self.bar_start.setStyleSheet(
                """
                background-color: transparent;
                background-image: url("qss_imgs/begin.png"); 
                """
            )
            # 检测完后，相关参数初始化
            self.timer.stop()
            self.timer = None
            self.cap.release()
            self.cap = None
            self.bar_start_account = 1
            self.camera_flag = False
            if self.save_imgVideo_flag:
                if not os.path.exists("./save/videos"):
                    os.makedirs("./save/videos")
                save_path = "save/videos/save_" + os.path.basename(self.file_path)
                xiu_fps = int(origin_fps * len(self.video_img_list) / self.frame_count)
                self.video_save_thread.set_params(self.video_img_list, save_path, xiu_fps)
                self.video_save_thread.start()
            # if self.video_writer is not None: # 判断是否保存图片
            #     self.video_writer.release()
            #     self.video_writer = None
            #     print("视频已保存在save/videos/文件下 save_{}".format(os.path.basename(self.file_path)))
        else:
            if frame is not None:  # 判断是否正确读取视频的帧图片
                if self.save_imgVideo_flag and current_frame == 1:
                    self.video_img_list = []
                self.show_image(frame)  # 显示图片

    # 上传文件按钮，即从电脑本地读取图片、视频
    def upload_file(self):
        # 打开文件对话框选择图片文件
        file_dialog = QFileDialog()
        path, _ = file_dialog.getOpenFileName(self, "选择文件", "",
                                              "图像文件 (*.png *.jpg *.jpeg *.mp4 *.MP4 *.avi *.mkv)")
        if path:
            self.file_path = path
            self.clear()
            if self.get_file_type(path) == 'image' or self.get_file_type(path) == 'video':
                self.welcome.setText("Load file: {}".format(self.file_path))
            else:
                print("文件选择错误")

    # 清除功能，即对界面进行初始化，使得正在播放、检测或者已经完成播放、检测的参数清空（可以理解为回到界面刚运行状态）
    def clear(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.timer is not None:
            self.timer.stop()
            self.timer = None

        self.camera_flag = False
        self.bar_start.setStyleSheet(
            """
            background-color: transparent;
            background-image: url("qss_imgs/begin.png"); 
            """
        )
        self.plays_left.clear()
        self.plays_right.clear()
        self.welcome.setText("Welcome")
        self.bar_progress.setValue(0)
        self.bar_start_account = 1
        self.camera_path = None
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None


    # def mousePressEvent(self, event):
    #     try:
    #         if self.draggable and event.button() == Qt.MouseButton.LeftButton:
    #             self.dragging_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
    #             event.accept()
    #     except AttributeError:
    #         return
    #
    # def mouseMoveEvent(self, event):
    #     try:
    #         if self.draggable and event.buttons() == Qt.MouseButton.LeftButton:
    #             self.move(event.globalPosition().toPoint() - self.dragging_position)
    #             event.accept()
    #     except AttributeError:
    #         return
    #
    # def mouseReleaseEvent(self, event):
    #     try:
    #         if self.draggable and event.button() == Qt.MouseButton.LeftButton:
    #             self.dragging_position = None
    #             event.accept()
    #     except AttributeError:
    #         return

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
