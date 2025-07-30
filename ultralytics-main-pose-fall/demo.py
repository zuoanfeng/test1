import numpy as np
import cv2
from ultralytics import YOLO
import torch

import math


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

def connect_keypoints(img, keypoints):
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

img = cv2.imread('demo/demo.jpg')
img_ = np.copy(img)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

det_model = YOLO('./weights/yolo11n-pose.pt')
results = det_model.predict(
    source=img_,
    conf=0.25,
    iou=0.45,
    verbose=False,
    device=device,
    save=False
)[0]
boxes = results.boxes.xyxy.detach().cpu().numpy().tolist()
conf = results.boxes.conf.detach().cpu().numpy().tolist()
keypoints = results.keypoints.data.detach().cpu().numpy().tolist()

fall_det = FallDetector()

for id, box in enumerate(boxes):
    box = [int(x) for x in box]
    x1, y1, x2, y2 = box
    keypoint = keypoints[id]
    keypoint = [(int(x), int(y)) for x, y, _ in keypoint]
    score = conf[id]

    img_ = connect_keypoints(img_, keypoint)
    fallen = fall_det.is_person_fallen(keypoint, img_.shape[0], img_.shape[1])

    cv2.rectangle(img_, (x1, y1), (x2, y2), (255, 0, 0), 2)

    cv2.imshow('image', img_)
    cv2.waitKey(0)
