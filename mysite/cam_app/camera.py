import pickle
from django.conf import settings
from cam_app import views
from django.http import StreamingHttpResponse
import sqlite3
import datetime

# import some common libraries
import numpy as np
import os, json, cv2, random, glob, uuid
import matplotlib.pyplot as plt
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import torch

from pathlib import Path
import time

weights_dir = settings.YOLOV8_WEIGTHS_DIR
yolov8m_model = YOLO(os.path.join(weights_dir, "final_model.pt"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
yolov8m_model.to(device)

class VideoCamera(object):
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame_with_detection(self):
        success, img = self.video.read()
        results = yolov8m_model(img, save=False, conf=0.2)
        annotated_img = annotate_img(img, results, scale=0.025, text_weight=1)
        annotated_img = np.asarray(annotated_img)
        ret, annotated_img = cv2.imencode('.jpg', annotated_img) # check if it work
        return annotated_img.tobytes(), annotated_img

def annotate_img(img, results, scale=0.03, text_weight=2):
    color_palette = [
        (255, 165, 0),
        (255, 255, 0),
        (255, 0, 0),
        (255, 0, 255),
        (0, 0, 255),
        (0, 255, 255),
        (0, 255, 0),
    ]

    h, w, channels = img.shape
    font_scale = w / (25 / scale)
    annotator = Annotator(img)
    for r in results:
        boxes = r.boxes
        total = len(boxes.cls.cpu().tolist())
        names = yolov8m_model.names
        class_list = boxes.cls.cpu().tolist()
        class_counts = [class_list.count(i) for i in range(len(names))]
        for box in boxes:
            b = box.xyxy[0]
            c = box.cls
            annotator.box_label(b, yolov8m_model.names[int(c)],
                                color=color_palette[int(c)],
                                txt_color=(0, 0, 0))
        total_text = f'Total: {total}'
        total_text_size, _ = cv2.getTextSize(
            total_text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=text_weight
        )
        padding = 5
        total_text_x = w // 2 - total_text_size[0] // 2
        total_text_y = total_text_size[1] + padding
        cv2.rectangle(
            img,
            (total_text_x - padding, total_text_y - total_text_size[1] - padding),
            (total_text_x + total_text_size[0] + padding, total_text_y + padding * 2),
            (37, 255, 225),
            -1,
        )
        cv2.putText(
            img, total_text, (total_text_x, total_text_y),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale,
            (0, 0, 0), text_weight * 2
        )

        class_counts_text = ", ".join([names[i] + ": " + str(class_counts[i]) for i in range(len(class_counts))])
        class_counts_text_size, _ = cv2.getTextSize(
            class_counts_text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=text_weight
        )

        class_counts_text_x = w // 2 - class_counts_text_size[0] // 2
        class_counts_text_y = class_counts_text_size[1] + total_text_y + padding * 2
        cv2.rectangle(
            img,
            (class_counts_text_x - padding, class_counts_text_y - class_counts_text_size[1] - padding),
            (class_counts_text_x + class_counts_text_size[0] + padding, class_counts_text_y + padding * 2),
            (37, 255, 225),
            -1,
        )
        cv2.putText(
            img, class_counts_text, (class_counts_text_x, class_counts_text_y),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale,
            (0, 0, 0), text_weight
        )
        return img

def generate_frames(camera):
    try:
        start_time = time.time()
        frame_count = 0
        while True:
            frame, img = camera.get_frame_with_detection()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
            frame_count += 1
            delta_time = time.time() - start_time
            # print(f'fps: {(frame_count / delta_time):.2f}')
    except Exception as e:
        print(e)
    finally:
        print("Reached finally, detection stopped")
        cv2.destroyAllWindows()
