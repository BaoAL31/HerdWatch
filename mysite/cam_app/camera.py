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

from pathlib import Path
import time

class VideoCamera(object):
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame_with_detection(self):
        color_palete = [
            (255, 165, 0),
            (255, 255, 0),
            (255, 0, 0),
            (255, 0, 255),
            (0, 0, 255),
            (0, 255, 255),
            (0, 255, 0),
        ]
        success, image = self.video.read()
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        outputs = image
        h, w, channels = outputs.shape

        weights_dir = settings.YOLOV8_WEIGTHS_DIR
        yolov8m_model = YOLO(os.path.join(weights_dir, "final_model.pt"))
        outputImage = yolov8m_model(image, save=False)
        annotator = Annotator(outputs)
        for r in outputImage:
            boxes = r.boxes
            count = len(boxes.cls.cpu().tolist())
            for box in boxes:
                b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
                c = box.cls
                annotator.box_label(b, yolov8m_model.names[int(c)], color=color_palete[int(c)])

            text_size, _ = cv2.getTextSize(
                str(count), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=2
            )
            text_x = w // 2 - text_size[0] * 2 - 15
            text_y = text_size[1]
            cv2.rectangle(
                outputs,
                (text_x - 5, text_y - text_size[1] - 5),
                (text_x + text_size[0] + 100, text_y + 5),
                (37, 255, 225),
                -1,
            )
            cv2.putText(
                outputs, "Counter: " + str(count), (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 0), 2
            )

        outputImagetoReturn = np.asarray(annotator.result())
        ret, outputImagetoReturn = cv2.imencode('.jpg', outputImagetoReturn) # check if it work
        return outputImagetoReturn.tobytes(), outputImage

def generate_frames(camera):
    try:
        while True:
            frame, img = camera.get_frame_with_detection()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    except Exception as e:
        print(e)

    finally:
        print("Reached finally, detection stopped")
        cv2.destroyAllWindows()
