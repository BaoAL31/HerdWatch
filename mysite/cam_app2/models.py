from django.db import models
from django.shortcuts import render
from django.conf import settings
from django import forms

from modelcluster.fields import ParentalKey

from wagtail.admin.edit_handlers import (
    FieldPanel,
    MultiFieldPanel,
    InlinePanel,
    StreamFieldPanel,
    PageChooserPanel,
)
from wagtail.core.models import Page, Orderable
from wagtail.core.fields import RichTextField, StreamField
from wagtail.images.edit_handlers import ImageChooserPanel
from django.core.files.storage import default_storage
from pathlib import Path
import shutil
import moviepy.editor as moviepy
import numpy as np
from PIL import Image
import cv2
from ultralytics.utils.plotting import Annotator
from shapely.geometry import Polygon
from shapely.geometry.point import Point

from streams import blocks
import os

import sqlite3, datetime, os, uuid, glob
from ultralytics import YOLO
import torch
str_uuid = uuid.uuid4()  # The UUID for image uploading

weights_dir = settings.YOLOV8_WEIGTHS_DIR
yolov8m_model = YOLO(os.path.join(weights_dir, "final_model.pt"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
yolov8m_model.to(device)

def reset():
    files_result = glob.glob(str(Path(f'{settings.MEDIA_ROOT}/Result/*.*')), recursive=True)
    files_upload = glob.glob(str(Path(f'{settings.MEDIA_ROOT}/uploadedPics/*.*')), recursive=True)
    files = []
    if len(files_result) != 0:
        files.extend(files_result)
    if len(files_upload) != 0:
        files.extend(files_upload)
    if len(files) != 0:
        for f in files:
            try:
                if (not (f.endswith(".txt"))):
                    os.remove(f)
            except OSError as e:
                print("Error: %s : %s" % (f, e.strerror))
        file_li = [Path(f'{settings.MEDIA_ROOT}/Result/Result.txt'),
                   Path(f'{settings.MEDIA_ROOT}/uploadedPics/img_list.txt'),
                   Path(f'{settings.MEDIA_ROOT}/Result/stats.txt')]
        for p in file_li:
            file = open(Path(p), "r+")
            file.truncate(0)
            file.close()

    result_dir = str(Path(f'{settings.MEDIA_ROOT}/Result/'))
    for item in os.listdir(result_dir):
        item_path = os.path.join(result_dir, item)
        # Check if the item is a directory
        if os.path.isdir(item_path):
            # Remove the directory and its contents recursively
            shutil.rmtree(item_path)


# Create your models here.
class MediaPage(Page):
    """Media Page."""

    template = "cam_app2/image.html"

    max_count = 2

    name_title = models.CharField(max_length=100, blank=True, null=True)
    name_subtitle = RichTextField(features=["bold", "italic"], blank=True)

    content_panels = Page.content_panels + [
        MultiFieldPanel(
            [
                FieldPanel("name_title"),
                FieldPanel("name_subtitle"),

            ],
            heading="Page Options",
        ),
    ]

    def reset_context(self, request):
        context = super().get_context(request)
        context["my_uploaded_file_names"] = []
        context["my_result_file_names"] = []
        context["my_staticSet_names"] = []
        context["my_lines"]: []
        return context

    def serve(self, request):
        print(request.POST.keys())
        emptyButtonFlag = False
        context = self.reset_context(request)
        try:
            if 'start' in request.POST:
                print("Start selected")
                uploaded_dir = os.path.join(default_storage.location, "uploadedPics")
                uploaded_files_txt = os.path.join(uploaded_dir, "img_list.txt")
                results_dir = os.path.join(default_storage.location, "Result")
                if os.path.getsize(uploaded_files_txt) != 0:
                    with open(uploaded_files_txt, 'r') as files_txt:
                        file_names = files_txt.read().split('\n')[:-1]
                        for file_name in file_names:
                            context["my_uploaded_file_names"].append(str(f'{str(file_name)}'))
                            file_name = file_name.split('\\')[-1]
                            file_path = os.path.join(uploaded_dir, file_name)
                            ext = file_name.split('.')[-1]
                            # print(file_path)
                            if ext not in ['mp4', 'mov']:
                                og_img = cv2.imread(file_path)
                                results = yolov8m_model(file_path, conf=0.15, iou=0.6)
                                annotated_img = annotate_img(og_img, results, scale=0.025, text_weight=2)
                                result_name = file_name.split('.')[-2] + ".jpg"
                                cv2.imwrite(os.path.join(results_dir, file_name.split('.')[-2] + ".jpg"), annotated_img)
                                result_path = Path(f"{settings.MEDIA_URL}Result/{result_name}")
                            else:
                                cap = cv2.VideoCapture(file_path)
                                assert cap.isOpened(), "Error reading video file"
                                w, h, fps = (int(cap.get(x)) for x in
                                             (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

                                video_name = file_name[:-4] + ".avi"
                                video_path = os.path.join(results_dir, video_name)
                                video_writer = cv2.VideoWriter(video_path,
                                                               cv2.VideoWriter_fourcc('M','J','P','G'), fps, (w, h))

                                while cap.isOpened():
                                    success, frame = cap.read()
                                    if not success:
                                        break
                                    results = yolov8m_model(frame, conf=0.15, iou=0.6)
                                    annotated_frame = annotate_img(frame, results, 0.025, 2)
                                    video_writer.write(annotated_frame)

                                video_writer.release()
                                cap.release()
                                cv2.destroyAllWindows()

                                new_video_name = video_name[:-4] + '.mp4'
                                new_video_path = video_path[:-4] + '.mp4'
                                convert_avi_to_mp4(video_path, new_video_path)
                                result_path = Path(f"{settings.MEDIA_URL}Result/{new_video_name}")

                                # print("Video path:", result_path)

                            with open(Path(f'{settings.MEDIA_ROOT}/Result/Result.txt'), 'a') as f:
                                f.write(str(result_path))
                                f.write("\n")
                            context["my_result_file_names"].append(str(result_path))

                return render(request, "cam_app2/image.html", context)

            elif 'restart' in request.POST:
                reset()
                context = self.reset_context(request)
                return render(request, "cam_app2/image.html", context)

            if (request.FILES and emptyButtonFlag == False):
                print("reached here files")
                context["my_uploaded_file_names"] = []
                for file_obj in request.FILES.getlist("file_data"):
                    uuidStr = uuid.uuid4()
                    filename = f"{file_obj.name.split('.')[0]}_{uuidStr}.{file_obj.name.split('.')[-1]}"
                    with default_storage.open(Path(f"uploadedPics/{filename}"), 'wb+') as destination:
                        for chunk in file_obj.chunks():
                            destination.write(chunk)
                    filename = Path(f"{settings.MEDIA_URL}uploadedPics/{file_obj.name.split('.')[0]}_{uuidStr}.{file_obj.name.split('.')[-1]}")
                    with open(Path(f'{settings.MEDIA_ROOT}/uploadedPics/img_list.txt'), 'a') as f:
                        f.write(str(filename))
                        f.write("\n")

                    context["my_uploaded_file_names"].append(str(f'{str(filename)}'))
                return render(request, "cam_app2/image.html", context)

            return render(request, "cam_app2/image.html", {'page': self})
        except Exception as e:
            print(e)
            reset()
            context = self.reset_context(request)
            return render(request, "cam_app2/image.html", context)


    def add_results_to_context(self, results_path, context):
        contents = os.listdir(results_path)
        for item in contents:
            item_path = os.path.join(results_path, item)
            if os.path.isdir(item_path):
                results = os.listdir(item_path)
                for result in results:
                    result_path = os.path.join(item_path, result)
                    print(result)
                    if result.split('.')[-1] == 'avi':
                        result = result[:-4] + '.mp4'
                        new_result_path = os.path.join(item_path, result)
                        convert_avi_to_mp4(result_path, new_result_path)
                        result_path = new_result_path
                    shutil.move(result_path, os.path.join(results_path, result))
                    filename = Path(f"{settings.MEDIA_URL}Result/{result}")
                    with open(Path(f'{settings.MEDIA_ROOT}/Result/Result.txt'), 'a') as f:
                        f.write(str(filename))
                        f.write("\n")
                    context["my_result_file_names"].append(str(f'{str(filename)}'))
                shutil.rmtree(item_path)
        return context

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
        total_text_x = w // 2 - total_text_size[0] // 2
        total_text_y = total_text_size[1] + 10
        padding = 10
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


def convert_avi_to_mp4(avi_file_path, output_name):
    clip = moviepy.VideoFileClip(avi_file_path)
    clip.write_videofile(output_name)
    return True