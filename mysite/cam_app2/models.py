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
from ultralytics.solutions import object_counter

str_uuid = uuid.uuid4()  # The UUID for image uploading

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

    color_palete = [
        (255, 165, 0),
        (255, 255, 0),
        (255, 0, 0),
        (255, 0, 255),
        (0, 0, 255),
        (0, 255, 255),
        (0, 255, 0),
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
                weights_dir = settings.YOLOV8_WEIGTHS_DIR
                uploaded_dir = os.path.join(default_storage.location, "uploadedPics")
                uploaded_files_txt = os.path.join(uploaded_dir, "img_list.txt")
                results_dir = os.path.join(default_storage.location, "Result")
                if os.path.getsize(uploaded_files_txt) != 0:
                    with open(uploaded_files_txt, 'r') as files_txt:
                        file_names = files_txt.read().split('\n')[:-1]
                        for file_name in file_names:
                            line_thickness = 2

                            context["my_uploaded_file_names"].append(str(f'{str(file_name)}'))
                            file_name = file_name.split('\\')[-1]
                            file_path = os.path.join(uploaded_dir, file_name)
                            ext = file_name.split('.')[-1]
                            print(file_path)
                            if ext not in ['mp4', 'mov']:
                                og_img = cv2.imread(file_path)
                                h, w, channels = og_img.shape
                                scale = 0.03
                                font_scale = w / (25 / scale)
                                print(f'Height: {h}, Width: {w}')
                                yolov8m_model = YOLO(os.path.join(weights_dir, "final_model.pt"))
                                yolov8m_model.to('cuda')
                                results = yolov8m_model(file_path, save=False, conf=0.1)
                                annotator = Annotator(og_img)
                                for r in results:
                                    boxes = r.boxes
                                    count = len(boxes.cls.cpu().tolist())
                                    print("Counter:", count)
                                    for box in boxes:
                                        b = box.xyxy[0]
                                        c = box.cls
                                        annotator.box_label(b, yolov8m_model.names[int(c)],
                                                            color=self.color_palete[int(c)],
                                                            txt_color=(0, 0, 0))

                                    text_size, _ = cv2.getTextSize(
                                        "Counter: " + str(count), cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=line_thickness
                                    )
                                    text_x = w // 2 - text_size[0] // 2
                                    text_y = text_size[1]
                                    cv2.rectangle(
                                        og_img,
                                        (text_x, text_y - text_size[1]),
                                        (text_x + text_size[0], text_y),
                                        (37, 255, 225),
                                        -1,
                                    )
                                    cv2.putText(
                                        og_img, "Counter: " + str(count), (text_x, text_y),
                                        cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                                        (0, 0, 0), line_thickness
                                    )

                                result_name = file_name.split('.')[-2] + ".jpg"
                                annotator.save(filename=os.path.join(results_dir, file_name.split('.')[-2] + ".jpg"))
                                result_path = Path(f"{settings.MEDIA_URL}Result/{result_name}")
                            else:
                                vid_frame_count = 0
                                yolov8m_model = YOLO(os.path.join(weights_dir, "final_model.pt"))
                                yolov8m_model.to('cuda')
                                names = yolov8m_model.names
                                cap = cv2.VideoCapture(file_path)
                                assert cap.isOpened(), "Error reading video file"
                                w, h, fps = (int(cap.get(x)) for x in
                                             (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
                                scale = 0.03
                                font_scale = w / (25 / scale)
                                video_name = file_name[:-4] + ".avi"
                                video_path = os.path.join(results_dir, video_name)
                                video_writer = cv2.VideoWriter(video_path,
                                                               cv2.VideoWriter_fourcc('F', 'M', 'P', '4'), fps, (w, h))
                                counting_regions = [
                                    {
                                        "name": "YOLOv8 Rectangle Region",
                                        "polygon": Polygon([(0, 0), (0, h), (w, h), (w, 0)]),
                                        # Polygon points
                                        "counts": 0,
                                        "dragging": False,
                                        "region_color": (37, 255, 225),  # BGR Value
                                        "text_color": (0, 0, 0),  # Region Text Color
                                    },
                                ]

                                while cap.isOpened():
                                    success, frame = cap.read()
                                    if not success:
                                        break
                                    vid_frame_count += 1

                                    # Extract the results
                                    results = yolov8m_model.track(frame, persist=True, conf=0.1)

                                    if results[0].boxes.id is not None:
                                        boxes = results[0].boxes.xyxy.cpu()
                                        track_ids = results[0].boxes.id.int().cpu().tolist()
                                        clss = results[0].boxes.cls.cpu().tolist()

                                        annotator = Annotator(frame)

                                        for box, track_id, cls in zip(boxes, track_ids, clss):
                                            annotator.box_label(box, str(names[cls]),
                                                                color=self.color_palete[int(cls)],
                                                                txt_color=(0, 0, 0))

                                            bbox_center = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2  # Bbox center

                                            # Check if detection inside region
                                            for region in counting_regions:
                                                if region["polygon"].contains(Point((bbox_center[0], bbox_center[1]))):
                                                    region["counts"] += 1

                                    # Draw regions (Polygons/Rectangles)
                                    for region in counting_regions:
                                        region_label = str(region["counts"])
                                        region_color = region["region_color"]
                                        region_text_color = region["text_color"]

                                        polygon_coords = np.array(region["polygon"].exterior.coords, dtype=np.int32)
                                        centroid_x, centroid_y = int(region["polygon"].centroid.x), int(
                                            region["polygon"].centroid.y)


                                        text_size, _ = cv2.getTextSize(
                                            "Counter: " + region_label, cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=line_thickness
                                        )
                                        text_x = centroid_x - text_size[0] // 2
                                        text_y = text_size[1]
                                        cv2.rectangle(
                                            frame,
                                            (text_x, text_y - text_size[1]),
                                            (text_x + text_size[0], text_y),
                                            region_color,
                                            -1,
                                        )
                                        cv2.putText(
                                            frame, "Counter: " + region_label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                                            region_text_color, line_thickness
                                        )

                                    video_writer.write(frame)

                                    for region in counting_regions:  # Reinitialize count for each region
                                        region["counts"] = 0

                                    if cv2.waitKey(1) & 0xFF == ord("q"):
                                        break
                                del vid_frame_count
                                video_writer.release()
                                cap.release()
                                cv2.destroyAllWindows()

                                new_video_path = video_path[:-4] + '.mp4'
                                self.convert_avi_to_mp4(video_path, new_video_path)
                                result_path = Path(f"{settings.MEDIA_URL}Result/{video_name[:-4] + '.mp4'}")

                                print("Video path:", result_path)

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

    def annotate_img(self, img, result):
        pass

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
                        self.convert_avi_to_mp4(result_path, new_result_path)
                        result_path = new_result_path
                    shutil.move(result_path, os.path.join(results_path, result))
                    filename = Path(f"{settings.MEDIA_URL}Result/{result}")
                    with open(Path(f'{settings.MEDIA_ROOT}/Result/Result.txt'), 'a') as f:
                        f.write(str(filename))
                        f.write("\n")
                    context["my_result_file_names"].append(str(f'{str(filename)}'))
                shutil.rmtree(item_path)
        return context

    def convert_avi_to_mp4(self, avi_file_path, output_name):
        clip = moviepy.VideoFileClip(avi_file_path)
        clip.write_videofile(output_name)
        return True