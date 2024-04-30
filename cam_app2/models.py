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

from streams import blocks
import os

import sqlite3, datetime, os, uuid, glob
from ultralytics import YOLO

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

# Create your models here.
class ImagePage(Page):
    """Image Page."""

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
        emptyButtonFlag = False
        context = self.reset_context(request)
        if request.POST.get('start') == "":
            print("Start selected")
            weights_dir = settings.YOLOV8_WEIGTHS_DIR
            yolov8m_model = YOLO(os.path.join(weights_dir, "best.pt"))
            uploaded_dir = os.path.join(default_storage.location, "uploadedPics")
            uploaded_files_txt = os.path.join(uploaded_dir, "img_list.txt")
            results_dir = os.path.join(default_storage.location, "Result")

            if os.path.getsize(uploaded_files_txt) != 0:
                results = yolov8m_model.predict(source=uploaded_dir, save=True, project=results_dir)
                # print(f'Results len: {len(results)}')
                with open(uploaded_files_txt, 'r') as files_txt:
                    file_names = files_txt.read().split('\n')[:-1]
                    for file_name in file_names:
                        # print(file_name)
                        context["my_uploaded_file_names"].append(str(f'{str(file_name)}'))
                        context = self.add_results_to_context(results_dir, context)



            # with default_storage.open(Path("uploadedPics")) as uploaded_files:
            # uploaded_files = default_storage.listdir('')[1]
            # for file in uploaded_files:
            #     print(file)

            return render(request, "cam_app2/image.html", context)

        if (request.FILES and emptyButtonFlag == False):
            print("reached here files")
            reset()
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
                    filename = Path(f"{settings.MEDIA_URL}Result/{result }")
                    with open(Path(f'{settings.MEDIA_ROOT}/Result/Result.txt'), 'a') as f:
                        f.write(str(filename))
                        f.write("\n")
                    context["my_result_file_names"].append(str(f'{str(filename)}'))
                shutil.rmtree(item_path)
        return context

    def convert_avi_to_mp4(self, avi_file_path, output_name):
        print("HEY", avi_file_path)
        clip = moviepy.VideoFileClip(avi_file_path)
        clip.write_videofile(output_name)
        return True