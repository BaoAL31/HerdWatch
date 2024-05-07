# Generated by Django 3.2.12 on 2024-05-02 05:53

from django.conf import settings
from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('wagtailforms', '0004_add_verbose_name_plural'),
        ('contenttypes', '0002_remove_content_type_name'),
        ('wagtailredirects', '0006_redirect_increase_max_length'),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('wagtailcore', '0066_collection_management_permissions'),
        ('cam_app', '0001_initial'),
    ]

    operations = [
        migrations.RenameModel(
            old_name='VideoPage',
            new_name='LivePage',
        ),
        migrations.AlterModelOptions(
            name='livepage',
            options={'verbose_name': 'Live Page', 'verbose_name_plural': 'Live Pages'},
        ),
    ]
