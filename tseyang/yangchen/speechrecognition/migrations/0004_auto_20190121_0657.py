# Generated by Django 2.1.5 on 2019-01-21 06:57

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('speechrecognition', '0003_auto_20190121_0630'),
    ]

    operations = [
        migrations.AlterField(
            model_name='voice',
            name='cover',
            field=models.ImageField(blank=True, null=True, upload_to='audio/cover/'),
        ),
    ]
