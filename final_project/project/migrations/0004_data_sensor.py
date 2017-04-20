# -*- coding: utf-8 -*-
# Generated by Django 1.10.6 on 2017-03-19 19:57
from __future__ import unicode_literals

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('project', '0003_sensors'),
    ]

    operations = [
        migrations.AddField(
            model_name='data',
            name='Sensor',
            field=models.ForeignKey(default=1, on_delete=django.db.models.deletion.CASCADE, to='project.Sensors'),
            preserve_default=False,
        ),
    ]