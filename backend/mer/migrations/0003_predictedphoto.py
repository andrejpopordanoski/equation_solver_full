# Generated by Django 2.2.5 on 2020-07-14 22:26

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('mer', '0002_auto_20200714_2159'),
    ]

    operations = [
        migrations.CreateModel(
            name='PredictedPhoto',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('photo', models.ImageField(upload_to='images/')),
            ],
        ),
    ]
