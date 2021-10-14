from django.db import models
import keras.models as km


class MerNN(models.Model):
    id = models.UUIDField(primary_key=True)
    title = models.CharField(max_length=120)

    def _str_(self):
        return self.title


class Latex(models.Model):
    created = models.DateTimeField(auto_now_add=True)
    latex_string = models.CharField(max_length=100, blank=True, default='')

    class Meta:
        ordering = ['created']


class PredictedPhoto(models.Model):
    photo = models.ImageField(upload_to='images/')


class CNNModel:
    def __init__(self, model=None):
        self.model = model

    def get_model(self):
        return self.model

    def set_model(self, model):
        self.model = model


class SingletonModel(models.Model):
    class Meta:
        abstract = True

    def save(self, *args, **kwargs):
        self.pk = 1
        super(SingletonModel, self).save(*args, **kwargs)

    def delete(self, *args, **kwargs):
        pass

    @classmethod
    def load(cls):
        obj, created = cls.objects.get_or_create(pk=1)
        return obj


# class SiteSettings(SingletonModel):
#     default_model = km.Sequential()
#     keras_model = models.Field(default=default_model)
