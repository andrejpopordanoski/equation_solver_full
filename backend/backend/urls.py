"""backend URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include                 # add this
from rest_framework import routers
from mer import views as mer_views
from mer import models as mer_models
from keras.models import model_from_json
import keras
import tensorflow as tf

# tf.keras.backend.clear_session()

# cnn = models.CNNModel()

mer_cnn = mer_models.CNNModel()

json_file = open('mer/assets/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("mer/assets/model_weights.h5")
mer_cnn.set_model(loaded_model)

# with keras.backend.get_session().graph.as_default():
# json_file = open('tomato/assets/model_sgd_lr_001_99_35e.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# # load weights into new model
# loaded_model.load_weights("tomato/assets/model_sgd_lr_001_99_35e.h5")
# cnn.set_model(loaded_model)
#
#
# json_file = open('tomato/assets/model_128_93.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# # load weights into new model
# loaded_model.load_weights("tomato/assets/model-128-93.h5")
#
# cnn.set_mobile_model(loaded_model)

router = routers.DefaultRouter()
# router.register(r'tomato_predict', tomato_views.TomatoView, 'tomato_predict')
# router.register(r'tomato_predict_all', tomato_views.TomatoViewAll, 'tomato_predict_all')
# router.register(r'tomato_filters', tomato_views.TomatoFilters, 'tomato_filters')
# router.register(r'layers_count', tomato_views.LayerCount, 'layers_count')
# router.register(r'merge_images', steganography_views.MergeImages, 'merge_images')
# router.register(r'unmerge_image', steganography_views.UnmergeImage, 'unmerge_image')

router.register(r'mer_predict', mer_views.MERView, 'mer_predict')

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include(router.urls))
]

def get_model():
    return mer_cnn.get_model()
#
# def get_mobile_model():
#     return mer_cnn.get_mobile_model()