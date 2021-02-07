from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('dynamic', views.dynamic, name='dynamic'),
    path('model_cheng', views.model_cheng, name='model_cheng'),
]
