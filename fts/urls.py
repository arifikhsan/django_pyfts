from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('dynamic', views.dynamic, name='dynamic'),
    path('predict', views.predict, name='predict'),
]
