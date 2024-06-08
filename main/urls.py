from django.contrib import admin
from django.urls import path 
from . import views

urlpatterns = [
    path('', views.index),
    path('train/', views.train,name='train_view'),
    path('predict/', views.predict,name='predict_view'),
]
