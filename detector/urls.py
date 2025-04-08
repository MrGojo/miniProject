from django.urls import path
from . import views

urlpatterns = [
    path('analyze/', views.analyze_media, name='analyze_media'),
    path('result/<int:result_id>/', views.get_result, name='get_result'),
] 