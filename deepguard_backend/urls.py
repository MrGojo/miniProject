"""
URL configuration for deepguard_backend project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
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
from django.urls import path, include, re_path
from django.conf import settings
from django.conf.urls.static import static
from django.views.generic import TemplateView
from django.views.defaults import page_not_found
from django.views.decorators.csrf import csrf_exempt
from detector import views  # Updated import to use views from detector app

# Helper function to create URL patterns with optional .html extension
def template_view(template_name):
    return TemplateView.as_view(template_name=template_name)

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('detector.urls')),
    
    # Frontend routes - handle both with and without .html
    path('', template_view('index.html'), name='home'),
    re_path(r'^index(?:\.html)?$', template_view('index.html'), name='home-html'),
    
    path('demo/', views.demo_view, name='demo'),
    re_path(r'^demo(?:\.html)?$', views.demo_view, name='demo-html'),  # Updated to use view function
    
    path('features/', template_view('features.html'), name='features'),
    re_path(r'^features(?:\.html)?$', template_view('features.html'), name='features-html'),
    
    path('solution/', template_view('solution.html'), name='solution'),
    re_path(r'^solution(?:\.html)?$', template_view('solution.html'), name='solution-html'),
    
    path('use-cases/', template_view('use-cases.html'), name='use-cases'),
    re_path(r'^use-cases(?:\.html)?$', template_view('use-cases.html'), name='use-cases-html'),
    
    path('contact/', template_view('contact.html'), name='contact'),
    re_path(r'^contact(?:\.html)?$', template_view('contact.html'), name='contact-html'),
    
    path('learn-more/', template_view('learn-more.html'), name='learn-more'),
    re_path(r'^learn-more(?:\.html)?$', template_view('learn-more.html'), name='learn-more-html'),
    
    path('result/', views.result_view, name='result'),
    path('analyze/', views.analyze_media, name='analyze_media'),  # Moved out of api/ prefix
]

# Serve static and media files in development
if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

# Handle 404 errors in development
handler404 = 'django.views.defaults.page_not_found'
