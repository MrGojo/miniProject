from django.db import models
from django.utils import timezone

# Create your models here.

class AnalysisResult(models.Model):
    media_file = models.FileField(upload_to='uploads/', null=True, blank=True)
    confidence_score = models.FloatField(default=0.0)
    summary = models.TextField(default='')
    methods = models.JSONField(default=list)
    recommendations = models.TextField(default='')
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Analysis Result {self.id} - {self.created_at}"
