from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.conf import settings
from .models import AnalysisResult
import json
import os
from datetime import datetime
import time

def demo_view(request):
    """Render the demo upload page"""
    return render(request, 'demo.html')

def result_view(request):
    """Render the results page"""
    return render(request, 'result.html')

# This is a placeholder for your actual model prediction function
def predict_deepfake(media_path):
    """
    Placeholder for the actual deepfake detection model.
    Replace this with your actual model implementation.
    """
    # Simulate processing time
    time.sleep(1.5)
    
    # Mock analysis results
    return {
        'confidence_score': 0.85,
        'facial_score': 0.92,
        'artifact_score': 0.88,
        'texture_score': 0.82,
        'summary': 'The analysis indicates a high likelihood of authenticity with minor inconsistencies in facial features.',
        'methods': ['Facial Landmark Analysis', 'Texture Analysis', 'Artifact Detection'],
        'recommendations': 'The media appears to be authentic, but we recommend additional verification for critical applications.'
    }

@csrf_exempt
@require_http_methods(["POST"])
def analyze_media(request):
    """Handle media file upload and analysis"""
    try:
        # Get the uploaded media file
        media_file = request.FILES.get('media_file')
        if not media_file:
            return JsonResponse({'error': 'No media file provided'}, status=400)

        # Validate file type
        allowed_types = ['image/jpeg', 'image/png']
        if media_file.content_type not in allowed_types:
            return JsonResponse({'error': 'Invalid file type. Please upload JPG or PNG files only.'}, status=400)

        # Validate file size (10MB max)
        if media_file.size > 10 * 1024 * 1024:
            return JsonResponse({'error': 'File size must be less than 10MB'}, status=400)

        # Create a timestamp for unique filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'upload_{timestamp}_{media_file.name}'
        
        # Create the full file path
        upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
        os.makedirs(upload_dir, exist_ok=True)
        filepath = os.path.join(upload_dir, filename)

        # Save the file
        start_time = time.time()
        with open(filepath, 'wb+') as destination:
            for chunk in media_file.chunks():
                destination.write(chunk)

        # Get prediction from model
        prediction = predict_deepfake(filepath)
        processing_time = time.time() - start_time

        # Save the analysis result
        result = AnalysisResult.objects.create(
            media_file=f'uploads/{filename}',
            confidence_score=prediction['confidence_score'],
            summary=prediction['summary'],
            methods=json.dumps(prediction['methods']),
            recommendations=prediction['recommendations']
        )

        # Prepare response
        response_data = {
            'confidence_score': prediction['confidence_score'],
            'facial_score': prediction['facial_score'],
            'artifact_score': prediction['artifact_score'],
            'texture_score': prediction['texture_score'],
            'summary': prediction['summary'],
            'methods': prediction['methods'],
            'recommendations': prediction['recommendations'],
            'file_name': media_file.name,
            'file_size': media_file.size,
            'file_type': media_file.content_type,
            'processing_time': round(processing_time, 2),
            'total_time': round(processing_time, 2),
            'model_version': 'DeepGuard AI v1.0',
            'image_url': f'{settings.MEDIA_URL}uploads/{filename}'
        }

        return JsonResponse(response_data)

    except Exception as e:
        # Log the error in production
        print(f"Error processing upload: {str(e)}")
        return JsonResponse({
            'error': 'An error occurred while processing your file. Please try again.'
        }, status=500)

def get_result(request, result_id):
    """Retrieve a specific analysis result"""
    try:
        result = AnalysisResult.objects.get(id=result_id)
        response_data = {
            'confidence_score': result.confidence_score,
            'summary': result.summary,
            'methods': json.loads(result.methods),
            'recommendations': result.recommendations,
            'media_url': result.media_file.url if result.media_file else None
        }
        return JsonResponse(response_data)
    except AnalysisResult.DoesNotExist:
        return JsonResponse({'error': 'Result not found'}, status=404)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
