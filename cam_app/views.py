# Create your views here.
from django.shortcuts import render
from django.views import View

import uuid
from django.views.decorators.clickjacking import xframe_options_sameorigin


str_uuid = uuid.uuid4()  # The UUID for image uploading

class ScannerVideoView(View):
    @xframe_options_sameorigin
    def get(self, request):
        return render(request, 'cam_app/video2.html')

class NoVideoView(View):
    def get(self, request):
        # print(request.POST)
        return render(request, 'cam_app/no_video.html')
