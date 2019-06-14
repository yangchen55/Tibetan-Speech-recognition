from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static



urlpatterns = [
    # path('home/', views.Home.as_view(),name='home'),
    # path('upload/', views.upload, name='upload'),
    path('', views.voice, name='voice'),
    path('details/<int:pk>', views.delete, name='delete'),
    path('details', views.details, name='details'),
    path('speechrecognition', views.speechrecognition, name='speechrecognition'),
    path('predict1',views.predict1, name="predict1"),
    path('giveResult',views.giveResult, name="giveResult"),
    path('output',views.output, name="output"),
    path('input',views.input, name="input"),


     # path('edit', views.edit, name='edit'),
    
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)



