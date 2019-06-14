from django.shortcuts import render,redirect
from django.views.generic import TemplateView
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.http import HttpResponseRedirect
from django import forms
from django.views.generic import TemplateView
from .models import Voice
from .forms import VoiceForm

import os
import pickle
import numpy as np
import pandas as pd
from sklearn import datasets
from django.conf import settings
from rest_framework import views
from rest_framework import status
from rest_framework.response import Response
from sklearn.ensemble import RandomForestClassifier





# Create your views here.
def home(request):
	voice=Voice.objects.all()
	context_dict={
	'voice':voice,
	}
	return render(request, 'home.html', context_dict)

# to delte file

def delete(request, pk):
	if request.method == 'POST':
		voice=Voice.objects.get(pk=pk)
		voice.delete()
	return redirect(details)

	# take input from client
             
def voice(request):
	if request.method == 'POST':
		form= VoiceForm(request.POST, request.FILES)
		if form.is_valid():
			form.save()
			return redirect('details')			
	else:
		form=VoiceForm()
	return render(request, 'voice.html',{
			'form': form
	})
	 # Take audio as input to predict the audio

def predict1(request):  
    if request.method == 'POST':  
        # f = request.files['file'] 
        # if f.is_valid(): 
           return redirect('giveResult')
    else:
    	voice=Voice.objects.all()
    return render(request,'predict.html',{
    	'voice':voice
    	})

def speechrecognition(request):
	if request.method == 'POST':
		return redirect('details')

	else:
		voice=Voice.objects.all()
	return render(request,'speechrecognition.html',{
    	'voice':voice
    	})   	
   

def giveResult(request): 
	 voice=Voice.objects.all()
	 return render(request, 'giveResult.html',{
	 	'voice':voice
	 	})
def details(request):
	 voice=Voice.objects.all()
	 return render(request, 'details.html', {
		'voice':voice
	  })

def predict(filepath, model):
    # Getting the MFCC
    channel = 1
    feature_dim_2 = 11
    feature_dim_1 = 20
    sample = wav2mfcc(filepath)
    # We need to reshape it remember?
    sample_reshaped = sample.reshape(1, feature_dim_1, feature_dim_2, channel)
    # Perform forward pass
    return get_labels()[0][
            np.argmax(model.predict(sample_reshaped))
    ]

def input(request):
	return render(request,'input.html')

def output(request):
	model = pickle.load(open("D:/myprojectTest/yangchen/speechrecognition/model.sav",'rb'))
	a = predict('./TEST/ka0.wav',model)
	context_dict = {'a':a}
	return render(request,'giveResult.html',context_dict)

# def giveResult(request):
# 	if request.method == 'POST':
# 		voice=Voice.file.get(pk=pk)
# 		voice.predict( models=trainedModel)
# 		return redirect(details)

	

# show details of uploaded file	


	 



  
        
    


# def predict(self, request):
#         predictions = []
#         for entry in request.data:
#             model_name = entry.pop('trainedModel.h5')
#             path = os.path.join(settings.MODEL_ROOT, model_name)
#             with open(path, 'rb') as file:
#                 model = pickle.load(file)
#             try:
#                 result = model.predict(pd.DataFrame([entry]))
#                 predictions.append(result[0])

#             except Exception as err:
#                 return Response(str(err), status=status.HTTP_400_BAD_REQUEST)

#         return Response(predictions, status=status.HTTP_200_OK)
# def edit(request, id):
# 	voice=Voice.objects.get(id=id)

# 	if request.method =="POST":
# 	form=VoiceForm(request.POST, instance=voice)
# 	if form.is_valid(0):
# 		form.save()
# 		return redirect('voice')

#     form = VoiceForm(instance=voice)
#     context_dict= {'form':form}
#     return render(request, 'edit.html', context_dict)

