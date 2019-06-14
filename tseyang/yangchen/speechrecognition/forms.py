from django import forms
from .models import Voice 

class VoiceForm(forms.ModelForm):
	class Meta:
		model= Voice
		fields='__all__'
		file = forms.FileField( label = (u"file" ))
   