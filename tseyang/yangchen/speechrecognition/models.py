
from django.db import models
from django.conf import settings
from django.utils import timezone
from PIL import Image


import datetime

# Create your models here.
class Voice(models.Model):
	title=models.CharField(max_length=100)
	created_date=models.DateTimeField(auto_now_add=True)
	file=models.FileField(upload_to='audio/file/', max_length=200)
	author =models.CharField(max_length=100)
	cover=models.ImageField(upload_to='audio/cover/', null=True, blank=True)

	def __str__(self):
		return self.title






