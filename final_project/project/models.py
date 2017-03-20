from __future__ import unicode_literals
from datetime import datetime

from django.db import models, connection
from django.utils import timezone
from django.contrib.auth.models import User 
from django.contrib.auth.models import (
    BaseUserManager, AbstractBaseUser
)
class Data(models.Model):
	category = models.CharField(max_length = 200)
	value = models.IntegerField(default = 0)
	source = models.CharField(max_length = 200)
	time = models.DateTimeField(default=timezone.now) 
	Sensor = models.ForeignKey(
        'Sensors',
        on_delete=models.CASCADE, 

    )

	#def addData(self, data):
		#with connection.cursor() as cur:
			#cur.execute('INSERT INTO project_data (category, value, source, time) VALUES (%s, %s, %s, %s);', [data.category, data.value, data.source, datetime.now()])

class MyUserManager(BaseUserManager):
    def create_user(self, email, date_of_birth, password=None):
        """
        Creates and saves a User with the given email, date of
        birth and password.
        """
        if not email:
            raise ValueError('Users must have an email address')

        user = self.model(
            email=self.normalize_email(email),
            date_of_birth=date_of_birth,
        )

        user.set_password(password)
        user.save(using=self._db)
        return user
"""class MyUser(AbstractBaseUser):
       verbose_name='email address',
        max_length=255,
        unique=True,
    )
    date_of_birth = models.DateField()
    is_active = models.BooleanField(default=True)
    is_admin = models.BooleanField(default=False)

    objects = MyUserManager()

    USERNAME_FIELD = 'email'""" 

def deleteData(self):
		with connection.cursor() as cur:
			cur.execute('DELETE FROM data_data WHERE ID = %s;', [self.ID])

def searchData(self, data = None):
		if data == None:
			datapoint = self.objects.raw('SELECT * FROM project_data')
		else:
			datapoint = self.objects.raw('SELECT * FROM project_data WHERE ID = %s', [data.ID])
		return datapoint

def updateData(self, data):
		with connection.cursor() as cur:
			cur.execute('UPDATE project_data SET category = %s, value = %s, source = %s WHERE time = %s', [data.category, data.value, data.source, data.time])

class Sensors(models.Model):
    
    sensor_id = models.AutoField(primary_key=True) 
    Sensor_name = models.CharField(max_length = 200) 
    user = models.ForeignKey(User, unique=True)

