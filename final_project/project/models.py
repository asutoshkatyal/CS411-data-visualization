from __future__ import unicode_literals
from datetime import datetime

from django.db import models, connection
from django.utils import timezone

class Data(models.Model):
	category = models.CharField(max_length = 200)
	value = models.IntegerField(default = 0)
	source = models.CharField(max_length = 200)
	time = models.DateTimeField(default=timezone.now) 
	Sensor = models.ForeignKey(
        'Sensors',
        on_delete=models.CASCADE, 
<<<<<<< HEAD
=======

    )
>>>>>>> e51a9617d6cf11d985daf7e8c0a84ba9b15e9bb1

    )

	#def addData(self, data):
		#with connection.cursor() as cur:
			#cur.execute('INSERT INTO project_data (category, value, source, time) VALUES (%s, %s, %s, %s);', [data.category, data.value, data.source, datetime.now()])

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