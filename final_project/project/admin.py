from django.contrib import admin

from .models import Data,Sensors


class DataAdmin(admin.ModelAdmin):
	list_display = ['category', 'value', 'source', 'time']
	class Meta:
		model = Data

admin.site.register(Data, DataAdmin)


class SensorAdmin(admin.ModelAdmin):
	list_display = ['sensor_id', 'Sensor_name', 'user']
	class Meta:
		model = Data 

admin.site.register(Sensors, SensorAdmin)