from django.urls import path
from . import views

urlpatterns = [
    # Rutas principales
    path('', views.inicio, name='inicio'),
    path('pacientes/', views.lista_pacientes, name='lista_pacientes'),
    path('pacientes/crear/', views.crear_paciente, name='crear_paciente'),
    path('pacientes/editar/<int:paciente_id>/', views.editar_paciente, name='editar_paciente'),
    path('pacientes/eliminar/<int:paciente_id>/', views.eliminar_paciente, name='eliminar_paciente'),
    
    # API para predicciones
    path('api/predecir/', views.api_predecir, name='api_predecir'),
    
    # Estadísticas
    path('estadisticas/', views.ver_estadisticas, name='estadisticas'),
]