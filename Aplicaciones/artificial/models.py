# models.py

from django.db import models

class Paciente(models.Model):
    """
    Modelo para almacenar la información de los pacientes.
    Incluye variables utilizadas para predecir riesgo de diabetes.
    """
    OPCIONES_ANTECEDENTES = [
        (True, 'Sí'),
        (False, 'No'),
    ]
    
    nombre = models.CharField(max_length=100, verbose_name="Nombre completo")
    edad = models.PositiveIntegerField(verbose_name="Edad")
    imc = models.FloatField(verbose_name="Índice de Masa Corporal")
    actividad_fisica = models.FloatField(verbose_name="Nivel de actividad física semanal (horas)")
    presion_arterial = models.IntegerField(verbose_name="Presión arterial sistólica")
    antecedentes_familiares = models.BooleanField(choices=OPCIONES_ANTECEDENTES, verbose_name="Antecedentes familiares de diabetes")
    glucosa_ayunas = models.FloatField(verbose_name="Nivel de glucosa en ayunas (mg/dL)")
    riesgo_diabetes = models.BooleanField(null=True, blank=True, verbose_name="Riesgo de diabetes")
    probabilidad_riesgo = models.FloatField(null=True, blank=True, verbose_name="Probabilidad de riesgo (%)")
    fecha_creacion = models.DateTimeField(auto_now_add=True, verbose_name="Fecha de creación")
    fecha_actualizacion = models.DateTimeField(auto_now=True, verbose_name="Fecha de actualización")

    def __str__(self):
        return f"{self.nombre} - {'Con riesgo' if self.riesgo_diabetes else 'Sin riesgo'}"
    
    class Meta:
        verbose_name = "Paciente"
        verbose_name_plural = "Pacientes"
        ordering = ['-fecha_creacion']

