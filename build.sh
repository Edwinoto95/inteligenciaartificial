#!/usr/bin/env bash
# exit on error
set -o errexit

# Mostrar información de diagnóstico
echo "Python version: $(python --version)"
echo "Current directory: $(pwd)"
echo "Listing files: $(ls -la)"

# Instalar pip y setuptools primero
python -m pip install --upgrade pip setuptools wheel

# Instalar gunicorn explícitamente primero
python -m pip install gunicorn==21.2.0

# Instalar todas las dependencias
python -m pip install -r requirements.txt

# Mostrar paquetes instalados para verificar
echo "Installed packages: $(python -m pip list)"

# Configurar archivos estáticos
python manage.py collectstatic --no-input

# Ejecutar migraciones
python manage.py migrate

# Nota: En producción Render usará gunicorn, no runserver
# Este comando no se ejecutará en Render pero es útil para desarrollo local
# Si estamos en local, podemos ejecutar el servidor de desarrollo
if [ "$RENDER" != "true" ]; then
  echo "Si deseas iniciar el servidor de desarrollo, ejecuta: python manage.py runserver"
fi