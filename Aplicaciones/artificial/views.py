

from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import Paciente
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report, f1_score

# === CORRECCIÓN: Configurar backend ANTES de importar pyplot ===
import matplotlib
matplotlib.use('Agg')  # Backend no interactivo
import matplotlib.pyplot as plt

import seaborn as sns
import io
import base64

# ... (el resto del código permanece igual) ...


# === Generación de datos de ejemplo para entrenamiento inicial ===
def generar_datos_ejemplo():
    np.random.seed(42)
    n_samples = 200
    edad = np.random.randint(18, 80, n_samples)
    imc = np.clip(np.random.normal(26, 5, n_samples), 16, 40)
    actividad_fisica = np.clip(np.random.normal(3, 2, n_samples), 0, 20)
    presion_arterial = np.clip(np.random.normal(120, 15, n_samples), 90, 180)
    antecedentes = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
    glucosa = []
    riesgo = []
    for i in range(n_samples):
        base = 85  # base normal
        if edad[i] > 50: base += 10
        if imc[i] >= 30: base += 10
        if actividad_fisica[i] < 1: base += 7
        if presion_arterial[i] > 135: base += 7
        if antecedentes[i]: base += 12
        valor = base + np.random.normal(0, 8)
        glucosa.append(valor)
        # Etiqueta de riesgo realista
        if valor >= 126:
            riesgo.append(1)
        elif valor >= 100 and (imc[i] >= 25 or antecedentes[i] or edad[i] > 45):
            riesgo.append(1)
        else:
            riesgo.append(0)
    data = pd.DataFrame({
        'edad': edad,
        'imc': imc,
        'actividad_fisica': actividad_fisica,
        'presion_arterial': presion_arterial,
        'antecedentes_familiares': antecedentes,
        'glucosa_ayunas': glucosa,
        'riesgo_diabetes': riesgo
    })
    return data

# === MODELOS DE MACHINE LEARNING ===
class ModeloGlucosa:
    def __init__(self):
        self.modelo = LinearRegression()
        self.scaler = StandardScaler()
        self.trained = False
        self.entrenar_con_datos_ejemplo()
    
    def entrenar_con_datos_ejemplo(self):
        data = generar_datos_ejemplo()
        X = data[['edad', 'imc', 'actividad_fisica', 'presion_arterial', 'antecedentes_familiares']]
        y = data['glucosa_ayunas']
        X_scaled = self.scaler.fit_transform(X)
        self.modelo.fit(X_scaled, y)
        self.trained = True
    
    def predecir(self, edad, imc, actividad_fisica, presion_arterial, antecedentes_familiares):
        if not self.trained:
            return None
        X = np.array([[edad, imc, actividad_fisica, presion_arterial, int(antecedentes_familiares)]])
        X_scaled = self.scaler.transform(X)
        return self.modelo.predict(X_scaled)[0]

class ModeloRiesgo:
    def __init__(self):
        self.modelo = LogisticRegression(random_state=42)
        self.scaler = StandardScaler()
        self.trained = False
        self.entrenar_con_datos_ejemplo()
    
    def entrenar_con_datos_ejemplo(self):
        data = generar_datos_ejemplo()
        X = data[['edad', 'imc', 'actividad_fisica', 'presion_arterial', 'antecedentes_familiares', 'glucosa_ayunas']]
        y = data['riesgo_diabetes']
        X_scaled = self.scaler.fit_transform(X)
        self.modelo.fit(X_scaled, y)
        self.trained = True
    
    def predecir(self, edad, imc, actividad_fisica, presion_arterial, antecedentes_familiares, glucosa_ayunas):
        if not self.trained:
            return None, None
        X = np.array([[edad, imc, actividad_fisica, presion_arterial, int(antecedentes_familiares), glucosa_ayunas]])
        X_scaled = self.scaler.transform(X)
        pred_proba = self.modelo.predict_proba(X_scaled)[0][1]
        pred_clase = 1 if pred_proba > 0.5 else 0
        return pred_clase, pred_proba * 100

modelo_glucosa = ModeloGlucosa()
modelo_riesgo = ModeloRiesgo()

# === FUNCIÓN CLÍNICA PARA RIESGO AJUSTADO ===
def predecir_riesgo_ajustado(edad, imc, actividad_fisica, presion_arterial, antecedentes_familiares, glucosa_ayunas):
    """
    Lógica clínica y de IA para riesgo de diabetes tipo 2.
    - Si glucosa < 100 mg/dL: sin riesgo.
    - Si glucosa 100-125: riesgo solo si hay al menos 2 factores de riesgo.
    - Si glucosa >= 126: riesgo alto.
    """
    if glucosa_ayunas < 100:
        return 0, 0.0
    elif 100 <= glucosa_ayunas < 126:
        puntaje = 0
        if edad >= 45:
            puntaje += 1
        if imc >= 25:
            puntaje += 1
        if antecedentes_familiares:
            puntaje += 1
        if puntaje >= 2:
            return 1, 60.0
        else:
            return 0, 25.0
    else:
        return 1, 90.0

# === VISTAS DJANGO ===

def inicio(request):
    # Obtener datos de la base de datos
    total_pacientes = Paciente.objects.count()
    pacientes_con_riesgo = Paciente.objects.filter(riesgo_diabetes=True).count()
    
    # Calcular el porcentaje con más precisión y manejo de casos límite
    if total_pacientes > 0:
        porcentaje_riesgo = round((pacientes_con_riesgo / total_pacientes) * 100, 1)
    else:
        porcentaje_riesgo = 0.0
    
    # Añadir registro para depuración
    print(f"[DEBUG] Inicio - Total: {total_pacientes}, Con riesgo: {pacientes_con_riesgo}, Porcentaje: {porcentaje_riesgo}%")
    
    # Generar gráfico si hay datos
    fig = generar_grafico_riesgo() if total_pacientes > 0 else None
    
    # Preparar datos para la plantilla
    contexto = {
        'total_pacientes': total_pacientes,
        'pacientes_con_riesgo': pacientes_con_riesgo,
        'porcentaje_riesgo': porcentaje_riesgo,
        'grafico': fig
    }
    return render(request, 'inicio.html', contexto)

def generar_grafico_riesgo():
    if Paciente.objects.count() == 0:
        return None
    pacientes = Paciente.objects.all()
    data = []
    for p in pacientes:
        data.append({
            'edad': p.edad,
            'imc': p.imc,
            'glucosa': p.glucosa_ayunas,
            'riesgo': 'Con riesgo' if p.riesgo_diabetes else 'Sin riesgo'
        })
    df = pd.DataFrame(data)
    fig, ax = plt.subplots(figsize=(8, 5))
    riesgo_counts = df['riesgo'].value_counts()
    riesgo_counts.plot(kind='bar', ax=ax, color=['green', 'red'])
    ax.set_title('Distribución de Pacientes por Riesgo de Diabetes')
    ax.set_ylabel('Número de Pacientes')
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    return f'data:image/png;base64,{img_str}'

def lista_pacientes(request):
    pacientes = Paciente.objects.all()
    return render(request, 'lista_pacientes.html', {'pacientes': pacientes})
import json
from django.shortcuts import render, redirect
from .models import Paciente

def crear_paciente(request):
    if request.method == 'POST':
        # --- NUEVO BLOQUE PARA INSERCIONES MASIVAS ---
        datos_masivos = request.POST.get('datos_masivos', '')
        if datos_masivos:
            try:
                lista_datos = json.loads(datos_masivos)
                pacientes = []
                # Obtener conjunto de pacientes existentes (nombre, edad) para evitar duplicados
                existentes = set(Paciente.objects.values_list('nombre', 'edad'))
                for d in lista_datos:
                    nombre = d['nombre']
                    edad = int(d['edad'])
                    # Validar duplicado
                    if (nombre, edad) in existentes:
                        # Saltar paciente duplicado
                        continue
                    imc = float(d['imc'])
                    actividad_fisica = float(d['actividad_fisica'])
                    presion_arterial = int(d['presion_arterial'])
                    antecedentes_familiares = True if d['antecedentes_familiares'] in ['True', 'true', True, 1, '1'] else False
                    glucosa_ayunas = float(d['glucosa_ayunas']) if d['glucosa_ayunas'] else modelo_glucosa.predecir(
                        edad, imc, actividad_fisica, presion_arterial, antecedentes_familiares
                    )
                    riesgo_pred, prob_riesgo = predecir_riesgo_ajustado(
                        edad, imc, actividad_fisica, presion_arterial, antecedentes_familiares, glucosa_ayunas
                    )
                    paciente = Paciente(
                        nombre=nombre,
                        edad=edad,
                        imc=imc,
                        actividad_fisica=actividad_fisica,
                        presion_arterial=presion_arterial,
                        antecedentes_familiares=antecedentes_familiares,
                        glucosa_ayunas=glucosa_ayunas,
                        riesgo_diabetes=bool(riesgo_pred),
                        probabilidad_riesgo=prob_riesgo
                    )
                    pacientes.append(paciente)
                if not pacientes:
                    # Si no hay pacientes nuevos para insertar
                    return render(request, 'formulario_paciente.html', {
                        'error': 'Todos los pacientes del archivo ya existen en la base de datos.',
                        'accion': 'crear'
                    })
                Paciente.objects.bulk_create(pacientes)
                return redirect('lista_pacientes')
            except Exception as e:
                return render(request, 'formulario_paciente.html', {
                    'error': f'Error al procesar inserciones masivas: {str(e)}',
                    'accion': 'crear'
                })
        # --- FIN BLOQUE MASIVO ---

        # --- RESTO DE TU LÓGICA INDIVIDUAL ---
        try:
            nombre = request.POST.get('nombre').strip()
            edad = int(request.POST.get('edad'))
            # Validar duplicados para paciente individual
            if Paciente.objects.filter(nombre=nombre, edad=edad).exists():
                return render(request, 'formulario_paciente.html', {
                    'error': f'Ya existe un paciente con nombre "{nombre}" y edad {edad}.',
                    'accion': 'crear',
                    'paciente': request.POST  # Para rellenar el formulario con los datos ingresados
                })

            imc = float(request.POST.get('imc'))
            actividad_fisica = float(request.POST.get('actividad_fisica'))
            presion_arterial = int(request.POST.get('presion_arterial'))
            antecedentes_str = request.POST.get('antecedentes_familiares')
            antecedentes_familiares = True if antecedentes_str == 'True' else False
            glucosa_input = request.POST.get('glucosa_ayunas')
            if glucosa_input:
                glucosa_ayunas = float(glucosa_input)
            else:
                glucosa_ayunas = modelo_glucosa.predecir(
                    edad, imc, actividad_fisica, presion_arterial, antecedentes_familiares
                )
            riesgo_pred, prob_riesgo = predecir_riesgo_ajustado(
                edad, imc, actividad_fisica, presion_arterial, antecedentes_familiares, glucosa_ayunas
            )
            paciente = Paciente(
                nombre=nombre,
                edad=edad,
                imc=imc,
                actividad_fisica=actividad_fisica,
                presion_arterial=presion_arterial,
                antecedentes_familiares=antecedentes_familiares,
                glucosa_ayunas=glucosa_ayunas,
                riesgo_diabetes=bool(riesgo_pred),
                probabilidad_riesgo=prob_riesgo
            )
            paciente.save()
            return redirect('lista_pacientes')
        except Exception as e:
            return render(request, 'formulario_paciente.html', {
                'error': f'Error al procesar datos: {str(e)}',
                'accion': 'crear',
                'paciente': request.POST
            })
    return render(request, 'formulario_paciente.html', {'accion': 'crear'})


def editar_paciente(request, paciente_id):
    paciente = get_object_or_404(Paciente, id=paciente_id)
    if request.method == 'POST':
        try:
            paciente.nombre = request.POST.get('nombre')
            paciente.edad = int(request.POST.get('edad'))
            paciente.imc = float(request.POST.get('imc'))
            paciente.actividad_fisica = float(request.POST.get('actividad_fisica'))
            paciente.presion_arterial = int(request.POST.get('presion_arterial'))
            antecedentes_str = request.POST.get('antecedentes_familiares')
            paciente.antecedentes_familiares = True if antecedentes_str == 'True' else False
            glucosa_input = request.POST.get('glucosa_ayunas')
            if glucosa_input:
                paciente.glucosa_ayunas = float(glucosa_input)
            else:
                paciente.glucosa_ayunas = modelo_glucosa.predecir(
                    paciente.edad, paciente.imc, paciente.actividad_fisica, 
                    paciente.presion_arterial, paciente.antecedentes_familiares
                )
            # USAR FUNCIÓN CLÍNICA
            riesgo_pred, prob_riesgo = predecir_riesgo_ajustado(
                paciente.edad, paciente.imc, paciente.actividad_fisica,
                paciente.presion_arterial, paciente.antecedentes_familiares, 
                paciente.glucosa_ayunas
            )
            print(f"[DEBUG] Editar paciente: {paciente.nombre}, Riesgo: {riesgo_pred}, Prob: {prob_riesgo:.2f}")
            paciente.riesgo_diabetes = bool(riesgo_pred)
            paciente.probabilidad_riesgo = prob_riesgo
            paciente.save()
            return redirect('lista_pacientes')
        except Exception as e:
            return render(request, 'formulario_paciente.html', {
                'paciente': paciente,
                'error': f'Error al procesar datos: {str(e)}',
                'accion': 'editar'
            })
    return render(request, 'formulario_paciente.html', {
        'paciente': paciente,
        'accion': 'editar'
    })

def eliminar_paciente(request, paciente_id):
    paciente = get_object_or_404(Paciente, id=paciente_id)
    if request.method == 'POST':
        paciente.delete()
        return redirect('lista_pacientes')
    return render(request, 'confirmar_eliminacion.html', {'paciente': paciente})

@csrf_exempt
def api_predecir(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            edad = int(data.get('edad'))
            imc = float(data.get('imc'))
            actividad_fisica = float(data.get('actividad_fisica'))
            presion_arterial = int(data.get('presion_arterial'))
            antecedentes_familiares = True if data.get('antecedentes_familiares') in [True, 'True', 1, '1'] else False
            glucosa_input = data.get('glucosa_ayunas')
            if glucosa_input:
                glucosa_ayunas = float(glucosa_input)
            else:
                glucosa_ayunas = modelo_glucosa.predecir(
                    edad, imc, actividad_fisica, presion_arterial, antecedentes_familiares
                )
            # USAR FUNCIÓN CLÍNICA
            riesgo_pred, prob_riesgo = predecir_riesgo_ajustado(
                edad, imc, actividad_fisica, presion_arterial, antecedentes_familiares, glucosa_ayunas
            )
            print(f"[DEBUG] API predicción: Riesgo: {riesgo_pred}, Prob: {prob_riesgo:.2f}")
            return JsonResponse({
                'glucosa_predicha': round(glucosa_ayunas, 2),
                'riesgo_diabetes': bool(riesgo_pred),
                'probabilidad_riesgo': round(prob_riesgo, 2)
            })
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)
    return JsonResponse({'error': 'Método no permitido'}, status=405)




def ver_estadisticas(request):
    if Paciente.objects.count() < 5:
        return render(request, 'estadisticas.html', {'error': 'Se necesitan al menos 5 pacientes para generar estadísticas'})
    
    pacientes = Paciente.objects.all()
    data = []
    for p in pacientes:
        data.append({
            'edad': p.edad,
            'imc': p.imc,
            'actividad_fisica': p.actividad_fisica,
            'presion_arterial': p.presion_arterial,
            'antecedentes_familiares': int(p.antecedentes_familiares),
            'glucosa_ayunas': p.glucosa_ayunas,
            'riesgo_diabetes': int(p.riesgo_diabetes) if p.riesgo_diabetes is not None else 0
        })
    df = pd.DataFrame(data)

    # Matriz de correlación
    correlacion = df.corr()
    fig, ax = plt.subplots(figsize=(10,8))
    sns.heatmap(correlacion, annot=True, cmap='coolwarm', ax=ax)
    ax.set_title('Matriz de Correlación entre Variables')
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    img_correlacion = base64.b64encode(buf.read()).decode('utf-8')

    # Múltiples gráficos de Regresión Lineal
    regresion_plots = []
    variables_x = ['edad', 'imc', 'actividad_fisica', 'presion_arterial']
    
    for var_x in variables_x:
        fig_lin, ax_lin = plt.subplots(figsize=(7, 5))
        X_lin = df[[var_x]]
        y_lin = df['glucosa_ayunas']
        
        # Ajuste de regresión lineal
        model_lin = LinearRegression()
        model_lin.fit(X_lin, y_lin)
        y_pred_lin = model_lin.predict(X_lin)
        
        # Calcular R-squared
        r_squared = r2_score(y_lin, y_pred_lin)
        
        # Graficar
        ax_lin.scatter(X_lin, y_lin, color='blue', alpha=0.6, label='Datos reales')
        ax_lin.plot(X_lin, y_pred_lin, color='red', linewidth=2, 
                    label=f'Ajuste lineal (R² = {r_squared:.2f})')
        ax_lin.set_xlabel(var_x.replace('_', ' ').title())
        ax_lin.set_ylabel('Glucosa en ayunas')
        ax_lin.set_title(f'Regresión Lineal: {var_x.replace("_", " ").title()} vs Glucosa')
        ax_lin.legend()
        
        # Guardar gráfico
        buf_lin = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf_lin, format='png')
        plt.close(fig_lin)
        buf_lin.seek(0)
        img_lin_str = base64.b64encode(buf_lin.read()).decode('utf-8')
        
        # Guardar información del gráfico
        regresion_plots.append({
            'variable': var_x,
            'grafico': f'data:image/png;base64,{img_lin_str}',
            'r_squared': r_squared
        })

    # Regresión Logística con matriz de confusión
    # Preparar datos para el modelo
    y_log = df['riesgo_diabetes']
    X_log = df[['glucosa_ayunas', 'imc', 'edad', 'presion_arterial']]
    
    # Dividir datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X_log, y_log, test_size=0.3, random_state=42)
    
    # Escalar características
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Entrenar modelo de regresión logística
    model_log = LogisticRegression(random_state=42)
    model_log.fit(X_train_scaled, y_train)
    
    # Predicciones
    y_pred = model_log.predict(X_test_scaled)
    
    # Crear matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    
    # Graficar matriz de confusión
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Sin Riesgo', 'Con Riesgo'],
                yticklabels=['Sin Riesgo', 'Con Riesgo'])
    plt.title('Matriz de Confusión - Predicción de Riesgo de Diabetes')
    plt.xlabel('Predicción')
    plt.ylabel('Valor Real')
    buf_cm = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf_cm, format='png')
    plt.close()
    buf_cm.seek(0)
    img_cm_str = base64.b64encode(buf_cm.read()).decode('utf-8')
    
    # Generar reporte de clasificación
    reporte_clasificacion = classification_report(y_test, y_pred, 
                                                  target_names=['Sin Riesgo', 'Con Riesgo'])
    
    # Graficar Regresión Logística (probabilidad de glucosa)
    fig_log, ax_log = plt.subplots(figsize=(7, 5))
    x_vals = np.linspace(X_log['glucosa_ayunas'].min(), X_log['glucosa_ayunas'].max(), 200).reshape(-1, 1)
    x_vals_scaled = scaler.transform(np.column_stack([x_vals, 
                                                      np.full_like(x_vals, X_log['imc'].mean()),
                                                      np.full_like(x_vals, X_log['edad'].mean()),
                                                      np.full_like(x_vals, X_log['presion_arterial'].mean())]))
    y_prob = model_log.predict_proba(x_vals_scaled)[:, 1]
    
    ax_log.scatter(X_log['glucosa_ayunas'], y_log, color='green', alpha=0.5, label='Datos reales')
    ax_log.plot(x_vals, y_prob, color='orange', linewidth=2, label='Probabilidad estimada')
    ax_log.set_xlabel('Glucosa en ayunas')
    ax_log.set_ylabel('Probabilidad de riesgo')
    ax_log.set_title('Regresión Logística: Glucosa vs Riesgo')
    ax_log.legend()
    
    buf_log = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf_log, format='png')
    plt.close(fig_log)
    buf_log.seek(0)
    img_log_str = base64.b64encode(buf_log.read()).decode('utf-8')

    contexto = {
        'correlacion': correlacion.to_html(),
        'grafico_correlacion': f'data:image/png;base64,{img_correlacion}',
        'graficos_regresion_lineal': regresion_plots,
        'grafico_regresion_logistica': f'data:image/png;base64,{img_log_str}',
        'matriz_confusion': f'data:image/png;base64,{img_cm_str}',
        'reporte_clasificacion': reporte_clasificacion.replace('\n', '<br>'),
        'total_pacientes': Paciente.objects.count()
    }
    return render(request, 'estadisticas.html', contexto)