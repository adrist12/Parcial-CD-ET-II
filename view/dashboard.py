# ==============================================================================
# DASHBOARD INTERACTIVO — PREDICCIÓN DE CONSUMO DE SUSTANCIAS DURAS
# ==============================================================================
# Propósito: Interfaz web interactiva para predicción de consumo de sustancias
#           basada en datos sociodemográficos y de historia de calle.
#
# Framework: Streamlit (aplicaciones web Python sin HTML/CSS/JS)
# 
# Características:
#   - Interfaz intuitiva con formularios
#   - Carga eficiente del modelo entrenado (caché)
#   - Predicción en tiempo real
#   - Visualización de resultados
#   - Estructura modular para agregar gráficas
#
# Ejecución:
#   streamlit run view/dashboard.py
# ==============================================================================

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from model.interview_model import PredictiveModel
import warnings
warnings.filterwarnings('ignore')


# ==============================================================================
# CONFIGURACIÓN INICIAL DE STREAMLIT
# ==============================================================================

# Configuración de la página (debe ser la primera línea de Streamlit)
st.set_page_config(
    page_title="🔮 Predictor CDT — Consumo de Sustancias Duras",
    page_icon="🔮",
    layout="wide",                    # layout ancho para aprovechar espacio
    initial_sidebar_state="expanded"  # sidebar expandida por defecto
)

# Aplicar tema personalizado (colores profesionales)
st.markdown("""
    <style>
    /* Estilos CSS para mejorar la apariencia del dashboard */
    
    /* Color principal: azul oscuro */
    :root {
        --primary-color: #1f77b4;
        --background-color: #f8f9fa;
        --text-color: #2c3e50;
    }
    
    /* Headers */
    h1 {
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    h2 {
        color: #34495e;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 0.5rem;
    }
    
    /* Containers */
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .info-box {
        background: #e8f4f8;
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)


# ==============================================================================
# FUNCIONES CON CACHE (para optimizar performance)
# ==============================================================================
# @st.cache_data: carga datos una sola vez y reutiliza en sesión
# @st.cache_resource: carga recursos (modelos, conexiones) una sola vez

@st.cache_resource
def cargar_modelo(ruta_modelo):
    """
    Carga el modelo entrenado desde joblib (CACHE).
    
    La decoración @st.cache_resource hace que Streamlit guarde el modelo
    en memoria y lo reutilice en cada interacción del usuario (en lugar
    de recargar cada vez que el usuario interactúa con el dashboard).
    
    Args:
        ruta_modelo (str): ruta al archivo .joblib del modelo
    
    Returns:
        Pipeline: modelo entrenado listo para predicciones
    
    Excepciones:
        - FileNotFoundError: si el modelo no existe
        - Muestra error en Streamlit
    """
    try:
        modelo = joblib.load(ruta_modelo)
        return modelo
    except FileNotFoundError:
        st.error(f"❌ Modelo no encontrado: {ruta_modelo}")
        st.info("ℹ️  Ejecuta primero: `python main.py` para entrenar el modelo")
        st.stop()


@st.cache_data
def cargar_diccionario(ruta_diccionario):
    """
    Carga el diccionario de preguntas/respuestas (CACHE).
    
    Args:
        ruta_diccionario (str): ruta al archivo dictionary.json
    
    Returns:
        dict: configuración con renombres, preguntas, respuestas, etc.
    """
    try:
        with open(ruta_diccionario, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"❌ Diccionario no encontrado: {ruta_diccionario}")
        st.stop()


@st.cache_data
def cargar_datos_predictor():
    """
    Carga lista de predictores para referencia (CACHE).
    
    Returns:
        list: nombres de las variables predictoras
    """
    return PredictiveModel.FEATURES

# ==============================================================================
# FUNCIONES DE UTILIDAD PARA LA INTERFAZ
# ==============================================================================

def crear_formulario_datos():
    """
    Crea un formulario Streamlit interactivo para capturar datos del usuario.
    
    Estructura:
    - Columnas para organizar inputs horizontalmente
    - Selectores para variables categóricas
    - Sliders y number inputs para variables numéricas
    - Checkboxes para variables binarias
    
    Decisiones de diseño:
    - Edad: slider 0-100 (realista para personas en calle)
    - Género: select (Hombre/Mujer de diccionario)
    - Sustancias: checkboxes (múltiples selecciones)
    - Tiempo en calle: number inputs separados (años y meses)
    
    Returns:
        dict: diccionario con todos los datos capturados
        
    Notas:
    - Los valores del diccionario se mapean a códigos numéricos (1, 2, 3...)
    - NaN en checkboxes no marcados = "no consume" (2 en diccionario)
    """
    
    dicc = cargar_diccionario("data/dictionary.json")
    datos = {}
    
    # ========================================================================
    # SECCIÓN 1: INFORMACIÓN SOCIODEMOGRÁFICA
    # ========================================================================
    st.markdown("### 👤 Información Sociodemográfica")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # EDAD: slider de 0 a 100 años
        edad = st.slider(
            "Edad",
            min_value=0,
            max_value=100,
            value=35,
            step=1,
            help="Rango típico en personas en calle: 18-70 años"
        )
        datos['edad'] = edad
    
    with col2:
        # GÉNERO: select (1=Hombre, 2=Mujer)
        genero_label = st.selectbox(
            "Género",
            ["Hombre", "Mujer"],
            index=0,
            help="Del diccionario P9"
        )
        datos['genero'] = 1 if genero_label == "Hombre" else 2
    
    with col3:
        # ORIENTACIÓN SEXUAL: select
        orientacion_label = st.selectbox(
            "Orientación Sexual",
            ["Heterosexual", "Homosexual (Gay/Lesbiana)", "Bisexual", "Otro"],
            index=0,
            help="Del diccionario P34"
        )
        orientacion_map = {
            "Heterosexual": 1,
            "Homosexual (Gay/Lesbiana)": 2,
            "Bisexual": 3,
            "Otro": 4
        }
        datos['orientacion_sexual'] = orientacion_map[orientacion_label]
    
    # Segunda fila sociodemográfica
    col1, col2 = st.columns(2)
    
    with col1:
        # NIVEL EDUCATIVO: select
        nivel_edu_label = st.selectbox(
            "Nivel Educativo",
            [
                "Ninguno", "Preescolar", "Primaria incompleta", "Primaria completa",
                "Secundaria/Bachillerato incompleto", "Secundaria/Bachillerato completo",
                "Técnico o Tecnológico", "Universitario/Profesional"
            ],
            index=0,
            help="Del diccionario P28"
        )
        nivel_edu_map = {
            "Ninguno": 1, "Preescolar": 2, "Primaria incompleta": 3,
            "Primaria completa": 4, "Secundaria/Bachillerato incompleto": 5,
            "Secundaria/Bachillerato completo": 6, "Técnico o Tecnológico": 7,
            "Universitario/Profesional": 8
        }
        datos['nivel_educativo'] = nivel_edu_map[nivel_edu_label]
    
    with col2:
        # SABE LEER/ESCRIBIR: select (1=Sí sin dificultad, 2=Sí con dificultad, 3=No)
        lee_escribe_label = st.selectbox(
            "¿Sabe leer y escribir?",
            ["Sí, sin dificultad", "Sí, con alguna dificultad", "No puede hacerlo"],
            index=0,
            help="Del diccionario P16S9"
        )
        lee_escribe_map = {
            "Sí, sin dificultad": 1,
            "Sí, con alguna dificultad": 2,
            "No puede hacerlo": 3
        }
        datos['sabe_leer_escribir'] = lee_escribe_map[lee_escribe_label]
    
    # ========================================================================
    # SECCIÓN 2: HISTORIA EN CALLE
    # ========================================================================
    st.markdown("### 🏙️ Historia en Calle")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # AÑOS EN CALLE
        años_calle = st.number_input(
            "Años en calle",
            min_value=0,
            max_value=80,
            value=5,
            step=1,
            help="Número de años viviendo en la calle"
        )
    
    with col2:
        # MESES EN CALLE (adicionales)
        meses_calle = st.number_input(
            "Meses adicionales",
            min_value=0,
            max_value=11,
            value=0,
            step=1,
            help="Meses adicionales (0-11)"
        )
    
    with col3:
        # TIEMPO TOTAL EN MESES (calculado)
        tiempo_total_meses = años_calle * 12 + meses_calle
        st.metric(
            "Tiempo total en calle",
            f"{tiempo_total_meses} meses",
            f"≈ {años_calle}.{meses_calle} años"
        )
    
    datos['tiempo_total_calle_meses'] = tiempo_total_meses
    
    # Razones (selects)
    col1, col2 = st.columns(2)
    
    with col1:
        # RAZÓN DE INICIO EN VIDA DE CALLE
        razon_inicio_label = st.selectbox(
            "Razón por la que comenzó en calle",
            [
                "Consumo de sustancias", "Dificultades/conflictos familiares",
                "Falta de oportunidades laborales", "Gusto por la calle",
                "Desplazamiento forzado/violencia", "Problemas de salud mental",
                "Abandono/falta de red de apoyo", "Otra razón"
            ],
            index=1,
            help="Del diccionario P22"
        )
        razon_inicio_map = {
            "Consumo de sustancias": 1, "Dificultades/conflictos familiares": 2,
            "Falta de oportunidades laborales": 3, "Gusto por la calle": 4,
            "Desplazamiento forzado/violencia": 5, "Problemas de salud mental": 6,
            "Abandono/falta de red de apoyo": 7, "Otra razón": 8
        }
        datos['razon_inicio_vida_calle'] = razon_inicio_map[razon_inicio_label]
    
    with col2:
        # RAZÓN POR LA QUE CONTINÚA EN CALLE
        razon_continua_label = st.selectbox(
            "Razón por la que continúa en calle",
            [
                "Dependencia/consumo de sustancias", "Falta de oportunidades/empleo",
                "Acostumbramiento/gusto por la calle", "No tener apoyo familiar",
                "Problemas legales/judiciales", "Otra razón"
            ],
            index=2,
            help="Del diccionario P24"
        )
        razon_continua_map = {
            "Dependencia/consumo de sustancias": 1, "Falta de oportunidades/empleo": 2,
            "Acostumbramiento/gusto por la calle": 3, "No tener apoyo familiar": 4,
            "Problemas legales/judiciales": 5, "Otra razón": 6
        }
        datos['razon_continua_en_calle'] = razon_continua_map[razon_continua_label]
    
    # ========================================================================
    # SECCIÓN 3: SUBSISTENCIA
    # ========================================================================
    st.markdown("### 💰 Forma de Subsistencia")
    
    forma_dinero_label = st.selectbox(
        "¿Cómo obtiene dinero?",
        [
            "Reciclaje/recolección de materiales", "Venta ambulante/limpieza vidrios",
            "Mendicidad/pedir dinero", "Trabajos informales u ocasionales",
            "Actividades ilegales", "Ayudas institucionales/fundaciones", "Otra forma"
        ],
        index=2,
        help="Del diccionario P29"
    )
    forma_dinero_map = {
        "Reciclaje/recolección de materiales": 1,
        "Venta ambulante/limpieza vidrios": 2,
        "Mendicidad/pedir dinero": 3,
        "Trabajos informales u ocasionales": 4,
        "Actividades ilegales": 5,
        "Ayudas institucionales/fundaciones": 6,
        "Otra forma": 7
    }
    datos['forma_obtener_dinero'] = forma_dinero_map[forma_dinero_label]
    
    # ========================================================================
    # SECCIÓN 4: DIAGNÓSTICOS DE SALUD
    # ========================================================================
    st.markdown("### 🏥 Diagnósticos de Salud")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # HIPERTENSIÓN
        dx_hipertension = st.checkbox(
            "Hipertensión",
            value=False,
            help="¿Ha sido diagnosticado con hipertensión?"
        )
        datos['dx_hipertension'] = 1 if dx_hipertension else 2
    
    with col2:
        # TUBERCULOSIS
        dx_tuberculosis = st.checkbox(
            "Tuberculosis",
            value=False,
            help="¿Ha sido diagnosticado con tuberculosis?"
        )
        datos['dx_tuberculosis'] = 1 if dx_tuberculosis else 2
    
    with col3:
        # VIH/SIDA
        dx_vih_sida = st.checkbox(
            "VIH/SIDA",
            value=False,
            help="¿Ha sido diagnosticado con VIH/SIDA?"
        )
        datos['dx_vih_sida'] = 1 if dx_vih_sida else 2
    
    with col4:
        # ACTIVIDADES SIN ESFUERZO FÍSICO
        actividades_label = st.selectbox(
            "¿Puede realizar actividades sin esfuerzo?",
            ["Sí, sin dificultad", "Sí, con dificultad", "No puede hacerlo"],
            index=0,
            key="actividades",
            help="Del diccionario P16S9"
        )
        actividades_map = {
            "Sí, sin dificultad": 1,
            "Sí, con dificultad": 2,
            "No puede hacerlo": 3
        }
        datos['actividades_sin_esfuerzo_fisico'] = actividades_map[actividades_label]
    
    # ========================================================================
    # SECCIÓN 5: CONSUMO DE SUSTANCIAS
    # ========================================================================
    st.markdown("### 🚬 Consumo de Sustancias")
    st.info(
        "ℹ️  Selecciona las sustancias que consume. Las NO seleccionadas se asumen como 'no consumo'.",
        icon="ℹ️"
    )
    
    col1, col2, col3, col4 = st.columns(4)
    
    sustancias = {
        'consume_cigarrillo': ('Cigarrillo', col1),
        'consume_alcohol': ('Alcohol', col2),
        'consume_marihuana': ('Marihuana', col3),
        'consume_inhalantes': ('Inhalantes', col4)
    }
    
    for variable, (label, col) in sustancias.items():
        with col:
            consume = st.checkbox(
                label,
                value=False,
                help=f"¿Consume {label}?"
            )
            datos[variable] = 1 if consume else 2  # 1=Sí, 2=No
    
    return datos


def preparar_datos_prediccion(datos_usuario, predictores):
    """
    Convierte los datos del usuario a formato DataFrame para predicción.
    
    Pasos:
    1. Crear diccionario con todos los predictores
    2. Asignar valores del usuario
    3. Asignar NaN a variables no capturadas (no aplicable aquí)
    4. Convertir a DataFrame con una única fila
    5. Retornar en formato esperado por el modelo
    
    Args:
        datos_usuario (dict): datos capturados del formulario
        predictores (list): lista de nombres de predictores
    
    Returns:
        DataFrame: una fila con todos los predictores en el orden correcto
    
    Nota: El modelo espera exactamente estos predictores en este orden
    """
    # Inicializar con NaN (por si falta algún predictor)
    fila = {pred: np.nan for pred in predictores}
    fila.update(datos_usuario)
    df = pd.DataFrame([fila])[predictores]  # filtra y ordena automáticamente
    return df


def obtener_interpretacion_riesgo(probabilidad):
    """
    Interpreta la probabilidad predicha en categoría de riesgo.
    
    Categorías:
    - [0.0, 0.25): Bajo riesgo (verde)
    - [0.25, 0.50): Riesgo moderado (amarillo)
    - [0.50, 0.75): Riesgo alto (naranja)
    - [0.75, 1.0]: Riesgo muy alto (rojo)
    
    Args:
        probabilidad (float): probabilidad predicha [0, 1]
    
    Returns:
        tuple: (categoría, color, descripción)
    """
    if probabilidad < 0.25:
        return "🟢 BAJO RIESGO", "green", "Baja probabilidad de consumo de sustancias duras"
    elif probabilidad < 0.50:
        return "🟡 RIESGO MODERADO", "orange", "Probabilidad moderada de consumo de sustancias duras"
    elif probabilidad < 0.75:
        return "🟠 RIESGO ALTO", "red", "Alta probabilidad de consumo de sustancias duras"
    else:
        return "🔴 RIESGO MUY ALTO", "red", "Muy alta probabilidad de consumo de sustancias duras"


# ==============================================================================
# FUNCIONES DE VISUALIZACIÓN
# ==============================================================================

def mostrar_seccion_graficas():
    """
    Sección reservada para gráficas futuras.
    
    Aquí se pueden agregar:
    - Gráfica de importancia de variables
    - Distribución de probabilidades en conjunto de datos
    - Comparación con población
    - Evolución de riesgos
    - Etc.
    
    Por ahora, muestra estructura vacía.
    """
    st.markdown("### 📊 Análisis Detallado (Gráficas)")
    
    tab1, tab2, tab3 = st.tabs(
        ["📈 Importancia de Variables", "📉 Distribuciones", "🎯 Análisis Comparativo"]
    )
    
    with tab1:
        st.info(
            "📌 Espacio reservado para gráfica de importancia de variables\n"
            "Muestra qué variables el modelo usa más para predecir consumo.",
            icon="📌"
        )
    
    with tab2:
        st.info(
            "📌 Espacio reservado para distribuciones\n"
            "Comparar distribución del usuario vs población general.",
            icon="📌"
        )
    
    with tab3:
        st.info(
            "📌 Espacio reservado para análisis comparativo\n"
            "Posicionar usuario en contexto de la población.",
            icon="📌"
        )


# ==============================================================================
# FUNCIÓN PRINCIPAL (FLUJO DEL DASHBOARD)
# ==============================================================================

def main():
    """
    Función principal que orquesta el dashboard.
    
    Flujo:
    1. Header y descripción
    2. Cargar modelo y diccionario (cached)
    3. Sidebar con navegación
    4. Según selección, mostrar:
       - Formulario interactivo (página principal)
       - Información del modelo
       - Gráficas
    5. En formulario: capturar datos → predecir → mostrar resultado
    """
    
    # ========================================================================
    # HEADER
    # ========================================================================
    st.markdown("""
    <div style='text-align: center; padding: 2rem;'>
        <h1>🔮 Predictor CDT</h1>
        <h3 style='color: #666;'>Predicción de Consumo de Sustancias Duras</h3>
        <p style='font-size: 0.9rem; color: #999;'>
            Personas en Situación de Calle — Técnicas Cuantitativas II
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ========================================================================
    # SIDEBAR: NAVEGACIÓN
    # ========================================================================
    with st.sidebar:
        st.markdown("### 🗂️ Navegación")
        
        pagina = st.radio(
            "Selecciona una sección:",
            ["🎯 Hacer Predicción", "ℹ️  Información del Modelo", "📊 Gráficas"],
            index=0
        )
        
        st.markdown("---")
        
        # Información en sidebar
        st.markdown("### 📖 Información Útil")
        st.markdown("""
        **Target**: Consumo de sustancias duras
        - Basuco
        - Heroína
        - Cocaína
        
        **Modelos**: 3 modelos entrenados
        - Regresión Logística
        - Random Forest
        - Gradient Boosting
        
        **Predictores**: 19 variables
        - Sociodemográficas
        - Historia en calle
        - Salud
        - Sustancias
        """)
    
    # ========================================================================
    # CONTENIDO PRINCIPAL SEGÚN PÁGINA
    # ========================================================================
    
    if pagina == "🎯 Hacer Predicción":
        # Cargar modelo (cached)
        modelo = cargar_modelo("outputs/mejor_modelo.joblib")
        predictores = cargar_datos_predictor()
        
        st.markdown("### 📝 Formulario Interactivo")
        st.markdown("Completa los datos para generar una predicción:")
        
        # Crear formulario
        datos_usuario = crear_formulario_datos()
        
        # Botón para predecir
        if st.button(
            "🔮 GENERAR PREDICCIÓN",
            key="btn_predecir",
            help="Haz clic para predecir el riesgo de consumo de sustancias duras"
        ):
            # Mostrar loading
            with st.spinner("Procesando..."):
                # Preparar datos
                X_nuevo = preparar_datos_prediccion(datos_usuario, predictores)
                
                # Predecir
                prediccion = modelo.predict(X_nuevo)[0]
                probabilidad = modelo.predict_proba(X_nuevo)[0, 1]
            
            # ================================================================
            # MOSTRAR RESULTADO
            # ================================================================
            st.markdown("---")
            st.markdown("### 🎯 Resultado de la Predicción")
            
            # Obtener interpretación
            categoria, color, descripcion = obtener_interpretacion_riesgo(probabilidad)
            
            # Mostrar en caja grande
            col1, col2 = st.columns(2)
            
            with col1:
                # Categoría de riesgo
                st.markdown(f"""
                <div style='
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 2rem;
                    border-radius: 10px;
                    text-align: center;
                '>
                    <h2 style='margin: 0; color: white;'>{categoria}</h2>
                    <p style='margin: 0.5rem 0 0 0; color: #eee;'>{descripcion}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # Probabilidad exacta
                st.metric(
                    "Probabilidad de Consumo Duro",
                    f"{probabilidad:.2%}",
                    delta=f"{probabilidad*100:.1f}%",
                    delta_color="inverse"
                )
            
            # Barra de progreso visual
            st.progress(probabilidad, text="Riesgo: " + f"{probabilidad:.1%}")
            
            # Información adicional
            st.markdown("---")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Edad",
                    f"{datos_usuario['edad']} años",
                    delta="Factor de riesgo bien documentado"
                )
            
            with col2:
                st.metric(
                    "Tiempo en Calle",
                    f"{datos_usuario['tiempo_total_calle_meses']} meses",
                    delta="Mayor tiempo = mayor exposición"
                )
            
            with col3:
                sustancias_consumidas = sum([
                    datos_usuario['consume_cigarrillo'] == 1,
                    datos_usuario['consume_alcohol'] == 1,
                    datos_usuario['consume_marihuana'] == 1,
                    datos_usuario['consume_inhalantes'] == 1
                ])
                st.metric(
                    "Sustancias Consumidas",
                    f"{sustancias_consumidas} de 4",
                    delta="Escalada de consumo"
                )
            
            # Advertencia/recomendación
            st.markdown("---")
            st.warning(
                """
                ⚠️  **Nota Importante**:
                - Esta predicción es un modelo estadístico, no un diagnóstico médico.
                - Los resultados deben interpretarse en contexto clínico y profesional.
                - Para evaluación clínica, contacta a profesionales de salud mental.
                """,
                icon="⚠️"
            )
    
    elif pagina == "ℹ️  Información del Modelo":
        st.markdown("### 📚 Información del Modelo Predictivo")
        
        st.markdown("""
        #### 🎯 Objetivo
        Predecir si una persona en situación de calle consume sustancias de **alta dependencia**
        (basuco, heroína o cocaína) basándose en variables sociodemográficas, historia en calle
        y patrones de consumo.
        
        #### 📊 Datos de Entrenamiento
        - **Registros**: ~5,000 personas en situación de calle
        - **Balance de clases**: ~49% consumo duro / ~51% no consumo duro (perfecto balance)
        - **Edad promedio**: 37 años (consumidores) vs 46 años (no consumidores)
        
        #### 🤖 Modelos Entrenados
        1. **Regresión Logística**: Interpretable, coeficientes directos
        2. **Random Forest**: Captura no-linealidades, importancia variable
        3. **Gradient Boosting**: Máxima precisión predictiva (típicamente mejor)
        
        #### 📈 Métricas de Rendimiento
        - **AUC-ROC**: ~0.90 (excelente discriminación entre clases)
        - **F1-Score**: ~0.82 (buen balance precisión/recall)
        - **Validación cruzada**: 5 folds estratificados
        
        #### 🔍 Hallazgos Clave
        - **Inhalantes**: 82% de consumidores → duro (predictor más fuerte)
        - **Marihuana**: 67% de consumidores → duro (escalada evidente)
        - **Edad**: Consumidores en promedio 9 años más jóvenes
        - **Educación técnica**: Mayor riesgo (posible efecto urbano)
        
        #### 🏗️ Arquitectura
        Pipeline con:
        1. **Imputación**: Mediana (estrategia para categóricas)
        2. **Escalado**: StandardScaler (estandarización)
        3. **Modelo**: Mejor modelo seleccionado por AUC-ROC
        
        #### ⚠️ Limitaciones
        - Modelo entrenado con datos de una región específica
        - Puede haber sesgos en la recolección de datos
        - No reemplaza evaluación clínica profesional
        - Predicciones solo tan buenas como los datos de entrada
        """)
    
    elif pagina == "📊 Gráficas":
        mostrar_seccion_graficas()


# ==============================================================================
# PUNTO DE ENTRADA
# ==============================================================================

if __name__ == "__main__":
    """
    Punto de entrada de Streamlit.
    
    Uso:
        streamlit run view/dashboard.py
    
    Luego abre: http://localhost:8501
    """
    main()
