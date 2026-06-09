# ==============================================================================
# DASHBOARD — PREDICCIÓN DE CONSUMO DE SUSTANCIAS DURAS
# Ejecución: streamlit run view/dashboard.py
# ==============================================================================

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import numpy as np
import joblib, json
from model.interview_model import PredictiveModel
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CDT Risk Predictor",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ──────────────────────────────────────────────────────────────────────────────
# CSS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, .stApp {
    background: #F5F6F8 !important;
    font-family: 'Inter', sans-serif;
    color: #111318;
}
#MainMenu, footer, .stDeployButton,
header[data-testid="stHeader"] { display: none !important; }
section[data-testid="stSidebar"] { display: none !important; }

.shell {
    max-width: 860px;
    margin: 0 auto;
    padding: 2.5rem 1.5rem 6rem 1.5rem;
}

.wordmark {
    display: flex;
    align-items: baseline;
    gap: 0.6rem;
    margin-bottom: 2.5rem;
}
.wordmark-main {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    font-weight: 600;
    color: #111318;
    letter-spacing: 0.06em;
}
.wordmark-sep { color: #C5C9D1; font-size: 0.85rem; }
.wordmark-sub { font-size: 0.75rem; color: #8A909E; letter-spacing: 0.04em; }

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid #DFE2E8 !important;
    gap: 0 !important;
    padding: 0 !important;
    margin-bottom: 2.5rem;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: #8A909E !important;
    padding: 0.85rem 1.5rem !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    margin: 0 !important;
}
.stTabs [aria-selected="true"] {
    color: #111318 !important;
    border-bottom: 2px solid #2451FF !important;
}
.stTabs [data-baseweb="tab-panel"] { padding: 0 !important; }

/* ── SECTION HEADER ── */
.sec-header {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding-top: 2rem;
    margin-bottom: 1.25rem;
}
.sec-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: #2451FF;
    font-weight: 600;
}
.sec-line { flex: 1; height: 1px; background: #E8EBF0; }
.sec-label {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #8A909E;
}

/* ── FIELD ROW ── */
.field-meta-title {
    font-size: 0.875rem;
    font-weight: 600;
    color: #111318;
    margin-bottom: 0.3rem;
    margin-top: 0.15rem;
}
.field-meta-desc {
    font-size: 0.78rem;
    color: #6E7585;
    line-height: 1.6;
}

/* ── INPUTS ── */
div[data-testid="stSelectbox"] > label,
div[data-testid="stNumberInput"] > label,
div[data-testid="stSlider"] > label { display: none !important; }

div[data-testid="stSelectbox"] > div > div {
    border: 1.5px solid #DFE2E8 !important;
    border-radius: 6px !important;
    font-size: 0.875rem !important;
    background: #fff !important;
    min-height: 44px !important;
    color: #111318 !important;
}

/* Texto del valor seleccionado visible dentro del input cerrado */
div[data-testid="stSelectbox"] * {
    color: #111318 !important;
}

/* Dropdown abierto — fondo blanco y texto oscuro en todas las opciones */
[data-baseweb="popover"] * {
    color: #111318 !important;
    background-color: #fff !important;
}

/* Hover sobre una opción */
[data-baseweb="popover"] [role="option"]:hover {
    background-color: #EEF2FF !important;
    color: #111318 !important;
}

/* Opción actualmente seleccionada en el dropdown abierto */
[data-baseweb="popover"] [aria-selected="true"] {
    background-color: #EEF2FF !important;
    color: #2451FF !important;
}
div[data-testid="stSelectbox"] > div > div:focus-within {
    border-color: #2451FF !important;
    box-shadow: 0 0 0 3px rgba(36,81,255,0.08) !important;
}
div[data-testid="stNumberInput"] input {
    border: 1.5px solid #DFE2E8 !important;
    border-radius: 6px !important;
    font-size: 0.875rem !important;
    min-height: 44px !important;
}
div[data-testid="stNumberInput"] input:focus {
    border-color: #2451FF !important;
    box-shadow: 0 0 0 3px rgba(36,81,255,0.08) !important;
    outline: none !important;
}
div[data-testid="stSlider"] div[role="slider"] { background: #2451FF !important; }

/* checkbox label override */
div[data-testid="stCheckbox"] label span { display: inline !important; }
div[data-testid="stCheckbox"] label {
    font-size: 0.875rem !important;
    color: #111318 !important;
}
/* ── FIX: texto visible en opciones seleccionadas y dropdown ── */

/* El item actualmente seleccionado en el selectbox (el que se ve cerrado) */
div[data-testid="stSelectbox"] [data-baseweb="select"] [data-baseweb="tag"],
div[data-testid="stSelectbox"] [data-baseweb="select"] div[class*="placeholder"],
div[data-testid="stSelectbox"] [data-baseweb="select"] div[class*="singleValue"] {
    color: #111318 !important;
}

/* El dropdown abierto — todas las opciones */
[data-baseweb="popover"] [role="option"] {
    color: #111318 !important;
    background: #fff !important;
}

/* La opción sobre la que está el mouse (hover) */
[data-baseweb="popover"] [role="option"]:hover {
    background: #EEF2FF !important;
    color: #111318 !important;
}

/* La opción actualmente seleccionada dentro del dropdown abierto */
[data-baseweb="popover"] [aria-selected="true"] {
    background: #EEF2FF !important;
    color: #2451FF !important;
}

/* row separator */
.row-sep {
    height: 1px;
    background: #E8EBF0;
    margin: 0.5rem 0 1.25rem 0;
}

/* ── BUTTON ── */
div[data-testid="stButton"] > button {
    background: #111318 !important;
    color: #fff !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    padding: 0.9rem 2.5rem !important;
    transition: background 0.15s !important;
}
div[data-testid="stButton"] > button:hover { background: #2451FF !important; }

/* ── RESULT CARD ── */
.result-card {
    background: #fff;
    border: 1px solid #DFE2E8;
    border-top: 3px solid #2451FF;
    border-radius: 10px;
    padding: 2.25rem 2.5rem;
    margin: 2rem 0 1.25rem 0;
}
.result-card.bajo   { border-top-color: #1A9E6A; }
.result-card.moderado { border-top-color: #D98A00; }
.result-card.alto   { border-top-color: #CF2C2C; }
.r-eyebrow {
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.18em;
    text-transform: uppercase; color: #8A909E; margin-bottom: 0.6rem;
}
.r-number {
    font-family: 'JetBrains Mono', monospace;
    font-size: 4rem; font-weight: 600; color: #111318;
    letter-spacing: -0.04em; line-height: 1;
}
.r-label { font-size: 1rem; font-weight: 700; margin-top: 0.5rem; }
.bajo   .r-label { color: #1A9E6A; }
.moderado .r-label { color: #D98A00; }
.alto   .r-label { color: #CF2C2C; }
.r-desc { font-size: 0.85rem; color: #5A6070; line-height: 1.7; margin-top: 0.75rem; max-width: 520px; }
.risk-track {
    background: #EEF0F4; height: 5px; border-radius: 3px;
    margin: 1.5rem 0 0.4rem 0; overflow: hidden;
}
.risk-fill { height: 100%; border-radius: 3px; }
.risk-ticks {
    display: flex; justify-content: space-between;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem; color: #B0B6C3;
}

/* ── STAT GRID ── */
.stat-grid {
    display: grid; grid-template-columns: repeat(4, 1fr);
    gap: 0.75rem; margin-top: 1.25rem;
}
.stat-card {
    background: #F9FAFB; border: 1px solid #E8EBF0;
    border-radius: 8px; padding: 1rem 1.25rem;
}
.stat-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.5rem; font-weight: 600; color: #111318;
}
.stat-lbl {
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; color: #8A909E; margin-top: 0.25rem;
}
.stat-note { font-size: 0.72rem; color: #A5ABB8; margin-top: 0.2rem; }

/* ── NOTICE ── */
.notice {
    background: #F9FAFB; border: 1px solid #E8EBF0;
    border-left: 3px solid #8A909E; border-radius: 4px;
    padding: 1rem 1.25rem; font-size: 0.8rem; color: #5A6070;
    line-height: 1.65; margin-top: 1.25rem;
}
.notice strong { color: #111318; }
.notice-blue { border-left-color: #2451FF; }

/* ── PROSE ── */
.prose { font-size: 0.875rem; color: #444C5E; line-height: 1.8; }
.prose h4 {
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.18em;
    text-transform: uppercase; color: #111318; margin-top: 2rem;
    margin-bottom: 0.6rem; padding-bottom: 0.5rem; border-bottom: 1px solid #E8EBF0;
}
.prose ul { padding-left: 1.25rem; margin: 0.5rem 0; }
.prose li { margin-bottom: 0.4rem; }
.prose code {
    font-family: 'JetBrains Mono', monospace; font-size: 0.8rem;
    background: #F0F2F6; color: #2451FF; padding: 0.1rem 0.4rem; border-radius: 3px;
}
.badge {
    display: inline-block; font-size: 0.65rem; font-weight: 700;
    letter-spacing: 0.06em; padding: 0.2rem 0.55rem; border-radius: 3px;
    text-transform: uppercase; margin-left: 0.4rem; vertical-align: middle;
}
.badge-red { background: #FEF0F0; color: #CF2C2C; }
.badge-grn { background: #EDFAF4; color: #1A9E6A; }
.badge-blu { background: #EEF2FF; color: #2451FF; }

/* ── KPI strip ── */
.kpi-strip { display: grid; grid-template-columns: repeat(3,1fr); gap: 0.75rem; margin-bottom: 2rem; }
.kpi-card { background: #fff; border: 1px solid #DFE2E8; border-radius: 8px; padding: 1.25rem 1.5rem; }
.kpi-val { font-family: 'JetBrains Mono', monospace; font-size: 2rem; font-weight: 600; color: #2451FF; }
.kpi-lbl { font-size: 0.68rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: #8A909E; margin-top: 0.3rem; }

/* ── PLACEHOLDER ── */
.placeholder {
    background: #fff; border: 1.5px dashed #DFE2E8; border-radius: 8px;
    padding: 4rem 2rem; text-align: center; color: #A5ABB8;
    font-size: 0.78rem; font-weight: 600; letter-spacing: 0.1em;
    text-transform: uppercase; margin-top: 1.5rem;
}
.placeholder span {
    display: block; font-weight: 400; text-transform: none;
    letter-spacing: 0; font-size: 0.73rem; margin-top: 0.4rem; color: #C5C9D1;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# CACHE
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def cargar_modelo(ruta):
    try:
        return joblib.load(ruta)
    except FileNotFoundError:
        st.error(f"Modelo no encontrado: `{ruta}` — ejecuta `python main.py` primero.")
        st.stop()

@st.cache_data
def get_features():
    # Lista COMPLETA tal como fue entrenada — incluyendo consume_marihuana
    return PredictiveModel.FEATURES


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def sec(num, label):
    st.markdown(f"""
    <div class="sec-header">
        <span class="sec-num">{num}</span>
        <span class="sec-line"></span>
        <span class="sec-label">{label}</span>
    </div>""", unsafe_allow_html=True)


def field_row(title, desc, right_content_fn):
    """
    Fila de formulario: columna izquierda = meta, columna derecha = widget.
    Retorna el valor devuelto por right_content_fn.
    """
    left, right = st.columns([5, 7])
    with left:
        st.markdown(
            f'<div class="field-meta-title">{title}</div>'
            f'<div class="field-meta-desc">{desc}</div>',
            unsafe_allow_html=True
        )
    with right:
        val = right_content_fn()
    st.markdown('<div class="row-sep"></div>', unsafe_allow_html=True)
    return val


def nivel_riesgo(prob):
    if prob < 0.25:
        return ("bajo", "Riesgo Bajo", "#1A9E6A",
                "El perfil analizado presenta baja probabilidad de consumo duro. "
                "Los factores de riesgo identificados son limitados y no convergen.")
    if prob < 0.50:
        return ("moderado", "Riesgo Moderado", "#D98A00",
                "Se identifican factores de riesgo relevantes. El perfil sugiere exposición "
                "a entornos de consumo sin convergencia de múltiples alertas.")
    if prob < 0.75:
        return ("alto", "Riesgo Alto", "#CF2C2C",
                "El perfil concentra múltiples predictores documentados. "
                "Se recomienda intervención prioritaria y evaluación clínica especializada.")
    return ("alto", "Riesgo Muy Alto", "#CF2C2C",
            "Confluencia de los principales predictores. Intervención urgente — "
            "derivar a unidad de salud mental y programa de reducción de daños.")


# ──────────────────────────────────────────────────────────────────────────────
# PÁGINA 1 — PREDICCIÓN
# ──────────────────────────────────────────────────────────────────────────────
def pagina_prediccion():
    modelo   = cargar_modelo("outputs/mejor_modelo.joblib")
    features = get_features()   # lista completa con consume_marihuana
    datos    = {}

    # ── 01 · SOCIODEMOGRÁFICO ──────────────────────────────────────────────────
    sec("01", "Perfil Sociodemográfico")
    st.markdown('<div class="prose" style="margin-bottom:1.25rem;">Variables de base del perfil de riesgo. La edad, el nivel educativo y el género presentan correlaciones estadísticamente significativas con patrones de consumo en poblaciones vulnerables.</div>', unsafe_allow_html=True)

    datos['edad'] = field_row(
        "Edad",
        "Los consumidores de sustancias duras son en promedio 9 años más jóvenes que no consumidores. La edad de inicio en vida de calle también modula el riesgo basal del individuo.",
        lambda: st.slider("edad_sl", 0, 100, 35, label_visibility="collapsed")
    )

    genero_sel = field_row(
        "Género",
        "Permite controlar diferencias en patrones de consumo y acceso a redes de soporte. El dataset muestra mayor prevalencia masculina en consumo duro.",
        lambda: st.selectbox("gen_sel", ["Hombre", "Mujer"], label_visibility="collapsed")
    )
    datos['genero'] = 1 if genero_sel == "Hombre" else 2

    orient_opts = ["Heterosexual", "Homosexual (Gay/Lesbiana)", "Bisexual", "Otro"]
    orient_sel = field_row(
        "Orientación Sexual",
        "Grupos LGBTQ+ en calle enfrentan discriminación adicional que puede amplificar el consumo como mecanismo de afrontamiento. Variable de contexto social.",
        lambda: st.selectbox("orient_sel", orient_opts, label_visibility="collapsed")
    )
    datos['orientacion_sexual'] = orient_opts.index(orient_sel) + 1

    nivel_opts = ["Ninguno", "Preescolar", "Primaria incompleta", "Primaria completa",
                  "Secundaria incompleta", "Secundaria completa", "Técnico/Tecnológico", "Universitario"]
    nivel_sel = field_row(
        "Nivel Educativo",
        "Capital humano y oportunidades previas. Contraintuitivamente, perfiles técnicos muestran mayor riesgo relativo, posiblemente por redes urbanas de consumo asociadas.",
        lambda: st.selectbox("nivel_sel", nivel_opts, label_visibility="collapsed")
    )
    datos['nivel_educativo'] = nivel_opts.index(nivel_sel) + 1

    lee_opts = ["Sí, sin dificultad", "Sí, con alguna dificultad", "No puede"]
    lee_sel = field_row(
        "Alfabetismo",
        "Indicador proxy de acceso al sistema educativo y posibles dificultades cognitivas que incrementan la vulnerabilidad al consumo problemático.",
        lambda: st.selectbox("lee_sel", lee_opts, label_visibility="collapsed")
    )
    datos['sabe_leer_escribir'] = lee_opts.index(lee_sel) + 1

    # ── 02 · HISTORIA EN CALLE ─────────────────────────────────────────────────
    sec("02", "Historia en Calle")
    st.markdown('<div class="prose" style="margin-bottom:1.25rem;">El tiempo en situación de calle es uno de los predictores más robustos del modelo. Las razones de inicio y permanencia revelan el perfil motivacional y permiten segmentar intervenciones de política pública de manera diferenciada.</div>', unsafe_allow_html=True)

    años_calle = field_row(
        "Años en Calle",
        "Duración acumulada en situación de calle. Cada año adicional incrementa la exposición a redes de consumo y reduce el capital social disponible para la reinserción.",
        lambda: st.number_input("años_in", min_value=0, max_value=80, value=5, label_visibility="collapsed")
    )

    meses_calle = field_row(
        "Meses Adicionales",
        "Complemento al conteo anual para precisar la duración total y capturar casos recientes o en transición (rango: 0 a 11 meses).",
        lambda: st.number_input("meses_in", min_value=0, max_value=11, value=0, label_visibility="collapsed")
    )

    tiempo_total = int(años_calle) * 12 + int(meses_calle)
    datos['tiempo_total_calle_meses'] = tiempo_total
    st.markdown(f'<div class="notice notice-blue" style="margin-top:0; margin-bottom:0.5rem;">Tiempo total calculado: <strong style="font-family:\'JetBrains Mono\',monospace; color:#2451FF;">{tiempo_total} meses</strong> — variable continua utilizada directamente por el modelo.</div>', unsafe_allow_html=True)

    razon_inicio_opts = ["Consumo de sustancias", "Conflictos familiares", "Falta de oportunidades",
                         "Gusto por la calle", "Desplazamiento/violencia", "Salud mental",
                         "Abandono/falta de apoyo", "Otra razón"]
    ri_sel = field_row(
        "Razón de Inicio",
        "El motivo de ingreso diferencia trayectorias de riesgo: quienes llegan por consumo previo tienen mayor riesgo basal que quienes llegan por conflictos familiares o desplazamiento.",
        lambda: st.selectbox("ri_sel", razon_inicio_opts, index=1, label_visibility="collapsed")
    )
    datos['razon_inicio_vida_calle'] = razon_inicio_opts.index(ri_sel) + 1

    razon_cont_opts = ["Dependencia/consumo", "Falta de oportunidades", "Acostumbramiento",
                       "Sin apoyo familiar", "Problemas legales", "Otra razón"]
    rc_sel = field_row(
        "Razón de Permanencia",
        "Revela el nivel de arraigo y las barreras de salida. La dependencia a sustancias como razón de permanencia corresponde al escenario de mayor riesgo acumulado.",
        lambda: st.selectbox("rc_sel", razon_cont_opts, index=2, label_visibility="collapsed")
    )
    datos['razon_continua_en_calle'] = razon_cont_opts.index(rc_sel) + 1

    # ── 03 · SUBSISTENCIA ──────────────────────────────────────────────────────
    sec("03", "Subsistencia")
    st.markdown('<div class="prose" style="margin-bottom:1.25rem;">La forma de obtención de recursos determina la inserción en redes sociales de la calle. Ciertas actividades mantienen vínculos con la economía formal, mientras que otras exponen al individuo a entornos de alta disponibilidad de sustancias.</div>', unsafe_allow_html=True)

    dinero_opts = ["Reciclaje/materiales", "Venta ambulante", "Mendicidad",
                   "Trabajos informales", "Actividades ilegales", "Ayudas institucionales", "Otra"]
    dinero_sel = field_row(
        "Fuente de Ingresos",
        "Las actividades ilegales y la mendicidad correlacionan con mayor prevalencia de consumo duro. El reciclaje muestra el perfil de menor riesgo relativo en el dataset.",
        lambda: st.selectbox("dinero_sel", dinero_opts, index=2, label_visibility="collapsed")
    )
    datos['forma_obtener_dinero'] = dinero_opts.index(dinero_sel) + 1

    # ── 04 · DIAGNÓSTICOS DE SALUD ─────────────────────────────────────────────
    sec("04", "Diagnósticos de Salud")
    st.markdown('<div class="prose" style="margin-bottom:1.25rem;">Las condiciones crónicas actúan como mediadoras o moderadoras del consumo. La tuberculosis tiene alta prevalencia en consumidores de basuco por la vía pulmonar. El VIH/SIDA está asociado a redes de consumo compartido.</div>', unsafe_allow_html=True)

    dx_hta = field_row(
        "Hipertensión",
        "Asociada a cocaína y estimulantes. Su presencia sugiere exposición previa o actual a estas sustancias y puede ser indicador temprano de consumo.",
        lambda: st.checkbox("Diagnosticado", key="dx_hta_cb")
    )
    datos['dx_hipertension'] = 1 if dx_hta else 2

    dx_tbc = field_row(
        "Tuberculosis",
        "Elevada prevalencia en consumidores de basuco por inhalación. Indicador fuerte de consumo crónico y prolongado de sustancias duras.",
        lambda: st.checkbox("Diagnosticado", key="dx_tbc_cb")
    )
    datos['dx_tuberculosis'] = 1 if dx_tbc else 2

    dx_vih = field_row(
        "VIH / SIDA",
        "Asociado a consumo intravenoso y redes de consumo compartido. Duplica el riesgo estimado de consumo de heroína en el modelo.",
        lambda: st.checkbox("Diagnosticado", key="dx_vih_cb")
    )
    datos['dx_vih_sida'] = 1 if dx_vih else 2

    activ_opts = ["Sin dificultad", "Con dificultad", "No puede"]
    activ_sel = field_row(
        "Capacidad Funcional",
        "Las limitaciones físicas severas reducen opciones de subsistencia lícita y pueden empujar hacia redes de consumo como mecanismo social de pertenencia.",
        lambda: st.selectbox("activ_sel", activ_opts, label_visibility="collapsed")
    )
    datos['actividades_sin_esfuerzo_fisico'] = activ_opts.index(activ_sel) + 1

    # ── 05 · CONSUMO DE SUSTANCIAS ─────────────────────────────────────────────
    sec("05", "Consumo de Sustancias")
    st.markdown('<div class="prose" style="margin-bottom:0.75rem;">El consumo de sustancias de menor dependencia actúa como predictor de escalada. El modelo utiliza cigarrillo, alcohol e inhalantes como variables proxy del entorno y la trayectoria de consumo del individuo.</div>', unsafe_allow_html=True)
    

    cig = field_row(
        "Cigarrillo",
        "El 78% de consumidores de sustancias duras reporta consumo previo o simultáneo de tabaco. Puerta de entrada documentada en la literatura.",
        lambda: st.checkbox("Consume actualmente", key="cig_cb")
    )
    datos['consume_cigarrillo'] = 1 if cig else 2

    alc = field_row(
        "Alcohol",
        "Alta correlación con otras sustancias en entornos de calle. Reduce inhibiciones y facilita la experimentación con sustancias de mayor dependencia.",
        lambda: st.checkbox("Consume actualmente", key="alc_cb")
    )
    datos['consume_alcohol'] = 1 if alc else 2

    inh = field_row(
        "Inhalantes",
        "Predictor más fuerte del modelo: el 82% de sus consumidores escala a sustancias duras. La alta accesibilidad y bajo costo explican la progresión.",
        lambda: st.checkbox("Consume actualmente", key="inh_cb")
    )
    datos['consume_inhalantes'] = 1 if inh else 2

    # Marihuana: incluida en features pero NUNCA en el formulario
    # Se envía como NaN → el SimpleImputer del pipeline la completa con la mediana
    datos['consume_marihuana'] = np.nan

    # ── BOTÓN ──────────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    _, btn_col, _ = st.columns([1, 2, 1])
    with btn_col:
        predecir = st.button("Ejecutar Análisis de Riesgo →")

    if predecir:
        # Construir DataFrame con TODAS las features en el orden exacto del modelo
        fila = {f: np.nan for f in features}
        fila.update(datos)
        X = pd.DataFrame([fila])[features]

        with st.spinner("Procesando perfil..."):
            probabilidad = modelo.predict_proba(X)[0, 1]

        nivel, label, color, descripcion = nivel_riesgo(probabilidad)

        sec("—", "Resultado del Análisis")

        st.markdown(f"""
        <div class="result-card {nivel}">
            <div class="r-eyebrow">Probabilidad Estimada · Consumo de Sustancias Duras</div>
            <div class="r-number">{probabilidad:.1%}</div>
            <div class="r-label">{label}</div>
            <div class="r-desc">{descripcion}</div>
            <div class="risk-track" style="margin-top:1.75rem;">
                <div class="risk-fill" style="width:{probabilidad*100:.1f}%; background:{color};"></div>
            </div>
            <div class="risk-ticks">
                <span>0%</span><span>25%</span><span>50%</span><span>75%</span><span>100%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        sust = sum([datos.get('consume_cigarrillo', 2) == 1,
                    datos.get('consume_alcohol', 2) == 1,
                    datos.get('consume_inhalantes', 2) == 1])
        dx = sum([datos.get('dx_hipertension', 2) == 1,
                  datos.get('dx_tuberculosis', 2) == 1,
                  datos.get('dx_vih_sida', 2) == 1])

        st.markdown(f"""
        <div class="stat-grid">
            <div class="stat-card">
                <div class="stat-val">{datos['edad']}</div>
                <div class="stat-lbl">Edad</div>
                <div class="stat-note">Promedio consumidores: 37 a.</div>
            </div>
            <div class="stat-card">
                <div class="stat-val">{tiempo_total}</div>
                <div class="stat-lbl">Meses en Calle</div>
                <div class="stat-note">Mayor tiempo = mayor riesgo</div>
            </div>
            <div class="stat-card">
                <div class="stat-val">{sust}/3</div>
                <div class="stat-lbl">Sustancias Previas</div>
                <div class="stat-note">Cigarrillo · Alcohol · Inhalantes</div>
            </div>
            <div class="stat-card">
                <div class="stat-val">{dx}/3</div>
                <div class="stat-lbl">Diagnósticos</div>
                <div class="stat-note">HTA · TBC · VIH/SIDA</div>
            </div>
        </div>
        <div class="notice">
            <strong>Uso responsable:</strong> este output es el resultado de un modelo de
            clasificación binaria (AUC-ROC ≈ 0.90). No constituye diagnóstico clínico.
            Debe interpretarse por profesionales de salud mental en contexto.
        </div>
        """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# PÁGINA 2 — MODELO
# ──────────────────────────────────────────────────────────────────────────────
def pagina_modelo():
    st.markdown("""
    <div class="kpi-strip">
        <div class="kpi-card"><div class="kpi-val">~0.90</div><div class="kpi-lbl">AUC-ROC</div></div>
        <div class="kpi-card"><div class="kpi-val">~0.82</div><div class="kpi-lbl">F1-Score</div></div>
        <div class="kpi-card"><div class="kpi-val">5-fold</div><div class="kpi-lbl">Validación Cruzada</div></div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Variables", "Pipeline", "Hallazgos"])

    with tab1:
        st.markdown("""
        <div class="prose">
        <h4>Variable Target</h4>
        Consumo de sustancias de alta dependencia: basuco, cocaína o heroína.
        Variable binaria — 1: consume / 0: no consume.
        Balance en dataset: ~49% / 51%, sin necesidad de re-muestreo.

        <h4>Predictores Activos (18 variables)</h4>
        <ul>
            <li><strong>Sociodemográficas:</strong> edad, género, orientación sexual, nivel educativo, alfabetismo</li>
            <li><strong>Historia en calle:</strong> tiempo total (meses), razón de inicio, razón de permanencia</li>
            <li><strong>Subsistencia:</strong> forma de obtención de ingresos</li>
            <li><strong>Salud:</strong> hipertensión, tuberculosis, VIH/SIDA, capacidad funcional</li>
            <li><strong>Sustancias previas:</strong> cigarrillo, alcohol, inhalantes</li>
        </ul>

        <h4>Marihuana — Incluida en entrenamiento, excluida del formulario</h4>
        <code>consume_marihuana</code> forma parte del modelo entrenado (el pipeline la requiere),
        pero en producción se imputa con <code>NaN</code> por las siguientes razones técnicas:
        <ul>
            <li><strong>Colinealidad con el target:</strong> 67% de co-ocurrencia con consumo duro.
            El modelo aprendería a detectar co-consumo, no riesgo causal independiente.</li>
            <li><strong>Data leakage conceptual:</strong> en intervención temprana el individuo
            puede no tener aún historial de marihuana. Un predictor que requiere conocer el
            co-consumo actual es inútil para prevención efectiva.</li>
            <li><strong>Gestión técnica:</strong> el <code>SimpleImputer</code> del pipeline
            reemplaza el <code>NaN</code> con la mediana de entrenamiento, manteniendo
            la estructura del modelo sin alterar sus parámetros.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        st.markdown("""
        <div class="prose">
        <h4>Arquitectura del Pipeline (scikit-learn)</h4>
        <ol>
            <li><strong>SimpleImputer (mediana):</strong> maneja valores faltantes en variables
            categóricas codificadas como enteros. La mediana es robusta a outliers y preserva
            la escala ordinal. También gestiona el <code>NaN</code> de marihuana en producción.</li>
            <li><strong>StandardScaler:</strong> estandariza a media 0 y desviación estándar 1.
            Necesario para regresión logística; mejora convergencia en todos los estimadores.</li>
            <li><strong>Clasificador:</strong> mejor modelo seleccionado por AUC-ROC
            en validación cruzada estratificada de 5 folds.</li>
        </ol>

        <h4>Modelos Evaluados</h4>
        <ul>
            <li><strong>Regresión Logística:</strong> línea base interpretable. Coeficientes
            mapeables directamente a odds ratios para cada variable.</li>
            <li><strong>Random Forest:</strong> captura interacciones no lineales entre predictores.
            Robusto ante outliers y variables de baja relevancia.</li>
            <li><strong>Gradient Boosting:</strong> mayor potencia predictiva. Ganador en la
            mayoría de iteraciones de la validación cruzada.</li>
        </ul>

        <h4>Criterio de Selección del Modelo Final</h4>
        Selección por <strong>AUC-ROC</strong> y no por accuracy, dado que en este contexto
        los <em>falsos negativos</em> (no detectar consumo real) tienen mayor costo social
        que los falsos positivos. La curva ROC refleja la capacidad discriminante real
        independientemente del umbral de decisión.
        </div>
        """, unsafe_allow_html=True)

    with tab3:
        st.markdown("""
        <div class="prose">
        <h4>Importancia de Variables — Hallazgos Clave</h4>
        <ul>
            <li><strong>Inhalantes</strong>
            <span class="badge badge-red">82% co-ocurrencia</span>
            Predictor más fuerte. La ruta inhalante → basuco está ampliamente documentada
            en literatura latinoamericana de consumo en población de calle.</li>
            <li><strong>Tiempo en calle</strong>
            <span class="badge badge-red">Alto impacto</span>
            Efecto no lineal. El umbral crítico se identifica alrededor de los 36 meses
            acumulados, punto de inflexión en la curva de probabilidad predicha.</li>
            <li><strong>Edad</strong>
            <span class="badge badge-red">−9 años en promedio</span>
            Consumidores significativamente más jóvenes. La ventana de intervención
            más efectiva es temprana, antes de la consolidación del patrón de consumo.</li>
            <li><strong>Tuberculosis</strong>
            <span class="badge badge-red">Alta sensibilidad</span>
            Fuerte correlación con consumo de basuco por vía pulmonar. Puede usarse como
            señal de alerta en registros clínicos existentes para tamizaje de riesgo.</li>
            <li><strong>Reciclaje como subsistencia</strong>
            <span class="badge badge-grn">Factor protector relativo</span>
            Perfil de menor riesgo. Mantiene estructura de actividad económica y reduce
            tiempo de exposición en entornos de alta disponibilidad de sustancias.</li>
        </ul>

        <h4>Limitaciones del Modelo</h4>
        <ul>
            <li>Entrenado con datos de una región específica — generalización limitada
            a otros contextos sin proceso de reentrenamiento.</li>
            <li>Variables por auto-reporte: posible subestimación del consumo real
            por sesgo de deseabilidad social en contexto de encuesta.</li>
            <li>Modelo estático: no captura trayectorias temporales de entrada y salida
            del consumo, ni cambios en el perfil del individuo a lo largo del tiempo.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# PÁGINA 3 — VISUALIZACIONES
# ──────────────────────────────────────────────────────────────────────────────
def pagina_graficas():
    import plotly.express as px
    from model.interview_model import InterviewModel

    @st.cache_resource
    def cargar_datos_limpios():
        data_path = "data/CHC_base_anonimizada09-09-2021.xlsx"
        dict_path = "data/dictionary.json"
        
        # Cargar datos con InterviewModel
        interview = InterviewModel(data_path, dict_path)
        interview.clean_data()
        df = interview.data
        
        # Cargar el diccionario de respuestas
        with open(dict_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        
        # 1. Mapeo de DEPARTAMENTO
        mapeo_departamentos = {
            1: "Amazonas", 2: "Antioquia", 3: "Arauca", 4: "Atlántico",
            5: "Bolívar", 6: "Boyacá", 7: "Caldas", 8: "Caquetá",
            9: "Casanare", 10: "Cauca", 11: "Cesar", 12: "Chocó",
            13: "Córdoba", 14: "Cundinamarca", 15: "Guainía", 16: "Guaviare",
            17: "Huila", 18: "La Guajira", 19: "Magdalena", 20: "Meta",
            21: "Nariño", 22: "Norte de Santander", 23: "Putumayo", 24: "Quindío",
            25: "Risaralda", 26: "San Andrés y Providencia", 27: "Santander",
            28: "Sucre", 29: "Tolima", 30: "Valle del Cauca", 31: "Vaupés",
            32: "Vichada", 33: "Bogotá D.C."
        }
        if 'departamento' in df.columns:
            df['departamento'] = df['departamento'].map(mapeo_departamentos)
        
        # 2. Mapeo para TIPO DE DOCUMENTO (P10)
        if 'tipo_documento' in df.columns and 'P10' in config['respuestas']:
            mapeo = {int(k): v for k, v in config['respuestas']['P10'].items()}
            df['tipo_documento'] = df['tipo_documento'].map(mapeo)
        
        # 3. Mapeo para ACTIVIDADES DIARIAS (P16S9)
        if 'actividades_sin_esfuerzo_fisico' in df.columns and 'P16S9' in config['respuestas']:
            mapeo = {int(k): v for k, v in config['respuestas']['P16S9'].items()}
            df['actividades_sin_esfuerzo_fisico'] = df['actividades_sin_esfuerzo_fisico'].map(mapeo)
        
        # 4. Mapeo para RAZÓN DE SEGUIR EN CALLE (P24)
        if 'razon_continua_en_calle' in df.columns and 'P24' in config['respuestas']:
            mapeo = {int(k): v for k, v in config['respuestas']['P24'].items()}
            df['razon_continua_en_calle'] = df['razon_continua_en_calle'].map(mapeo)
        
        # 5. Mapeo para FORMA DE OBTENER DINERO (P29)
        if 'forma_obtener_dinero' in df.columns and 'P29' in config['respuestas']:
            mapeo = {int(k): v for k, v in config['respuestas']['P29'].items()}
            df['forma_obtener_dinero'] = df['forma_obtener_dinero'].map(mapeo)
        
        # 6. Mapeo para ORIENTACIÓN SEXUAL (P34)
        if 'orientacion_sexual' in df.columns and 'P34' in config['respuestas']:
            mapeo = {int(k): v for k, v in config['respuestas']['P34'].items()}
            df['orientacion_sexual'] = df['orientacion_sexual'].map(mapeo)
        
        # 7. Mapeo para NIVEL EDUCATIVO (P28)
        if 'nivel_educativo' in df.columns and 'P28' in config['respuestas']:
            mapeo = {int(k): v for k, v in config['respuestas']['P28'].items()}
            df['nivel_educativo'] = df['nivel_educativo'].map(mapeo)
        
        # 8. Crear columna de número de enfermedades
        dx_cols = ['dx_hipertension', 'dx_tuberculosis', 'dx_vih_sida']
        if all(col in df.columns for col in dx_cols):
            df['num_enfermedades'] = (df[dx_cols] == 1).sum(axis=1)
        else:
            df['num_enfermedades'] = 0
        
        return df
    
    df = cargar_datos_limpios()
    
    # Diccionario de columnas a graficar (nombre amigable -> nombre real en df)
    columnas_a_graficar = {
        'Departamento': 'departamento',
        'Tipo de documento': 'tipo_documento',
        'Actividades diarias sin esfuerzo físico': 'actividades_sin_esfuerzo_fisico',
        'Número de enfermedades': 'num_enfermedades',
        'Tiempo viviendo en la calle (meses)': 'tiempo_total_calle_meses',
        'Razón de seguir en la calle': 'razon_continua_en_calle',
        'Principal forma de obtener dinero': 'forma_obtener_dinero',
        'Orientación sexual': 'orientacion_sexual',
        'Nivel educativo': 'nivel_educativo'
    }
    
    # Verificar qué columnas existen realmente
    columnas_existentes = []
    for amigable, real in columnas_a_graficar.items():
        if real in df.columns:
            columnas_existentes.append((amigable, real))
        else:
            st.warning(f"La columna '{real}' no existe en los datos. Se omitirá.")
    
    if not columnas_existentes:
        st.error("No hay columnas válidas para graficar. Verifica los nombres.")
        return
    
    opciones = {amigable: real for amigable, real in columnas_existentes}
    
    # Pestañas
    tab1, tab3 = st.tabs(["📊 Distribuciones", "🔗 Relaciones"])
    
    with tab1:
        st.markdown("### Distribución de cada variable")
        var_amigable = st.selectbox("Elige una variable:", list(opciones.keys()))
        col_real = opciones[var_amigable]
        
        if pd.api.types.is_numeric_dtype(df[col_real]) and col_real not in ['num_enfermedades']:
            # Numérica continua (ej. tiempo en meses)
            fig = px.histogram(df, x=col_real, title=f"Distribución de {var_amigable}",
                               labels={col_real: var_amigable}, color_discrete_sequence=['#2451FF'])
        else:
            # Categórica o número de enfermedades (entero)
            conteos = df[col_real].value_counts().reset_index()
            conteos.columns = [col_real, 'Frecuencia']
            fig = px.bar(conteos, x=col_real, y='Frecuencia', title=f"Frecuencia de {var_amigable}",
                         color_discrete_sequence=['#2451FF'])
            fig.update_layout(xaxis={'categoryorder': 'total descending'})
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("### Relación entre dos variables")
        col1 = st.selectbox("Variable X:", list(opciones.keys()), key='x')
        col2 = st.selectbox("Variable Y:", list(opciones.keys()), key='y')
        col1r = opciones[col1]
        col2r = opciones[col2]
        
        if pd.api.types.is_numeric_dtype(df[col1r]) and pd.api.types.is_numeric_dtype(df[col2r]):
            fig = px.scatter(df, x=col1r, y=col2r, title=f"{col1} vs {col2}",
                             labels={col1r: col1, col2r: col2}, opacity=0.6)
            st.plotly_chart(fig, use_container_width=True)
        else:
            contingency = pd.crosstab(df[col1r], df[col2r])
            fig = px.imshow(contingency, text_auto=True, aspect="auto",
                            title=f"Frecuencias cruzadas: {col1} vs {col2}",
                            color_continuous_scale='Blues')
            st.plotly_chart(fig, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    st.markdown('<div class="shell">', unsafe_allow_html=True)

    st.markdown("""
    <div class="wordmark">
        <span class="wordmark-main">◈ CDT Predictor</span>
        <span class="wordmark-sep">·</span>
        <span class="wordmark-sub">Técnicas Cuantitativas II &nbsp;·&nbsp; Análisis de Riesgo en Población de Calle</span>
    </div>
    """, unsafe_allow_html=True)

    tab_pred, tab_mod, tab_graf = st.tabs([
        "Análisis Individual",
        "Documentación del Modelo",
        "Visualizaciones"
    ])

    with tab_pred:
        pagina_prediccion()
    with tab_mod:
        pagina_modelo()
    with tab_graf:
        pagina_graficas()

    st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
