"""
MÓDULO DE MODELOS: LIMPIEZA Y PREDICCIÓN
=========================================
Componentes:
  InterviewModel  — carga y limpia los datos del Excel
  PredictiveModel — entrena modelos y genera predicciones
"""

import pandas as pd
import numpy as np
import json
import warnings
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

warnings.filterwarnings('ignore')


# ==============================================================================
# INTERVIEW MODEL
# ==============================================================================

class InterviewModel:
    """
    Responsabilidad única: cargar y limpiar los datos del Excel.

    No sabe nada de modelos ni predicciones. Solo prepara los datos
    para que PredictiveModel los reciba listos.
    """

    def __init__(self, data_path: str, dictionary_path: str):
        # Cargamos el Excel tal como está, sin modificar nada todavía.
        self.data = pd.read_excel(data_path)

        # Leemos el diccionario JSON que centraliza toda la configuración:
        # qué columnas seleccionar, cómo renombrarlas, cuáles son categóricas, etc.
        # Así si el Excel cambia de nombre en una columna, solo tocamos el JSON.
        with open(dictionary_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        self.select              = config["preguntas"]           # columnas a extraer del Excel
        self.rename              = config["renombres"]           # mapeo original → legible
        self.categorical_columns = config["columnas_categoricas"]
        self.initial_age         = config["columnas_edad_inicio"]

        # excluded se llenará en clean_data()
        self.excluded = None

    def clean_data(self):
        """
        Limpia y transforma los datos en 6 pasos ordenados.
        Modifica self.data y crea self.excluded.
        """

        # ------------------------------------------------------------------
        # PASO 1 — Seleccionar y renombrar columnas
        # ------------------------------------------------------------------
        # Reducimos de ~130 columnas a las 36 relevantes, y las renombramos
        # a nombres legibles (ej: P8R → edad).
        # .copy() evita el SettingWithCopyWarning de pandas al modificar
        # subconjuntos más adelante.

        self.data = self.data[self.select].copy()
        self.data = self.data.rename(columns=self.rename)

        # ------------------------------------------------------------------
        # PASO 2 — Excluir registros sin entrevista
        # ------------------------------------------------------------------
        # Si 'edad' está vacía, significa que la entrevista no se realizó.
        # Los separamos en self.excluded en lugar de eliminarlos silenciosamente,
        # para tener trazabilidad de cuántos y cuáles son.

        sin_entrevista  = self.data['edad'].isna()
        self.excluded   = self.data[sin_entrevista].copy()
        self.data       = self.data[~sin_entrevista].copy()

        print(f"  Registros sin entrevista (excluidos): {len(self.excluded)}")
        print(f"  Registros válidos:                    {len(self.data)}")

        # ------------------------------------------------------------------
        # PASO 3 — Convertir tipos numéricos
        # ------------------------------------------------------------------
        # Las columnas de tiempo llegaron como float64 porque pandas convierte
        # automáticamente cuando hay NaN. Las pasamos a Int64 (entero nullable)
        # para que no aparezcan como "3.0" sino "3" en reportes.

        for col in ['edad', 'años_en_calle', 'meses_en_calle']:
            self.data[col] = self.data[col].astype('Int64')

        # ------------------------------------------------------------------
        # PASO 4 — Corregir valores fuera del diccionario
        # ------------------------------------------------------------------
        # Dos variables tienen códigos que no existen en el diccionario original
        # porque el formulario fue actualizado. Los reasignamos a "Otra razón"
        # en lugar de convertirlos en NaN (perderíamos ~550 registros reales).

        razon_valida_p22 = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.data['razon_inicio_vida_calle'] = self.data['razon_inicio_vida_calle'].apply(
            lambda x: x if x in razon_valida_p22 else (8 if pd.notna(x) else np.nan)
        )

        self.data['orientacion_sexual'] = self.data['orientacion_sexual'].apply(
            lambda x: x if x in [1, 2, 3, 4, 9] else (4 if pd.notna(x) else np.nan)
        )

        # ------------------------------------------------------------------
        # PASO 5 — Convertir columnas categóricas y edades de inicio
        # ------------------------------------------------------------------
        # Int64 (nullable) en lugar de float64 para que los códigos se lean
        # como "1" y no como "1.0".
        # Las edades de inicio pueden ser NaN con pleno sentido: significa
        # "no consume esa sustancia", NO es un dato faltante.

        for col in self.categorical_columns:
            self.data[col] = self.data[col].astype('Int64')

        for col in self.initial_age:
            self.data[col] = self.data[col].astype('Int64')

        # ------------------------------------------------------------------
        # PASO 6 — Crear variable derivada y validar consistencia
        # ------------------------------------------------------------------
        # Combinamos años y meses en una sola variable continua en meses,
        # más útil para correlaciones y modelos que dos columnas separadas.

        self.data['tiempo_total_calle_meses'] = (
            self.data['años_en_calle'].astype('float') * 12 +
            self.data['meses_en_calle'].astype('float')
        )

        # Alerta de inconsistencia lógica (años en calle > edad actual)
        inconsistencias = self.data[
            self.data['años_en_calle'].notna() &
            self.data['edad'].notna() &
            (self.data['años_en_calle'] > self.data['edad'])
        ]
        if len(inconsistencias) > 0:
            print(f"  ⚠ {len(inconsistencias)} registros con años_en_calle > edad")

        print(f"  Variables finales: {self.data.shape[1]}")
        return self


# ==============================================================================
# PREDICTIVE MODEL
# ==============================================================================
class PredictiveModel:
    """
    Responsabilidad única: entrenar modelos predictivos y generar predicciones.
    Recibe los datos ya limpios de InterviewModel.
    
    TARGET: consumo_duro = 1 si consume basuco, heroína o cocaína.
    """

    # 1. Expandimos la lista para incluir las 17 variables de tu formulario
    FEATURES = [
        # ── 01 · SOCIODEMOGRÁFICO ──
        'edad', 
        'genero', 
        'orientacion_sexual', 
        'nivel_educativo', 
        'sabe_leer_escribir',
        
        # ── 02 · HISTORIA EN CALLE ──
        'tiempo_total_calle_meses', 
        'razon_inicio_vida_calle', 
        'razon_continua_en_calle',
        
        # ── 03 · SUBSISTENCIA ──
        'forma_obtener_dinero',
        
        # ── 04 · DIAGNÓSTICOS DE SALUD ──
        'dx_hipertension', 
        'dx_tuberculosis', 
        'dx_vih_sida', 
        'actividades_sin_esfuerzo_fisico',
        
        # ── 05 · CONSUMO DE SUSTANCIAS ──
        'consume_cigarrillo', 
        'consume_alcohol', 
        'consume_inhalantes', 
        'consume_marihuana'  # Se mantiene (el Imputer manejará su NaN en predicción)
    ]

    def __init__(self, cleaned_data: pd.DataFrame):
        self.data           = cleaned_data.copy()
        self.results        = {}       # rendimiento de cada modelo
        self.best_model_name = None    # nombre del ganador
        self.X_train = self.X_test = self.y_train = self.y_test = None

    def create_target(self):
        """Construye consumo_duro como variable binaria."""
        sustancias = ['consume_basuco', 'consume_heroina', 'consume_cocaina']

        self.data['consumo_duro'] = self.data[sustancias].apply(
            lambda row: 1 if (row == 1).any() else (0 if row.notna().all() else np.nan),
            axis=1
        )

        excluidos = self.data['consumo_duro'].isna().sum()
        self.data = self.data.dropna(subset=['consumo_duro'])
        self.data['consumo_duro'] = self.data['consumo_duro'].astype(int)

        positivos = self.data['consumo_duro'].sum()
        total     = len(self.data)
        print(f"  Target creado — positivos: {positivos} ({positivos/total:.1%}) "
              f"| negativos: {total - positivos} ({1 - positivos/total:.1%})")
        print(f"  Registros excluidos (target indefinido): {excluidos}")
        return self

    def select_predictors(self):
        """Separa X e y, y hace el split train/test."""
        X = self.data[self.FEATURES].copy()
        y = self.data['consumo_duro'].copy()

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"  Train: {len(self.X_train)} | Test: {len(self.X_test)}")
        return self

    def build_pipelines(self):
        """Construye los tres pipelines: preprocesamiento + modelo."""
        pre = [
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler',  StandardScaler()),
        ]

        self._pipelines = {
            'Regresión Logística': Pipeline(pre + [
                ('model', LogisticRegression(max_iter=1000, random_state=42))
            ]),
            'Random Forest': Pipeline(pre + [
                ('model', RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))
            ]),
            'Gradient Boosting': Pipeline(pre + [
                ('model', GradientBoostingClassifier(n_estimators=200, random_state=42))
            ]),
        }
        print(f"  {len(self._pipelines)} pipelines construidos")
        return self

    def train(self):
        """Entrena cada pipeline con validación cruzada de 5 folds."""
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for nombre, pipeline in self._pipelines.items():
            scores_auc = cross_val_score(
                pipeline, self.X_train, self.y_train,
                cv=cv, scoring='roc_auc', n_jobs=-1
            )
            scores_f1 = cross_val_score(
                pipeline, self.X_train, self.y_train,
                cv=cv, scoring='f1', n_jobs=-1
            )

            pipeline.fit(self.X_train, self.y_train)
            y_pred = pipeline.predict(self.X_test)
            y_prob = pipeline.predict_proba(self.X_test)[:, 1]

            self.results[nombre] = {
                'pipeline':    pipeline,
                'y_pred':      y_pred,
                'y_prob':      y_prob,
                'auc_cv':      scores_auc.mean(),
                'auc_cv_std':  scores_auc.std(),
                'f1_cv':       scores_f1.mean(),
                'auc_test':    roc_auc_score(self.y_test, y_prob),
                'f1_test':     f1_score(self.y_test, y_pred),
            }

            print(f"\n  {nombre}:")
            print(f"    AUC-ROC CV:   {scores_auc.mean():.3f} ± {scores_auc.std():.3f}")
            print(f"    AUC-ROC Test: {roc_auc_score(self.y_test, y_prob):.3f}")
            print(f"    F1 Test:      {f1_score(self.y_test, y_pred):.3f}")

        self.best_model_name = max(self.results, key=lambda k: self.results[k]['auc_test'])
        print(f"\n  ✅ Mejor modelo: {self.best_model_name} (AUC = {self.results[self.best_model_name]['auc_test']:.3f})")
        return self

    def evaluate(self):
        mejor  = self.results[self.best_model_name]
        y_pred = mejor['y_pred']

        print(f"\n  {self.best_model_name}")
        print(classification_report(
            self.y_test, y_pred,
            target_names=['No consume duro', 'Consume duro']
        ))
        return self

    def feature_importance(self):
        mejor = self.results[self.best_model_name]
        model = mejor['pipeline'].named_steps['model']

        if self.best_model_name == 'Regresión Logística':
            coefs = model.coef_[0]
            df_imp = pd.DataFrame({
                'Variable':    self.FEATURES,
                'Coeficiente': coefs,
                'Odds Ratio':  np.exp(coefs),
            }).sort_values('Coeficiente', ascending=False)
        else:
            df_imp = pd.DataFrame({
                'Variable':    self.FEATURES,
                'Importancia': model.feature_importances_,
            }).sort_values('Importancia', ascending=False)

        print(df_imp.to_string(index=False))
        return self

    def save(self, path: str = 'outputs/mejor_modelo.joblib'):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.results[self.best_model_name]['pipeline'], path)
        print(f"  Modelo guardado en '{path}'")
        return self