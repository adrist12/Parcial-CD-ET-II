# ==============================================================================
# MÓDULO DE MODELOS DE ENTREVISTA Y PREDICCIÓN
# ==============================================================================
# Propósito: Gestionar la limpieza, preparación e entrenamiento de modelos
#           para predicción de consumo de sustancias de alta dependencia.
#
# Componentes:
#   1. InterviewModel: carga y limpia datos de encuestas
#   2. PredictiveModel: entrena modelos predictivos y genera predicciones
#
# Flujo típico:
#   1. Instanciar InterviewModel con archivo de datos
#   2. Llamar clean_data() para preparar los datos
#   3. Instanciar PredictiveModel con datos limpios
#   4. Llamar train() para entrenar modelos
#   5. Llamar predict() para generar predicciones
# ==============================================================================

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
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, f1_score
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Suprimir advertencias de librerías (opcional)
warnings.filterwarnings('ignore')


class InterviewModel:
    """
    Gestor de limpieza y preparación de datos de entrevistas.
    
    Responsabilidades:
    - Cargar datos desde Excel
    - Aplicar renombres y selecciones de columnas desde diccionario
    - Validar datos y valores fuera del rango
    - Crear variables derivadas (ej: tiempo_total_calle_meses)
    - Mantener registro de registros excluidos
    
    Atributos:
        data (DataFrame): datos limpios y procesados
        excluded (DataFrame): registros sin entrevista válida (excluidos)
        rename (dict): mapeo de renombres de columnas
        select (list): columnas a seleccionar del Excel
        dictionary (dict): diccionario de respuestas válidas
        categorical_columns (list): columnas categóricas
        initial_age (list): columnas de edad de inicio de consumo
    """
    
    def __init__(self, data_path, dictionary_path="data/dictionary.json"):
        """
        Inicializa el modelo de entrevista.
        
        Args:
            data_path (str): ruta al archivo Excel con datos de encuestas
            dictionary_path (str): ruta al diccionario de configuración (default: data/dictionary.json)
        """
        # Cargar datos del Excel
        self.data = pd.read_excel(data_path)
        
        # Cargar configuración del diccionario JSON
        with open(dictionary_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        
        # Extraer componentes del diccionario
        self.rename = config["renombres"]           # mapeo: columna_original → columna_limpia
        self.select = config["preguntas"]           # lista de columnas a seleccionar
        self.dictionary = config["respuestas"]      # diccionario de códigos → descripciones
        self.categorical_columns = config["columnas_categoricas"]  # variables categóricas
        self.initial_age = config["columnas_edad_inicio"]  # variables de edad de inicio

    def clean_data(self):
        """
        Realiza la limpieza, validación y transformación de datos.
        
        Pasos:
        1. Selecciona columnas relevantes según diccionario
        2. Renombra columnas para legibilidad
        3. Excluye registros sin edad (sin entrevista válida)
        4. Convierte tipos de datos (Int64, float)
        5. Valida y corrige valores fuera de rango
        6. Crea variables derivadas (tiempo total en calle)
        7. Valida consistencia lógica
        
        Efectos secundarios:
        - Modifica self.data (datos limpios)
        - Crea self.excluded (registros excluidos)
        """
        
        # Paso 1-2: Seleccionar y renombrar columnas
        self.data = self.data[self.select]
        self.data = self.data.rename(columns=self.rename)
        
        # Paso 3: Identificar y excluir registros sin entrevista (edad NaN)
        sin_entrevista = self.data['edad'].isna()
        self.excluded = self.data[sin_entrevista].copy()
        self.data = self.data[~sin_entrevista].copy()
        print(f"[CLEAN] Registros excluidos (sin edad/entrevista): {len(self.excluded)}")
        
        # Paso 4: Conversión de tipos numéricos
        # Usar Int64 (nullable integer) para permitir NaN en datos categóricos
        self.data['edad'] = self.data['edad'].astype('Int64')
        self.data['años_en_calle'] = self.data['años_en_calle'].astype('Int64')
        self.data['meses_en_calle'] = self.data['meses_en_calle'].astype('Int64')
        
        # NOTA: NO imputar edad → dejar NaN como "no responde" para análisis posterior
        
        # Paso 5: Validación de valores fuera de diccionario
        # razon_inicio_vida_calle: códigos válidos 1-8 (+ 9 = no responde)
        razon_valida_p22 = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.data['razon_inicio_vida_calle'] = self.data['razon_inicio_vida_calle'].apply(
            lambda x: x if x in razon_valida_p22 else (8 if pd.notna(x) else np.nan)
        )
        
        # orientacion_sexual: códigos válidos 1-4 (heterosexual, homosexual, bisexual, otro)
        self.data['orientacion_sexual'] = self.data['orientacion_sexual'].apply(
            lambda x: x if x in [1, 2, 3, 4, 9] else (4 if pd.notna(x) else np.nan)
        )
        
        # Convertir todas las columnas categóricas a Int64 (permite NaN)
        for col in self.categorical_columns:
            self.data[col] = self.data[col].astype('Int64')
        
        # Convertir edades de inicio a Int64
        # NOTA: NaN aquí significa "no consume esa sustancia", no es dato faltante
        for col in self.initial_age:
            self.data[col] = self.data[col].astype('Int64')
        
        # Paso 6: Crear variable derivada: tiempo total en calle (en meses)
        # Fórmula: años*12 + meses (si ambos son conocidos)
        self.data['tiempo_total_calle_meses'] = (
            self.data['años_en_calle'].astype('float') * 12 +
            self.data['meses_en_calle'].astype('float')
        )
        
        # Paso 7: Validación de consistencia lógica
        # Advertencia si alguien reporta más años en calle que su edad actual
        inconsistencias_edad = self.data[
            self.data['años_en_calle'].notna() & 
            self.data['edad'].notna() &
            (self.data['años_en_calle'] > self.data['edad'])
        ]
        if len(inconsistencias_edad) > 0:
            print(f"⚠️  [CLEAN] {len(inconsistencias_edad)} registros con años_en_calle > edad (inconsistencia lógica)")
        
        print(f"[CLEAN] Datos limpios: {self.data.shape[0]} registros, {self.data.shape[1]} variables")
        return self


class PredictiveModel:
    """
    Modelo predictivo para consumo de sustancias de alta dependencia.
    
    TARGET BINARIO:
        consumo_duro = 1  →  consume basuco, heroína y/o cocaína
        consumo_duro = 0  →  no consume ninguna de las tres
    
    PREDICTORES:
        - Sociodemográficas: edad, género, nivel educativo, etc.
        - Historia en calle: tiempo total, razones de inicio/continuidad
        - Subsistencia: forma de obtener dinero
        - Salud: diagnósticos de condiciones crónicas
        - Escalada de consumo: cigarrillo, alcohol, marihuana, inhalantes
    
    MODELOS ENTRENADOS:
        1. Regresión Logística: interpretable, coeficientes directos
        2. Random Forest: captura no-linealidades, importancia de variables
        3. Gradient Boosting: máxima precisión predictiva
    
    Responsabilidades:
    - Crear target binario (consumo_duro) a partir de tres sustancias
    - Dividir datos en train/test con estratificación
    - Entrenar 3 modelos diferentes con validación cruzada
    - Generar reportes de rendimiento (AUC-ROC, F1, matriz de confusión)
    - Proporcionar importancia de variables
    - Generar predicciones en nuevos datos
    """
    
    def __init__(self, cleaned_data):
        """
        Inicializa el modelo predictivo con datos limpios.
        
        Args:
            cleaned_data (DataFrame): datos limpios de InterviewModel
        """
        self.data = cleaned_data.copy()
        self.models = {}                # diccionario de modelos entrenados
        self.results = {}              # resultados de evaluación
        self.best_model_name = None    # nombre del mejor modelo
        self.predictores = None         # lista de variables predictoras
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def create_target(self):
        """
        Construye la variable target binaria: consumo de sustancias duras.
        
        Lógica:
        - consumo_duro = 1 si responde Sí (1) a CUALQUIERA de:
          basuco, heroína, cocaína
        - consumo_duro = 0 si responde No (2) a TODAS las tres
        - consumo_duro = NaN si hay respuestas faltantes en las tres
        
        Después elimina los registros con target NaN (data leakage prevention).
        """
        sustancias_duras = ['consume_basuco', 'consume_heroina', 'consume_cocaina']
        
        # Crear target: 1 si consume cualquier sustancia dura, 0 si no consume ninguna, NaN si datos faltantes
        self.data['consumo_duro'] = self.data[sustancias_duras].apply(
            lambda row: 1 if any(row == 1) else (0 if all(row.notna()) else np.nan),
            axis=1
        )
        
        # Excluir registros con target no definido
        registros_excluidos = self.data['consumo_duro'].isna().sum()
        self.data = self.data.dropna(subset=['consumo_duro'])
        self.data['consumo_duro'] = self.data['consumo_duro'].astype(int)
        
        # Estadísticas del target
        positivos = self.data['consumo_duro'].sum()
        negativos = (self.data['consumo_duro'] == 0).sum()
        pct_positivos = positivos / len(self.data) * 100
        
        print(f"\n[TARGET] Variable creada: consumo_duro")
        print(f"  - Registros excluidos (datos faltantes): {registros_excluidos}")
        print(f"  - Consume duro (1): {positivos} ({pct_positivos:.1f}%)")
        print(f"  - NO consume duro (0): {negativos} ({100-pct_positivos:.1f}%)")
        print(f"  - Balance: {'EXCELENTE - casi perfecto' if 45 < pct_positivos < 55 else 'OK' if 30 < pct_positivos < 70 else 'DESBALANCEADO'}")
        
        return self
        
    def select_predictors(self):
        """
        Selecciona variables predictoras siguiendo criterios de validez conceptual.
        
        Criterios:
        1. NO incluir las sustancias duras (evitar definición circular del target)
        2. SÍ incluir otras sustancias (escalada de consumo válida, no data leakage)
        3. Incluir variables sociodemográficas, historia en calle, salud, subsistencia
        4. Variables deben ser conocibles ANTES de saber si consume duro
        
        La "escalada de consumo" (cigarrillo → alcohol → marihuana → duro) es
        bien documentada en la literatura de adicciones.
        """
        self.predictores = [
            # === SOCIODEMOGRÁFICAS ===
            'edad',                          # Factor de riesgo bien documentado
            'genero',                        # Distribuciones diferentes por género
            'nivel_educativo',               # Factor protector/riesgo contextual
            'sabe_leer_escribir',            # Indicador de educación temprana
            'orientacion_sexual',            # Variabilidad de riesgos reportada
            
            # === HISTORIA EN CALLE ===
            'tiempo_total_calle_meses',      # Mayor tiempo → mayor exposición
            'razon_inicio_vida_calle',       # Contexto inicial de vulnerabilidad
            'razon_continua_en_calle',       # Motivaciones actuales
            
            # === SUBSISTENCIA ===
            'forma_obtener_dinero',          # Acceso a dinero/actividades ilegales
            
            # === SALUD ===
            'dx_hipertension',               # Comorbilidad
            'dx_tuberculosis',               # Comorbilidad (frecuente en calle)
            'dx_vih_sida',                   # Comorbilidad
            'actividades_sin_esfuerzo_fisico',  # Indicador de salud general
            
            # === ESCALADA DE CONSUMO (bien documentada en literatura) ===
            # Estas NO son data leakage porque son SUSTANCIAS DIFERENTES
            'consume_cigarrillo',            # Primer nivel: sustancias legales
            'consume_alcohol',               # Primer nivel: sustancias legales
            'consume_marihuana',             # Nivel intermedio
            'consume_inhalantes',            # Fuerte predictor de duro (82% concordancia)
        ]
        
        X = self.data[self.predictores].copy()
        y = self.data['consumo_duro'].copy()
        
        nulos_totales = X.isnull().sum().sum()
        print(f"\n[PREDICTORES] {len(self.predictores)} variables seleccionadas")
        print(f"  - Datos faltantes totales: {nulos_totales}")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"  - Split train/test: {len(self.X_train)} / {len(self.X_test)}")
        return self
    
    def build_pipelines(self):
        """
        Construye pipelines de preprocesamiento + modelado.
        
        Cada pipeline encadena:
        1. SimpleImputer(strategy='median'): rellena NaN con mediana
           - Mediana (vs media) porque muchas variables son categóricas ordinales
           - Solo ajusta en train, aplica a test → evita data leakage
        
        2. StandardScaler: estandariza (media=0, std=1)
           - Crítico para Logistic Regression (sensible a escala)
           - Neutral para Random Forest y GBM (invariantes a escala)
        
        3. Modelo clasificador
           - Logistic Regression: interpretable, coeficientes directos
           - Random Forest: captura relaciones no-lineales
           - Gradient Boosting: máxima precisión con boosting iterativo
        """
        preprocesamiento = [
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]
        
        self.models = {
            'Regresión Logística': Pipeline(preprocesamiento + [
                ('model', LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1))
            ]),
            'Random Forest': Pipeline(preprocesamiento + [
                ('model', RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))
            ]),
            'Gradient Boosting': Pipeline(preprocesamiento + [
                ('model', GradientBoostingClassifier(n_estimators=200, random_state=42, subsample=0.8))
            ]),
        }
        
        print(f"\n[PIPELINES] {len(self.models)} modelos construidos")
        return self
    
    def train(self):
        """
        Entrena los 3 modelos con validación cruzada estratificada (5 folds).
        
        Validación cruzada:
        - El modelo se evalúa 5 veces en subconjuntos disjuntos
        - Cada fold: 4/5 entrenamiento, 1/5 validación
        - Reportamos: promedio ± desviación estándar (más robusto que 1 split)
        
        Métricas reportadas:
        - AUC-ROC: área bajo la curva ROC (0.5=azar, 1.0=perfecto)
        - F1-Score: media armónica de precisión y recall
        
        Resultado: self.results contiene rendimiento de cada modelo
        """
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        print(f"\n[TRAIN] Entrenamiento y validación cruzada (5 folds)...")
        print("-" * 70)
        
        self.results = {}
        
        for nombre, pipeline in self.models.items():
            print(f"\n  {nombre}:")
            
            # Validación cruzada: AUC-ROC y F1
            scores_auc = cross_val_score(
                pipeline, self.X_train, self.y_train, 
                cv=cv, scoring='roc_auc', n_jobs=-1
            )
            scores_f1 = cross_val_score(
                pipeline, self.X_train, self.y_train,
                cv=cv, scoring='f1', n_jobs=-1
            )
            
            # Entrenar en todo el train para evaluar en test
            pipeline.fit(self.X_train, self.y_train)
            y_pred = pipeline.predict(self.X_test)
            y_prob = pipeline.predict_proba(self.X_test)[:, 1]
            
            # Métricas en test
            auc_test = roc_auc_score(self.y_test, y_prob)
            f1_test = f1_score(self.y_test, y_pred)
            
            # Guardar resultados
            self.results[nombre] = {
                'pipeline': pipeline,
                'y_pred': y_pred,
                'y_prob': y_prob,
                'auc_cv': scores_auc.mean(),
                'auc_cv_std': scores_auc.std(),
                'f1_cv': scores_f1.mean(),
                'auc_test': auc_test,
                'f1_test': f1_test,
            }
            
            # Imprimir resumen
            print(f"    AUC-ROC (CV):  {scores_auc.mean():.3f} ± {scores_auc.std():.3f}")
            print(f"    F1-Score (CV): {scores_f1.mean():.3f} ± {scores_f1.std():.3f}")
            print(f"    AUC-ROC (Test): {auc_test:.3f}")
            print(f"    F1-Score (Test): {f1_test:.3f}")
        
        # Identificar mejor modelo (por AUC en test)
        self.best_model_name = max(self.results, key=lambda k: self.results[k]['auc_test'])
        mejor_auc = self.results[self.best_model_name]['auc_test']
        print(f"\n  ✅ MEJOR MODELO: {self.best_model_name} (AUC-ROC = {mejor_auc:.3f})")
        
        return self
    
    def evaluate_best_model(self):
        """
        Genera reportes detallados del mejor modelo.
        
        Incluye:
        - Reporte de clasificación (precisión, recall, F1 por clase)
        - Matriz de confusión (verdaderos positivos/negativos, falsos positivos/negativos)
        """
        if self.best_model_name is None:
            print("⚠️  No se ha entrenado modelo aún. Llamar train() primero.")
            return self
        
        mejor = self.results[self.best_model_name]
        y_pred = mejor['y_pred']
        
        print(f"\n[EVALUACIÓN] {self.best_model_name}")
        print("-" * 70)
        
        # Reporte de clasificación
        print("\nReporte de clasificación:")
        print(classification_report(
            self.y_test, y_pred,
            target_names=['No consume duro', 'Consume duro']
        ))
        
        # Matriz de confusión
        cm = confusion_matrix(self.y_test, y_pred)
        tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
        
        print("Matriz de confusión:")
        print(f"  Verdaderos negativos (TP):   {tn}  (predijo No → era No)  ← Aciertos en negativos")
        print(f"  Falsos positivos (FP):       {fp}  (predijo Sí → era No)  ← Error: alarma falsa")
        print(f"  Falsos negativos (FN):       {fn}  (predijo No → era Sí)  ← Error: riesgo perdido")
        print(f"  Verdaderos positivos (TP):   {tp}  (predijo Sí → era Sí)  ← Aciertos en positivos")
        
        return self
    
    def feature_importance(self):
        """
        Muestra la importancia de variables según el mejor modelo.
        
        Interpretación:
        - Regresión Logística: coeficientes estandarizados (cambio en log-odds)
                               y odds_ratio (multiplicador de probabilidad)
        - Random Forest/GBM:   importancia basada en reducción de impureza (Gini)
        
        Las variables más importantes son las que el modelo usa más para
        separar consumidores de no consumidores.
        """
        if self.best_model_name is None:
            print("⚠️  No se ha entrenado modelo aún. Llamar train() primero.")
            return self
        
        mejor = self.results[self.best_model_name]
        print(f"\n[IMPORTANCIA] Variables — {self.best_model_name}")
        print("-" * 70)
        
        if self.best_model_name == 'Regresión Logística':
            # Coeficientes estandarizados (después de scaler)
            coefs = mejor['pipeline'].named_steps['model'].coef_[0]
            df_imp = pd.DataFrame({
                'Variable': self.predictores,
                'Coeficiente': coefs,
                'Odds Ratio': np.exp(coefs)
            }).sort_values('Coeficiente', ascending=False)
            
            print("\nInterpretación:")
            print("  - Coef > 0: aumenta probabilidad de consumo duro")
            print("  - Coef < 0: disminuye probabilidad de consumo duro")
            print("  - Odds Ratio: multiplicador de probabilidad ante aumento unitario\n")
            print(df_imp.to_string(index=False))
            
        else:  # Random Forest o Gradient Boosting
            importancias = mejor['pipeline'].named_steps['model'].feature_importances_
            df_imp = pd.DataFrame({
                'Variable': self.predictores,
                'Importancia': importancias
            }).sort_values('Importancia', ascending=False)
            
            print("\nInterpretación:")
            print("  - Importancia: proporción de reducción de impureza (Gini)")
            print("  - Rango: [0, 1], suma = 1")
            print("  - Variables con mayor importancia = más usadas para separar clases\n")
            print(df_imp.to_string(index=False))
        
        # Siempre mostrar Random Forest como referencia si no es el mejor
        if self.best_model_name != 'Random Forest':
            print(f"\n--- Random Forest (referencia) ---")
            rf = self.results['Random Forest']['pipeline'].named_steps['model']
            df_rf = pd.DataFrame({
                'Variable': self.predictores,
                'Importancia': rf.feature_importances_
            }).sort_values('Importancia', ascending=False)
            print(df_rf.to_string(index=False))
        
        return self
    
    def predict(self, X_new):
        """
        Genera predicciones en nuevos datos usando el mejor modelo.
        
        Args:
            X_new (DataFrame): nuevos datos con mismas variables que predictores
        
        Returns:
            dict: diccionario con predicciones (clase) y probabilidades
        """
        if self.best_model_name is None:
            raise ValueError("No hay modelo entrenado. Llamar train() primero.")
        
        mejor = self.results[self.best_model_name]['pipeline']
        predicciones = mejor.predict(X_new)
        probabilidades = mejor.predict_proba(X_new)[:, 1]
        
        return {
            'predicciones': predicciones,
            'probabilidades': probabilidades,
            'modelo': self.best_model_name
        }
    
    def export_results(self, output_dir='outputs'):
        """
        Exporta resultados a archivos Excel.
        
        Archivos generados:
        1. {output_dir}/predicciones.xlsx: datos + predicciones + probabilidades
        2. {output_dir}/resumen_modelos.xlsx: tabla resumen de rendimiento de modelos
        3. {output_dir}/mejor_modelo.joblib: modelo entrenado (para reutilización)
        
        Args:
            output_dir (str): directorio de salida (default: 'outputs')
        """
        # Crear directorio si no existe
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 1. Exportar predicciones
        df_resultado = self.data[self.predictores + ['consumo_duro']].copy()
        mejor = self.results[self.best_model_name]
        df_resultado['prob_consumo_duro'] = mejor['pipeline'].predict_proba(self.X_train.append(self.X_test))[:, 1]
        df_resultado['pred_consumo_duro'] = mejor['pipeline'].predict(self.X_train.append(self.X_test))
        
        archivo_pred = Path(output_dir) / 'predicciones.xlsx'
        df_resultado.to_excel(archivo_pred, index=False)
        print(f"\n[EXPORT] {archivo_pred}")
        
        # 2. Exportar resumen de modelos
        resumen = pd.DataFrame([{
            'Modelo': nombre,
            'AUC-ROC (CV)': f"{r['auc_cv']:.3f} ± {r['auc_cv_std']:.3f}",
            'F1 (CV)': f"{r['f1_cv']:.3f}",
            'AUC-ROC (Test)': f"{r['auc_test']:.3f}",
            'F1 (Test)': f"{r['f1_test']:.3f}",
        } for nombre, r in self.results.items()])
        
        archivo_resumen = Path(output_dir) / 'resumen_modelos.xlsx'
        resumen.to_excel(archivo_resumen, index=False)
        print(f"[EXPORT] {archivo_resumen}")
        
        # 3. Guardar mejor modelo
        archivo_modelo = Path(output_dir) / 'mejor_modelo.joblib'
        joblib.dump(self.results[self.best_model_name]['pipeline'], archivo_modelo)
        print(f"[EXPORT] {archivo_modelo}")
        
        print(f"\n✅ Resultados exportados a {output_dir}/")
        return self