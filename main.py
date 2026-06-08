# ==============================================================================
# SCRIPT PRINCIPAL: PIPELINE COMPLETO DE ANÁLISIS Y PREDICCIÓN
# ==============================================================================
# Propósito: Orquestar el flujo completo desde carga de datos hasta predicciones
#
# Flujo:
# 1. Cargar y limpiar datos de encuestas (InterviewModel)
# 2. Crear target binario y seleccionar predictores (PredictiveModel)
# 3. Entrenar 3 modelos con validación cruzada (PredictiveModel)
# 4. Evaluar y reportar resultados (PredictiveModel)
# 5. Exportar predicciones y modelos (PredictiveModel)
#
# Requisitos:
# - Archivo de datos: data/CHC_datos.xlsx (encuesta de personas en calle)
# - Diccionario: data/dictionary.json (mapeo de preguntas y respuestas)
# - Dependencias: ver requirements.txt
# ==============================================================================

from model.interview_model import InterviewModel, PredictiveModel
import sys

# ==============================================================================
# CONFIGURACIÓN
# ==============================================================================

# Rutas de datos
RUTA_DATOS = 'data/CHC_datos.xlsx'           # Archivo Excel con encuestas
RUTA_DICCIONARIO = 'data/dictionary.json'    # Diccionario de preguntas/respuestas
RUTA_SALIDA = 'outputs'                      # Directorio para resultados

# ==============================================================================
# FUNCIÓN PRINCIPAL
# ==============================================================================

def main():
    """
    Ejecuta el pipeline completo de análisis y predicción.
    
    Pasos:
    1. Fase de limpieza: InterviewModel
       - Carga datos Excel
       - Aplica renombres según diccionario
       - Valida datos y valores fuera de rango
       - Crea variables derivadas
    
    2. Fase de modelado: PredictiveModel
       - Crea target binario (consumo_duro)
       - Selecciona predictores
       - Divide datos train/test
       - Entrena 3 modelos
       - Reporta resultados
       - Exporta archivos
    """
    
    print("=" * 80)
    print("PREDICCIÓN DE CONSUMO DE SUSTANCIAS DURAS — PERSONAS EN SITUACIÓN DE CALLE")
    print("=" * 80)
    
    # ========================================================================
    # FASE 1: LIMPIEZA Y PREPARACIÓN DE DATOS
    # ========================================================================
    
    print("\n[FASE 1] CARGA Y LIMPIEZA DE DATOS")
    print("-" * 80)
    
    try:
        # Instanciar modelo de entrevista
        modelo_entrevista = InterviewModel(
            data_path=RUTA_DATOS,
            dictionary_path=RUTA_DICCIONARIO
        )
        
        # Limpiar datos
        modelo_entrevista.clean_data()
        
        datos_limpios = modelo_entrevista.data
        
    except FileNotFoundError as e:
        print(f"❌ ERROR: Archivo no encontrado: {e}")
        print(f"   Verifica que existan:")
        print(f"   - {RUTA_DATOS}")
        print(f"   - {RUTA_DICCIONARIO}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ ERROR en limpieza de datos: {e}")
        sys.exit(1)
    
    # ========================================================================
    # FASE 2: CONSTRUCCIÓN Y ENTRENAMIENTO DE MODELOS PREDICTIVOS
    # ========================================================================
    
    print("\n[FASE 2] MODELADO PREDICTIVO")
    print("-" * 80)
    
    try:
        # Instanciar modelo predictivo con datos limpios
        modelo_predictor = PredictiveModel(datos_limpios)
        
        # Crear variable target (consumo_duro)
        modelo_predictor.create_target()
        
        # Seleccionar variables predictoras
        modelo_predictor.select_predictors()
        
        # Construir pipelines (preprocesamiento + modelos)
        modelo_predictor.build_pipelines()
        
        # Entrenar modelos con validación cruzada
        modelo_predictor.train()
        
    except Exception as e:
        print(f"❌ ERROR en entrenamiento: {e}")
        sys.exit(1)
    
    # ========================================================================
    # FASE 3: EVALUACIÓN Y ANÁLISIS DEL MEJOR MODELO
    # ========================================================================
    
    print("\n[FASE 3] EVALUACIÓN DEL MEJOR MODELO")
    print("-" * 80)
    
    try:
        # Evaluar mejor modelo (reporte de clasificación + matriz de confusión)
        modelo_predictor.evaluate_best_model()
        
        # Mostrar importancia de variables
        modelo_predictor.feature_importance()
        
    except Exception as e:
        print(f"❌ ERROR en evaluación: {e}")
        sys.exit(1)
    
    # ========================================================================
    # FASE 4: EXPORTACIÓN DE RESULTADOS
    # ========================================================================
    
    print("\n[FASE 4] EXPORTACIÓN DE RESULTADOS")
    print("-" * 80)
    
    try:
        # Exportar predicciones, resumen de modelos y mejor modelo
        modelo_predictor.export_results(output_dir=RUTA_SALIDA)
        
    except Exception as e:
        print(f"❌ ERROR en exportación: {e}")
        sys.exit(1)
    
    # ========================================================================
    # RESUMEN FINAL
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("✅ PIPELINE COMPLETADO EXITOSAMENTE")
    print("=" * 80)
    print(f"\nResultados guardados en: {RUTA_SALIDA}/")
    print(f"Mejor modelo: {modelo_predictor.best_model_name}")
    print(f"Rendimiento (AUC-ROC): {modelo_predictor.results[modelo_predictor.best_model_name]['auc_test']:.3f}")
    print("\nArchivos generados:")
    print(f"  1. {RUTA_SALIDA}/predicciones.xlsx — datos + predicciones")
    print(f"  2. {RUTA_SALIDA}/resumen_modelos.xlsx — tabla de rendimiento")
    print(f"  3. {RUTA_SALIDA}/mejor_modelo.joblib — modelo entrenado")
    print("\n" + "=" * 80)


# ==============================================================================
# PUNTO DE ENTRADA
# ==============================================================================

if __name__ == '__main__':
    """
    Ejecuta el script principal si se llama directamente.
    
    Uso:
        python main.py
    
    Salida esperada:
        - Logs de cada fase del pipeline
        - Archivos Excel y modelo en carpeta 'outputs'
    """
    main()
