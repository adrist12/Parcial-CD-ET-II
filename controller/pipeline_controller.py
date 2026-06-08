"""
CONTROLADOR DEL PIPELINE
=========================
Orquesta el flujo completo: limpieza → entrenamiento → evaluación → guardado.
No contiene lógica de negocio, solo decide qué se llama y en qué orden.
"""

import sys
from model.interview_model import InterviewModel, PredictiveModel


class PipelineController:

    def __init__(self, data_path: str, dictionary_path: str, model_path: str):
        # Guardamos las rutas como atributos para que cada método
        # del controlador pueda acceder a ellas sin recibirlas como parámetro.
        self.data_path       = data_path
        self.dictionary_path = dictionary_path
        self.model_path      = model_path

        # Estos se llenan conforme avanza el pipeline.
        # Inicializarlos en None permite detectar si alguien intenta
        # usar el modelo antes de entrenarlo.
        self.modelo = None

    def run_cleaning(self):
        print("\n[FASE 1] LIMPIEZA DE DATOS")
        print("-" * 60)
        entrevista = InterviewModel(self.data_path, self.dictionary_path)
        entrevista.clean_data()

        # Devolvemos los datos limpios para pasarlos a la siguiente fase.
        # El controlador no los almacena porque no es su responsabilidad
        # manejar datos — eso es del modelo.
        return entrevista.data

    def run_training(self, datos_limpios):
        print("\n[FASE 2] ENTRENAMIENTO")
        print("-" * 60)
        self.modelo = PredictiveModel(datos_limpios)
        (self.modelo
            .create_target()
            .select_predictors()
            .build_pipelines()
            .train())

    def run_evaluation(self):
        print("\n[FASE 3] EVALUACIÓN")
        print("-" * 60)
        self.modelo.evaluate()
        self.modelo.feature_importance()

    def run_save(self):
        print("\n[FASE 4] GUARDANDO MODELO")
        print("-" * 60)
        self.modelo.save(self.model_path)

    def run(self):
        """
        Ejecuta el pipeline completo de principio a fin.
        Cada fase está en un try/except propio para identificar
        exactamente dónde falló sin ocultar el error real.
        """
        try:
            datos = self.run_cleaning()
        except Exception as e:
            print(f"❌ Error en limpieza: {e}")
            sys.exit(1)

        try:
            self.run_training(datos)
        except Exception as e:
            print(f"❌ Error en entrenamiento: {e}")
            sys.exit(1)

        try:
            self.run_evaluation()
        except Exception as e:
            print(f"❌ Error en evaluación: {e}")
            sys.exit(1)

        try:
            self.run_save()
        except Exception as e:
            print(f"❌ Error guardando modelo: {e}")
            sys.exit(1)

        print("\n" + "=" * 60)
        print(f"✅ Pipeline completado — {self.modelo.best_model_name}")
        print(f"   AUC-ROC: {self.modelo.results[self.modelo.best_model_name]['auc_test']:.3f}")
        print("=" * 60)