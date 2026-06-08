"""
PUNTO DE ENTRADA
=================
Responsabilidad única: decidir si entrenar el modelo o lanzar la app.
No contiene lógica de negocio ni de orquestación — eso es del controlador.

Uso:
    python main.py          → entrena el modelo
    python main.py --app    → lanza la app de Streamlit
"""

import sys
import subprocess
from controller.pipeline_controller import PipelineController

RUTA_DATOS       = 'data/CHC_base_anonimizada09-09-2021.xlsx'
RUTA_DICCIONARIO = 'data/dictionary.json'
RUTA_MODELO      = 'outputs/mejor_modelo.joblib'


def main():
    # Si el usuario pasa --app como argumento, lanzamos Streamlit.
    # subprocess.run() ejecuta el comando como si lo escribieras en la terminal.
    # El proceso de Streamlit reemplaza este proceso — cuando cierras la app,
    # main.py termina también.
    if '--app' in sys.argv:
        print("Lanzando dashboard...")
        subprocess.run([sys.executable, '-m', 'streamlit', 'run', 'view/dashboard.py'])
        return

    # Sin argumentos: correr el pipeline de entrenamiento completo.
    print("=" * 60)
    print("PIPELINE CHC — CONSUMO DE SUSTANCIAS DURAS")
    print("=" * 60)

    controller = PipelineController(RUTA_DATOS, RUTA_DICCIONARIO, RUTA_MODELO)
    controller.run()


if __name__ == '__main__':
    main()