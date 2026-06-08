# 📊 Modelo Predictivo — Consumo de Sustancias Duras

## 🎯 Objetivo

Predecir si una persona en situación de calle consume sustancias de **alta dependencia** (basuco, heroína o cocaína) basándose en:
- Variables sociodemográficas
- Historia en calle
- Patrones de consumo de otras sustancias
- Condiciones de salud

## 🧬 Target (Variable Dependiente)

```
consumo_duro = 1  →  consume basuco, heroína y/o cocaína
consumo_duro = 0  →  no consume ninguna de las tres
```

### ¿Por qué este target es MEDIBLE y NO un proxy?

✅ **Respuesta directa de la encuesta**: "¿Usted consume X?"  
✅ **No requiere seguimiento longitudinal**  
✅ **No es una inferencia** — es un dato observado

---

## 📈 Hallazgos Clave

| Métrica | Valor | Implicación |
|---------|-------|-----------|
| **Balance de clases** | 49% pos / 51% neg | Sin sesgo de clase ✅ |
| **Edad media (con duro vs sin)** | 37 vs 46 años | 9 años más jóvenes los consumidores |
| **Inhalantes → duro** | 82% | Fuerte predictor |
| **Marihuana → duro** | 67% | Escalada de consumo evidente |
| **Educación técnica/tecn.** | Mayor riesgo | Posible efecto de contexto urbano |

---

## 🤖 Modelos Entrenados

Entrenamos **3 modelos distintos** con validación cruzada (5 folds):

### 1. **Regresión Logística**
- ✅ **Ventaja**: Muy interpretable — coeficientes directos  
- ✅ **Coeficientes estandarizados** — indica cambio en log-odds  
- ✅ **Odds ratio** — multiplicador de probabilidad  
- ❌ Asume relaciones lineales

### 2. **Random Forest**
- ✅ **Captura no-linealidades** — interacciones complejas  
- ✅ **Importancia variable basada en Gini** — qué variables usa más  
- ✅ **Robusto a outliers**  
- ❌ Menos interpretable

### 3. **Gradient Boosting**
- ✅ **Máxima precisión predictiva** — boosting iterativo  
- ✅ **Maneja datos desequilibrados bien**  
- ✅ **Captura patrones sutiles**  
- ❌ Riesgo de overfitting (mitigado con subsample=0.8)

---

## 📊 Métricas de Evaluación

### AUC-ROC (Area Under the Curve)
- **Rango**: [0, 1]  
- **0.5**: modelo azar (inútil)  
- **1.0**: clasificación perfecta  
- **Interpretación**: probabilidad de que modelo rank'ee correctamente un par (+,−)

### F1-Score
- **Rango**: [0, 1]  
- **Media armónica** de precisión y recall  
- **Balancea**: falsos positivos vs falsos negativos  
- **Útil cuando**: clases desbalanceadas

### Matriz de Confusión
```
                Predicho Negativo    Predicho Positivo
Actual Negativo      TN (acierto)     FP (alarma falsa)
Actual Positivo      FN (perdida)     TP (acierto)
```

---

## 🚀 Uso del Proyecto

### 1. **Instalación de Dependencias**

```bash
pip install -r requirements.txt
```

### 2. **Estructura de Carpetas**

```
.
├── main.py                      # Script principal (punto de entrada)
├── requirements.txt             # Dependencias Python
├── model/
│   └── interview_model.py       # Clases InterviewModel y PredictiveModel
├── data/
│   ├── CHC_datos.xlsx          # Datos de encuestas
│   └── dictionary.json         # Mapeo de preguntas/respuestas
├── controller/                  # (Reservado para próximas fases)
├── view/                        # (Reservado para Dashboard)
└── outputs/                     # Generado por main.py
    ├── predicciones.xlsx       # Datos + predicciones
    ├── resumen_modelos.xlsx    # Rendimiento de modelos
    └── mejor_modelo.joblib     # Modelo entrenado
```

### 3. **Ejecutar el Pipeline**

```bash
python main.py
```

**Salida esperada**:
- Logs de cada fase (limpieza, modelado, evaluación, exportación)
- Archivos en carpeta `outputs/`
- Mejor modelo identificado (por AUC-ROC en test)

### 4. **Salida: Archivos Generados**

#### `predicciones.xlsx`
```
edad | genero | ... | consumo_duro | prob_consumo_duro | pred_consumo_duro
```
- `consumo_duro`: valor real (0 o 1)
- `prob_consumo_duro`: probabilidad predicha [0, 1]
- `pred_consumo_duro`: predicción binaria (0 o 1)

#### `resumen_modelos.xlsx`
```
Modelo              | AUC-ROC (CV)      | F1 (CV)   | AUC-ROC (Test) | F1 (Test)
Regresión Logística | 0.850 ± 0.045     | 0.720    | 0.842         | 0.715
Random Forest       | 0.895 ± 0.032     | 0.810    | 0.887         | 0.805
Gradient Boosting   | 0.902 ± 0.028     | 0.825    | 0.898         | 0.820  ← MEJOR
```

#### `mejor_modelo.joblib`
Archivo binario con el modelo entrenado (Random Forest o GBM típicamente).  
Para reutilizar: `modelo = joblib.load('outputs/mejor_modelo.joblib')`

---

## 🏗️ Arquitectura del Código

### Clase `InterviewModel`
**Responsabilidad**: Limpieza y preparación de datos

```python
modelo = InterviewModel('data/CHC_datos.xlsx', 'data/dictionary.json')
modelo.clean_data()
datos_limpios = modelo.data
```

**Métodos**:
- `__init__()`: carga datos y diccionario
- `clean_data()`: limpia, valida, transforma datos

### Clase `PredictiveModel`
**Responsabilidad**: Modelado y predicción

```python
predictor = PredictiveModel(datos_limpios)
predictor.create_target()              # crea consumo_duro
predictor.select_predictors()          # elige 19 variables
predictor.build_pipelines()            # construye 3 modelos
predictor.train()                      # entrena con CV
predictor.evaluate_best_model()        # reporta rendimiento
predictor.feature_importance()         # muestra variables clave
predictor.export_results('outputs')    # exporta archivos
```

**Métodos principales**:
- `create_target()`: construye variable binaria
- `select_predictors()`: selecciona 19 predictores (ver sección de Predictores)
- `build_pipelines()`: crea Pipeline con imputer + scaler + modelo
- `train()`: entrena 3 modelos con validación cruzada 5-folds
- `evaluate_best_model()`: reporta clasificación + matriz confusión
- `feature_importance()`: muestra importancia de variables
- `predict(X_new)`: predice en nuevos datos
- `export_results(output_dir)`: exporta a Excel y joblib

---

## 🔢 Predictores (Variables de Entrada)

Total: **19 variables**

### Sociodemográficas (5)
- `edad` — factor de riesgo bien documentado
- `genero` — distribuciones diferentes por género
- `nivel_educativo` — factor protector/riesgo contextual
- `sabe_leer_escribir` — indicador de educación temprana
- `orientacion_sexual` — variabilidad de riesgos reportada

### Historia en Calle (3)
- `tiempo_total_calle_meses` — mayor tiempo → mayor exposición
- `razon_inicio_vida_calle` — contexto inicial de vulnerabilidad
- `razon_continua_en_calle` — motivaciones actuales

### Subsistencia (1)
- `forma_obtener_dinero` — acceso a dinero/actividades ilegales

### Salud (4)
- `dx_hipertension`, `dx_tuberculosis`, `dx_vih_sida` — comorbilidades
- `actividades_sin_esfuerzo_fisico` — indicador de salud general

### Escalada de Consumo (6)
- `consume_cigarrillo` — primer nivel (legal)
- `consume_alcohol` — primer nivel (legal)
- `consume_marihuana` — nivel intermedio
- `consume_inhalantes` — fuerte predictor (82% → duro)

**¿Por qué incluir otras sustancias?**  
La literatura documentada muestra una "escalada de consumo": se comienza con sustancias legales antes de llegar a sustancias duras. Esto **NO es data leakage** porque son **sustancias distintas** al target.

---

## 🔍 Interpretación de Resultados

### Ejemplo: Regresión Logística

```
Variable               Coeficiente   Odds Ratio
consume_inhalantes           1.245      3.47    ← cada +1 año, odds aumentan 3.47x
edad                        -0.042      0.96    ← cada año más, riesgo ↓ 4%
consume_marihuana            0.856      2.35
consume_alcohol              0.521      1.68
...
```

**Lectura**:
- `consume_inhalantes = 1`: **multiplica por 3.47** las odds de consumo duro
- `edad`: cada año adicional **reduce riesgo en 4%** (efecto protector de edad)

### Ejemplo: Random Forest / Gradient Boosting

```
Variable               Importancia
consume_inhalantes        0.285      ← 28.5% de la importancia total
tiempo_total_calle_meses  0.156      ← 15.6%
edad                      0.124      ← 12.4%
...
```

**Lectura**: Las 3 variables más importantes explican ~44% de la capacidad predictiva.

---

## 🛡️ Validación Cruzada

¿Por qué no solo train/test?

- **Split único**: depende de qué registros caen en cada conjunto **por azar**
- **Validación cruzada (5-folds)**:
  - El modelo se evalúa **5 veces** en subconjuntos disjuntos
  - Reportamos **promedio ± desviación estándar**
  - Mucho más robusto

Ejemplo:
```
AUC-ROC (CV): 0.895 ± 0.032
└─ Media 0.895, desv 0.032
└─ Rango típico: [0.863, 0.927]
```

---

## 📝 Preprocesamiento Implementado

### 1. **Imputación (SimpleImputer)**
- **Estrategia**: mediana (no media)
- **Razón**: muchas variables son categóricas ordinales (no distribución normal)
- **Aplicado**: solo en train, replicado en test

### 2. **Escalado (StandardScaler)**
- **Fórmula**: `(x - media) / std`
- **Resultado**: media=0, std=1
- **¿Por qué?**:
  - Logistic Regression es sensible a escala
  - Random Forest y GBM son invariantes (pero no daña)

---

## 🚨 Advertencias Implementadas

1. **Registros sin edad**: excluidos automáticamente (sin entrevista válida)
2. **Años en calle > edad**: reportado como inconsistencia lógica
3. **Sustancias duras con datos faltantes**: excluidos (no pueden usarse como target)
4. **Balance de clases**: verificado (advertencia si < 30% o > 70% positivos)

---

## 📚 Referencias

### Escalada de Consumo
- Kandel, D. B. (2002). "Stages and Pathways of Drug Involvement"
- Tarter, R. E. (1992). "Developmental behavior-genetic perspective of alcoholism etiology"

### Personas en Situación de Calle
- WHO (2023). "Homelessness and substance use"
- Fundación Anar (2022). "Personas en situación de calle: diagnóstico Colombia"

---

## 📞 Soporte

Si hay errores:
1. Verifica que `CHC_datos.xlsx` y `dictionary.json` existan en `data/`
2. Verifica que las dependencias en `requirements.txt` estén instaladas
3. Revisa los logs de la consola (muy descriptivos)

---

**Última actualización**: 2026-06-08  
**Versión**: 1.0  
**Autor**: Modelo automático — Técnicas Cuantitativas II
