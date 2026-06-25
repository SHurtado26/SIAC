# SIAC — Sistema Inteligente de Apertura de Cursos

> **Dashboard ejecutivo de planeación académica con simulación Monte Carlo**  
> Desarrollado para el Departamento de Matemáticas y Estadística

---

## 📋 Descripción

SIAC es una solución institucional de Business Intelligence que responde las preguntas clave de la planeación académica universitaria:

| Pregunta estratégica | Vista en el dashboard |
|---|---|
| ¿Qué estudiantes están en riesgo académico? | 👨‍🎓 Riesgo Estudiantil |
| ¿Qué cursos presentan riesgo de baja aprobación? | 🚦 Riesgo por Grupos |
| ¿Cuántos estudiantes aprobarán o reprobarán? | 📈 Predicciones Monte Carlo |
| ¿Cuántos grupos deben abrirse el próximo período? | 📊 Resumen Ejecutivo |
| ¿Cuántos docentes serán necesarios? | 📊 Resumen Ejecutivo (KPI: Docentes) |

---

## 🗂️ Estructura del Proyecto

```
siac/
├── data/
│   ├── matricula_retiros.xlsx       # Estadísticas por NRC + retiros
│   └── Notas_parciales_abril.xlsx   # Notas individuales P1 y P2
├── src/
│   ├── app.py                       # Dashboard principal (Streamlit)
│   └── data_processor.py            # Pipeline de datos + Monte Carlo
├── requirements.txt                 # Dependencias Python
└── README.md                        # Este archivo
```

---

## ⚡ Instalación y Ejecución

### Requisitos
- Python 3.10 o superior
- pip

### Pasos

```bash
# 1. Clonar o descomprimir el proyecto
cd siac

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Ejecutar el dashboard
streamlit run src/app.py
```

El dashboard se abrirá automáticamente en `http://localhost:8501`

---

## 🔬 Metodología: Simulación Monte Carlo

### ¿Por qué Monte Carlo?

El Parcial 3 de cada asignatura **aún no ha ocurrido** en el período 202410. La metodología Monte Carlo es la herramienta estadística adecuada porque:

1. **Incertidumbre cuantificada**: No predice un único número, sino una distribución de probabilidades con intervalos de confianza del 95%.
2. **Datos reales como base**: Usa la media y desviación estándar observadas en P1 y P2 de cada grupo para parametrizar las distribuciones.
3. **Escalable**: Corre 10,000 iteraciones por NRC en segundos, sin requerir infraestructura especial.
4. **Interpretable**: El resultado es directo: "El NRC 2562 tiene 73% de probabilidad de que sus estudiantes aprueben."

### Modelo matemático

```
Nota_Final_sim = 0.35 × P1 + 0.35 × P2 + 0.30 × P3_sim

Donde:
  P3_sim ~ Normal(μ_P3, σ_P3)  ← corte aún no evaluado
  μ_P3   = 0.5 × P1 + 0.5 × P2  ← regresión a la media
  σ_P3   = max(σ_P1, σ_P2, 0.5)  ← incertidumbre conservadora

Regla de aprobación: Nota_Final >= 3.0

Resultado: P(Aprobación) = #{sim | Nota_Final_sim >= 3.0} / 10,000
```

### Clasificación de riesgo

| Nivel | Criterio | Color |
|---|---|---|
| 🔴 ALTO | P(Reprobación) ≥ 40% | Rojo |
| 🟡 MEDIO | 20% ≤ P(Reprobación) < 40% | Ámbar |
| 🟢 BAJO | P(Reprobación) < 20% | Verde |

---

## 📊 Fuentes de Datos

| Archivo | Hoja | Descripción |
|---|---|---|
| `matricula_retiros.xlsx` | Hoja1 | Estadísticas P1 por NRC (prom, desv, min, max) |
| `matricula_retiros.xlsx` | Hoja2 | Matrícula inicial, activos y retiros por NRC |
| `Notas_parciales_abril.xlsx` | Hoja1 | Notas individuales P1 y P2 por estudiante-NRC |

**Período**: 202410  
**Departamento**: Matemáticas y Estadística  
**NRCs analizados**: 128 grupos  
**Asignaturas**: 36  
**Programas**: 27  

---

## 🧭 Navegación del Dashboard

| Vista | Descripción |
|---|---|
| **📊 Resumen Ejecutivo** | KPIs globales, semáforo de riesgo, embudo académico |
| **🚦 Riesgo por Grupos** | Tabla detallada por NRC con filtros, top retiros |
| **👨‍🎓 Riesgo Estudiantil** | Listado de estudiantes en riesgo académico con filtros |
| **📈 Monte Carlo** | Distribuciones simuladas, IC 95%, detalle por NRC |
| **🏛️ Vista por Programa** | Treemap, heatmap programa × asignatura |
| **📋 Tabla Completa** | Exportación CSV con todos los resultados |

---

## 🏛️ Valor Institucional

Esta herramienta permite a la universidad:

- **Anticipar** grupos en riesgo de reprobación masiva antes del cierre del período
- **Planificar** la apertura de grupos del siguiente período con base en proyecciones de demanda
- **Asignar** docentes según carga proyectada
- **Intervenir** tempranamente en estudiantes con acumulado crítico
- **Reportar** a directivas con visualizaciones de alta calidad

---

## 👥 Equipo de Desarrollo

Solución diseñada como un producto institucional de consultoría académica avanzada.

- Científico de Datos Senior  
- Arquitecto de Soluciones Analíticas  
- Especialista en Dashboard Ejecutivo  
- Consultor de Planeación Académica Universitaria  
- Especialista en Modelos Predictivos y Simulación Monte Carlo
