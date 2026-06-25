"""
SIAC - Sistema Inteligente de Apertura de Cursos
Módulo de Procesamiento de Datos y Simulación Monte Carlo

Autor: Equipo Multidisciplinario
Versión: 1.0.0
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# CONSTANTES DEL NEGOCIO
# ─────────────────────────────────────────────────────────────
NOTA_APROBACION = 3.0          # Nota mínima de aprobación en Colombia
NOTA_MIN = 0.0
NOTA_MAX = 5.0
N_SIMULACIONES = 10_000        # Iteraciones Monte Carlo
PESO_PARCIAL_1 = 0.35
PESO_PARCIAL_2 = 0.35
PESO_FINAL_EST = 0.30          # Peso estimado del corte 3 (aún no ocurre)
UMBRAL_RIESGO_ALTO = 0.40      # >40% prob de reprobar → rojo
UMBRAL_RIESGO_MEDIO = 0.20     # 20-40% → ámbar


# ─────────────────────────────────────────────────────────────
# 1. CARGA Y VALIDACIÓN DE DATOS
# ─────────────────────────────────────────────────────────────

def cargar_matriculas(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carga y limpia los datos de matrícula y retiros."""
    
    # --- Hoja 1: estadísticas por NRC (Parcial 1) ---
    raw1 = pd.read_excel(path, sheet_name="Hoja1", header=2)
    raw1 = raw1.rename(columns={
        "Asignatura": "Asignatura", "Profesor": "Profesor", "NRC": "NRC",
        "Matricula Inicial": "Matricula_Inicial",
        "promedio": "Prom_P1", "Desv estand": "Desv_P1",
        "Min": "Min_P1", "Max": "Max_P1"
    })
    # Forward-fill de asignatura (celdas combinadas)
    raw1["Asignatura"] = raw1["Asignatura"].ffill()
    raw1 = raw1.dropna(subset=["NRC"])
    raw1["NRC"] = raw1["NRC"].astype(int)
    raw1["Matricula_Inicial"] = pd.to_numeric(raw1["Matricula_Inicial"], errors="coerce").fillna(0).astype(int)
    raw1["Prom_P1"] = pd.to_numeric(raw1["Prom_P1"], errors="coerce")
    raw1["Desv_P1"] = pd.to_numeric(raw1["Desv_P1"], errors="coerce")
    
    # --- Hoja 2: retiros por NRC ---
    raw2 = pd.read_excel(path, sheet_name="Hoja2")
    raw2.columns = raw2.columns.str.strip()
    raw2 = raw2.rename(columns={
        "Asignatura": "Asignatura",
        "Profesor": "Profesor",
        "NRC": "NRC",
        "Matricula Inicial": "Matricula_Inicial",
        "Estudiantes matriculados": "Matriculados_Actuales",
        "Retirados": "Retirados"
    })
    raw2["NRC"] = raw2["NRC"].astype(int)
    raw2["Retirados"] = pd.to_numeric(raw2["Retirados"], errors="coerce").fillna(0).astype(int)
    raw2["Tasa_Retiro"] = raw2["Retirados"] / raw2["Matricula_Inicial"].replace(0, np.nan)
    raw2["Tasa_Retiro"] = raw2["Tasa_Retiro"].fillna(0).clip(0, 1)

    return raw1, raw2


def cargar_notas(path: str) -> pd.DataFrame:
    """Carga y limpia notas parciales individuales."""
    df = pd.read_excel(path)
    
    # Renombrar columnas relevantes
    df = df.rename(columns={
        "SCBCRSE_TITLE": "Asignatura",
        "CF_PROGRAMA_DESC": "Programa",
        "SGBSTDN_PROGRAM_1": "Codigo_Programa",
        "NOMBRE_ESTUD": "Estudiante",
        "SPRIDEN_ID": "ID_Estudiante",
        "SHRMRKS_CRN": "NRC",
        "CF_COMPONENTE_DESC": "Corte",
        "NVL_SHRMRKS_SCORE_0": "Nota",
        "NVL_SHRMRKS_PERCENTAGE_0": "Porcentaje",
        "Perdidos": "Sin_Nota",
        "SCBCRSE_DEPT_CODE": "Dept_Code"
    })
    
    # Limpiar notas
    df["Nota"] = pd.to_numeric(df["Nota"], errors="coerce")
    df["Nota"] = df["Nota"].clip(NOTA_MIN, NOTA_MAX)
    
    # Solo parciales 1 y 2
    df_parciales = df[df["Corte"].isin(["PARCIAL 1", "PARCIAL 2"])].copy()
    df_parciales["Corte_Num"] = df_parciales["Corte"].map({"PARCIAL 1": 1, "PARCIAL 2": 2})
    
    return df_parciales


# ─────────────────────────────────────────────────────────────
# 2. CONSTRUCCIÓN DE TABLA MAESTRA POR NRC
# ─────────────────────────────────────────────────────────────

def construir_tabla_nrc(df_stats: pd.DataFrame, df_retiros: pd.DataFrame, 
                         df_notas: pd.DataFrame) -> pd.DataFrame:
    """
    Construye tabla maestra por NRC con todos los indicadores académicos.
    Responde: ¿Cuántos grupos existen? ¿Cuáles están en riesgo?
    """
    # Estadísticas de Parcial 2 desde notas individuales
    p2 = df_notas[df_notas["Corte_Num"] == 2].groupby("NRC").agg(
        Prom_P2=("Nota", "mean"),
        Desv_P2=("Nota", "std"),
        Evaluados_P2=("Nota", "count"),
        Sin_Nota_P2=("Sin_Nota", "sum")
    ).reset_index()
    
    p1 = df_notas[df_notas["Corte_Num"] == 1].groupby("NRC").agg(
        Prom_P1_ind=("Nota", "mean"),
        Desv_P1_ind=("Nota", "std"),
        Evaluados_P1=("Nota", "count"),
        Sin_Nota_P1=("Sin_Nota", "sum")
    ).reset_index()
    
    # Unir todo
    base = df_retiros.merge(df_stats[["NRC", "Prom_P1", "Desv_P1"]], on="NRC", how="left")
    base = base.merge(p1, on="NRC", how="left")
    base = base.merge(p2, on="NRC", how="left")
    
    # Programa y asignatura desde notas
    info_nrc = df_notas.groupby("NRC").agg(
        Programa_Princ=("Programa", lambda x: x.value_counts().index[0] if len(x) > 0 else ""),
        Asignatura_Det=("Asignatura", "first")
    ).reset_index()
    base = base.merge(info_nrc, on="NRC", how="left")
    
    # Nota acumulada estimada = 35%*P1 + 35%*P2
    p1_col = base["Prom_P1"].fillna(base["Prom_P1_ind"])
    p2_col = base["Prom_P2"].fillna(0)
    base["Acum_P1_P2"] = (p1_col * PESO_PARCIAL_1 + p2_col * PESO_PARCIAL_2).round(3)
    
    # Tendencia: diferencia P2 - P1
    base["Tendencia"] = (p2_col - p1_col).round(3)
    
    # Tasa de estudiantes "sin nota" (potencial abandono no formal)
    base["Tasa_Sin_Nota"] = (
        base["Sin_Nota_P1"].fillna(0) / base["Matriculados_Actuales"].replace(0, np.nan)
    ).fillna(0).round(3)
    
    return base.copy()


# ─────────────────────────────────────────────────────────────
# 3. SIMULACIÓN MONTE CARLO
# ─────────────────────────────────────────────────────────────

def simular_monte_carlo_nrc(row: pd.Series, n_sim: int = N_SIMULACIONES) -> dict:
    """
    Simulación Monte Carlo para un NRC específico.
    
    JUSTIFICACIÓN METODOLÓGICA:
    ─────────────────────────────
    Monte Carlo es adecuado porque:
    1. La nota final depende de múltiples componentes con incertidumbre (corte 3 no ha ocurrido)
    2. Los datos históricos permiten estimar distribuciones por asignatura
    3. La variabilidad observada (desviación estándar real) alimenta directamente el muestreo
    4. Permite propagar incertidumbre: no solo da un pronóstico puntual sino una 
       distribución de probabilidades (P(aprobación), intervalos de confianza)
    5. Con 10.000 iteraciones, la convergencia del estimador de aprobación 
       tiene error estándar < 0.5%
    
    MODELO:
        Nota_Final_sim = 0.35*P1 + 0.35*P2 + 0.30*P3_sim
        P3_sim ~ Normal(media_P3_est, sigma_P3_est) clipado [0,5]
        media_P3_est = media(P1, P2)  [regresión a la media]
        sigma_P3_est = max(desv_P1, desv_P2, 0.5)  [incertidumbre conservadora]
    """
    np.random.seed(42)
    
    p1 = row.get("Prom_P1", np.nan)
    p2 = row.get("Prom_P2", np.nan)
    desv_p1 = row.get("Desv_P1", 1.0)
    desv_p2 = row.get("Desv_P2", 1.0)
    n_estudiantes = int(row.get("Matriculados_Actuales", 30))
    
    # Si no hay datos, retornar nulos
    if pd.isna(p1) and pd.isna(p2):
        return {
            "Prob_Aprobacion": np.nan,
            "Nota_Final_Media": np.nan,
            "Nota_Final_P5": np.nan,
            "Nota_Final_P95": np.nan,
            "IC_95_Lower": np.nan,
            "IC_95_Upper": np.nan,
            "Est_Aprobaran": np.nan,
            "Est_Reprobaran": np.nan,
            "Nivel_Riesgo": "Sin datos"
        }
    
    p1 = p1 if not pd.isna(p1) else (p2 if not pd.isna(p2) else 3.0)
    p2 = p2 if not pd.isna(p2) else p1
    
    desv_p1 = desv_p1 if not pd.isna(desv_p1) and desv_p1 > 0 else 1.0
    desv_p2 = desv_p2 if not pd.isna(desv_p2) and desv_p2 > 0 else 1.0
    
    # Estimación del corte 3 con regresión a la media
    media_p3_est = 0.5 * p1 + 0.5 * p2
    sigma_p3_est = max(desv_p1, desv_p2, 0.5)
    
    # Simulación vectorizada (n_estudiantes x n_sim)
    sim_p3 = np.random.normal(media_p3_est, sigma_p3_est, size=(n_sim,))
    sim_p3 = np.clip(sim_p3, NOTA_MIN, NOTA_MAX)
    
    nota_final_sim = PESO_PARCIAL_1 * p1 + PESO_PARCIAL_2 * p2 + PESO_FINAL_EST * sim_p3
    nota_final_sim = np.clip(nota_final_sim, NOTA_MIN, NOTA_MAX)
    
    prob_aprobacion = float(np.mean(nota_final_sim >= NOTA_APROBACION))
    media_final = float(np.mean(nota_final_sim))
    p5 = float(np.percentile(nota_final_sim, 5))
    p95 = float(np.percentile(nota_final_sim, 95))
    
    # IC 95% para la probabilidad de aprobación (intervalo de Wilson)
    n = n_sim
    p = prob_aprobacion
    z = 1.96
    ic_lo = max(0, (p + z**2/(2*n) - z*np.sqrt(p*(1-p)/n + z**2/(4*n**2))) / (1 + z**2/n))
    ic_hi = min(1, (p + z**2/(2*n) + z*np.sqrt(p*(1-p)/n + z**2/(4*n**2))) / (1 + z**2/n))
    
    prob_reprobacion = 1 - prob_aprobacion
    est_aprobaran = round(n_estudiantes * prob_aprobacion)
    est_reprobaran = n_estudiantes - est_aprobaran
    
    # Clasificación de riesgo
    if prob_reprobacion >= UMBRAL_RIESGO_ALTO:
        nivel_riesgo = "ALTO"
    elif prob_reprobacion >= UMBRAL_RIESGO_MEDIO:
        nivel_riesgo = "MEDIO"
    else:
        nivel_riesgo = "BAJO"
    
    return {
        "Prob_Aprobacion": round(prob_aprobacion, 4),
        "Prob_Reprobacion": round(prob_reprobacion, 4),
        "Nota_Final_Media": round(media_final, 3),
        "Nota_Final_P5": round(p5, 3),
        "Nota_Final_P95": round(p95, 3),
        "IC_95_Lower": round(ic_lo, 4),
        "IC_95_Upper": round(ic_hi, 4),
        "Est_Aprobaran": est_aprobaran,
        "Est_Reprobaran": est_reprobaran,
        "Nivel_Riesgo": nivel_riesgo
    }


def ejecutar_simulacion(df_nrc: pd.DataFrame) -> pd.DataFrame:
    """Ejecuta Monte Carlo para todos los NRCs y retorna DataFrame enriquecido."""
    resultados = df_nrc.apply(lambda row: simular_monte_carlo_nrc(row), axis=1)
    resultados_df = pd.DataFrame(list(resultados))
    return pd.concat([df_nrc.reset_index(drop=True), resultados_df], axis=1)


# ─────────────────────────────────────────────────────────────
# 4. KPIs INSTITUCIONALES
# ─────────────────────────────────────────────────────────────

def calcular_kpis(df_sim: pd.DataFrame) -> dict:
    """
    Calcula KPIs institucionales clave.
    Responde: ¿Cuántos grupos en riesgo? ¿Cuántos docentes? ¿Cuántos aprueban?
    """
    total_nrc = len(df_sim)
    nrc_con_datos = df_sim.dropna(subset=["Prob_Aprobacion"])
    
    nrc_riesgo_alto = int((nrc_con_datos["Nivel_Riesgo"] == "ALTO").sum())
    nrc_riesgo_medio = int((nrc_con_datos["Nivel_Riesgo"] == "MEDIO").sum())
    nrc_riesgo_bajo = int((nrc_con_datos["Nivel_Riesgo"] == "BAJO").sum())
    
    total_matriculados = int(df_sim["Matriculados_Actuales"].fillna(0).sum())
    total_retirados = int(df_sim["Retirados"].fillna(0).sum())
    total_aprobaran = int(nrc_con_datos["Est_Aprobaran"].fillna(0).sum())
    total_reprobaran = int(nrc_con_datos["Est_Reprobaran"].fillna(0).sum())
    
    tasa_retiro_global = total_retirados / max(df_sim["Matricula_Inicial"].fillna(0).sum(), 1)
    
    docentes_unicos = df_sim["Profesor"].nunique()
    asignaturas_unicas = df_sim["Asignatura"].nunique()
    
    prob_aprobacion_global = nrc_con_datos["Prob_Aprobacion"].mean()
    
    return {
        "total_nrc": total_nrc,
        "total_asignaturas": asignaturas_unicas,
        "docentes_requeridos": docentes_unicos,
        "total_matriculados": total_matriculados,
        "total_retirados": total_retirados,
        "tasa_retiro_pct": round(tasa_retiro_global * 100, 1),
        "nrc_riesgo_alto": nrc_riesgo_alto,
        "nrc_riesgo_medio": nrc_riesgo_medio,
        "nrc_riesgo_bajo": nrc_riesgo_bajo,
        "est_aprobaran": total_aprobaran,
        "est_reprobaran": total_reprobaran,
        "prob_aprobacion_global_pct": round(prob_aprobacion_global * 100, 1),
    }


# ─────────────────────────────────────────────────────────────
# 5. ANÁLISIS POR ASIGNATURA
# ─────────────────────────────────────────────────────────────

def resumen_por_asignatura(df_sim: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega métricas por asignatura. 
    Responde: ¿Cuántos grupos debe abrir la universidad?
    """
    grp = df_sim.groupby("Asignatura").agg(
        Grupos=("NRC", "count"),
        Total_Matriculados=("Matriculados_Actuales", "sum"),
        Total_Retirados=("Retirados", "sum"),
        Prom_Nota_P1=("Prom_P1", "mean"),
        Prom_Nota_P2=("Prom_P2", "mean"),
        Prom_Nota_Final_Est=("Nota_Final_Media", "mean"),
        Prob_Aprobacion_Prom=("Prob_Aprobacion", "mean"),
        Est_Aprobaran=("Est_Aprobaran", "sum"),
        Est_Reprobaran=("Est_Reprobaran", "sum"),
        Grupos_Riesgo_Alto=("Nivel_Riesgo", lambda x: (x == "ALTO").sum()),
        Grupos_Riesgo_Medio=("Nivel_Riesgo", lambda x: (x == "MEDIO").sum()),
    ).reset_index()
    
    grp["Tasa_Retiro_Pct"] = (grp["Total_Retirados"] / grp["Total_Matriculados"].replace(0, np.nan) * 100).round(1)
    grp["Pct_Aprobacion"] = (grp["Prob_Aprobacion_Prom"] * 100).round(1)
    grp["Nivel_Riesgo_Asig"] = grp["Prob_Aprobacion_Prom"].apply(
        lambda p: "ALTO" if (1-p) >= UMBRAL_RIESGO_ALTO 
                  else ("MEDIO" if (1-p) >= UMBRAL_RIESGO_MEDIO else "BAJO") 
                  if not pd.isna(p) else "Sin datos"
    )
    
    return grp.sort_values("Prob_Aprobacion_Prom")


# ─────────────────────────────────────────────────────────────
# 6. ANÁLISIS DE RIESGO ESTUDIANTIL
# ─────────────────────────────────────────────────────────────

def estudiantes_en_riesgo(df_notas: pd.DataFrame) -> pd.DataFrame:
    """
    Identifica estudiantes en riesgo académico.
    Responde: ¿Qué estudiantes están en riesgo académico?
    """
    pivot = df_notas.pivot_table(
        index=["ID_Estudiante", "Estudiante", "NRC", "Asignatura", "Programa"],
        columns="Corte_Num",
        values="Nota"
    ).reset_index()
    pivot.columns = ["ID_Estudiante", "Estudiante", "NRC", "Asignatura", "Programa", "Nota_P1", "Nota_P2"]
    
    # Nota acumulada parcial
    pivot["Acum_P1_P2"] = (pivot["Nota_P1"].fillna(0) * PESO_PARCIAL_1 + 
                            pivot["Nota_P2"].fillna(0) * PESO_PARCIAL_2)
    
    # Riesgo individual
    def clasificar_riesgo_est(row):
        p1 = row["Nota_P1"] if not pd.isna(row["Nota_P1"]) else None
        p2 = row["Nota_P2"] if not pd.isna(row["Nota_P2"]) else None
        
        if p1 is not None and p1 < NOTA_APROBACION and p2 is not None and p2 < NOTA_APROBACION:
            return "CRITICO"  # Reprobó ambos
        elif p1 is not None and p1 < NOTA_APROBACION and p2 is None:
            return "ALTO"
        elif p1 is not None and p1 < 2.0:
            return "ALTO"
        elif row["Acum_P1_P2"] < NOTA_APROBACION * 0.70:
            return "MEDIO"
        else:
            return "BAJO"
    
    pivot["Riesgo_Academico"] = pivot.apply(clasificar_riesgo_est, axis=1)
    pivot["Tendencia"] = (pivot["Nota_P2"].fillna(pivot["Nota_P1"]) - 
                          pivot["Nota_P1"].fillna(pivot["Nota_P2"])).round(2)
    
    return pivot.sort_values(["Riesgo_Academico", "Acum_P1_P2"])


# ─────────────────────────────────────────────────────────────
# 7. FUNCIÓN PRINCIPAL
# ─────────────────────────────────────────────────────────────

def procesar_todo(path_matriculas: str, path_notas: str) -> dict:
    """Pipeline completo: carga → limpieza → simulación → KPIs."""
    
    print("[1/5] Cargando datos de matrícula y retiros...")
    df_stats, df_retiros = cargar_matriculas(path_matriculas)
    
    print("[2/5] Cargando notas parciales individuales...")
    df_notas = cargar_notas(path_notas)
    
    print("[3/5] Construyendo tabla maestra por NRC...")
    df_nrc = construir_tabla_nrc(df_stats, df_retiros, df_notas)
    
    print(f"[4/5] Ejecutando simulación Monte Carlo ({N_SIMULACIONES:,} iteraciones por NRC)...")
    df_sim = ejecutar_simulacion(df_nrc)
    
    print("[5/5] Calculando KPIs y resúmenes...")
    kpis = calcular_kpis(df_sim)
    df_asig = resumen_por_asignatura(df_sim)
    df_riesgo = estudiantes_en_riesgo(df_notas)
    
    # Distribución por programa
    df_programa = df_notas.groupby("Programa").agg(
        Total_Registros=("ID_Estudiante", "count"),
        Estudiantes=("ID_Estudiante", "nunique"),
        NRCs=("NRC", "nunique"),
        Asignaturas=("Asignatura", "nunique"),
        Nota_Prom=("Nota", "mean")
    ).reset_index().sort_values("Estudiantes", ascending=False)
    
    print("\n✅ Procesamiento completado.")
    print(f"   → {len(df_sim)} NRCs analizados")
    print(f"   → {kpis['total_matriculados']} estudiantes matriculados")
    print(f"   → {kpis['nrc_riesgo_alto']} grupos en riesgo ALTO")
    
    return {
        "df_sim": df_sim,
        "df_asig": df_asig,
        "df_riesgo": df_riesgo,
        "df_notas": df_notas,
        "df_programa": df_programa,
        "kpis": kpis
    }
