"""
SIAC - Sistema Inteligente de Apertura de Cursos
Dashboard Ejecutivo Institucional

Tecnología: Streamlit + Plotly
Versión: 1.0.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# ── Path ──────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from data_processor import procesar_todo

# ─────────────────────────────────────────────────────────────
# CONFIGURACIÓN DE LA PÁGINA
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SIAC · Sistema Inteligente de Apertura de Cursos",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# PALETA INSTITUCIONAL
# ─────────────────────────────────────────────────────────────
COLOR = {
    "primary":   "#1E3A5F",   # Azul institucional oscuro
    "accent":    "#2563EB",   # Azul brillante
    "success":   "#10B981",   # Verde
    "warning":   "#F59E0B",   # Ámbar
    "danger":    "#EF4444",   # Rojo
    "neutral":   "#64748B",   # Gris medio
    "bg":        "#F0F4F8",   # Fondo muy suave
    "card_bg":   "#FFFFFF",
    "text_dark": "#0F172A",
    "text_muted":"#64748B",
}

RISK_COLORS = {
    "ALTO":      COLOR["danger"],
    "MEDIO":     COLOR["warning"],
    "BAJO":      COLOR["success"],
    "CRITICO":   "#7C3AED",
    "Sin datos": COLOR["neutral"],
}

# ─────────────────────────────────────────────────────────────
# ESTILOS CSS PERSONALIZADOS
# ─────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  /* Reset y base */
  .main {{ background-color: {COLOR['bg']}; }}
  
  /* Header institucional */
  .siac-header {{
    background: linear-gradient(135deg, {COLOR['primary']} 0%, {COLOR['accent']} 100%);
    padding: 1.5rem 2rem;
    border-radius: 12px;
    color: white;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 20px rgba(30,58,95,0.3);
  }}
  .siac-header h1 {{ 
    font-size: 1.8rem; font-weight: 700; margin: 0; letter-spacing: -0.02em;
  }}
  .siac-header p {{ 
    font-size: 0.9rem; margin: 0.3rem 0 0 0; opacity: 0.85;
  }}
  
  /* KPI Cards */
  .kpi-card {{
    background: {COLOR['card_bg']};
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    box-shadow: 0 1px 6px rgba(0,0,0,0.08);
    border-left: 4px solid {COLOR['accent']};
    margin-bottom: 0.5rem;
  }}
  .kpi-card.danger  {{ border-left-color: {COLOR['danger']}; }}
  .kpi-card.warning {{ border-left-color: {COLOR['warning']}; }}
  .kpi-card.success {{ border-left-color: {COLOR['success']}; }}
  .kpi-card.primary {{ border-left-color: {COLOR['primary']}; }}
  
  .kpi-value {{
    font-size: 2rem; font-weight: 800; color: {COLOR['text_dark']}; line-height: 1;
  }}
  .kpi-label {{ 
    font-size: 0.75rem; color: {COLOR['text_muted']}; 
    text-transform: uppercase; letter-spacing: 0.05em; margin-top: 0.3rem;
  }}
  .kpi-delta {{ font-size: 0.8rem; margin-top: 0.2rem; }}
  .kpi-delta.pos {{ color: {COLOR['success']}; }}
  .kpi-delta.neg {{ color: {COLOR['danger']}; }}
  
  /* Badges semáforo */
  .badge-alto    {{ background:{COLOR['danger']};  color:white; padding:2px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; }}
  .badge-medio   {{ background:{COLOR['warning']}; color:white; padding:2px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; }}
  .badge-bajo    {{ background:{COLOR['success']}; color:white; padding:2px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; }}
  .badge-critico {{ background:#7C3AED;            color:white; padding:2px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; }}
  
  /* Sidebar */
  section[data-testid="stSidebar"] {{
    background: {COLOR['primary']};
  }}
  section[data-testid="stSidebar"] * {{ color: white !important; }}
  section[data-testid="stSidebar"] .stSelectbox > div > div {{ 
    background: rgba(255,255,255,0.1) !important; 
  }}

  /* Sección títulos */
  .section-title {{
    font-size: 1.05rem; font-weight: 700; color: {COLOR['primary']};
    border-bottom: 2px solid {COLOR['accent']}; padding-bottom: 0.4rem;
    margin: 1.2rem 0 0.8rem 0;
  }}
  
  /* Tabla de riesgo */
  .risk-table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
  .risk-table th {{ 
    background: {COLOR['primary']}; color: white; padding: 0.5rem 0.8rem; text-align: left;
  }}
  .risk-table td {{ padding: 0.4rem 0.8rem; border-bottom: 1px solid #e2e8f0; }}
  .risk-table tr:nth-child(even) {{ background: #f8fafc; }}
  .risk-table tr:hover {{ background: #eff6ff; }}
  
  /* Alertas */
  .alert-box {{
    padding: 0.8rem 1rem; border-radius: 8px; margin: 0.5rem 0;
    border-left: 4px solid;
  }}
  .alert-danger  {{ background:#FEF2F2; border-color:{COLOR['danger']};  }}
  .alert-warning {{ background:#FFFBEB; border-color:{COLOR['warning']}; }}
  .alert-success {{ background:#F0FDF4; border-color:{COLOR['success']}; }}
  
  /* Plotly charts fondo blanco */
  .js-plotly-plot {{ border-radius: 10px; }}
  
  /* Ocultar Streamlit branding */
  #MainMenu, footer, header {{ visibility: hidden; }}
  
  /* Scrollbar */
  ::-webkit-scrollbar {{ width: 6px; }}
  ::-webkit-scrollbar-track {{ background: #f1f5f9; }}
  ::-webkit-scrollbar-thumb {{ background: {COLOR['accent']}; border-radius: 3px; }}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# CARGA DE DATOS (CACHE)
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def cargar_datos():
    BASE = os.path.join(os.path.dirname(__file__), "..", "data")
    return procesar_todo(
        path_matriculas=os.path.join(BASE, "matricula_retiros.xlsx"),
        path_notas=os.path.join(BASE, "Notas_parciales_abril.xlsx")
    )


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def badge(nivel: str) -> str:
    cls = {"ALTO": "alto", "MEDIO": "medio", "BAJO": "bajo", 
           "CRITICO": "critico"}.get(nivel, "bajo")
    return f'<span class="badge-{cls}">{nivel}</span>'


def plotly_theme() -> dict:
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=COLOR["text_dark"]),
        margin=dict(l=10, r=10, t=40, b=10),
    )


def kpi_card(value, label, delta=None, tipo="primary"):
    delta_html = ""
    if delta is not None:
        sign = "+" if delta >= 0 else ""
        cls = "pos" if delta >= 0 else "neg"
        delta_html = f'<div class="kpi-delta {cls}">{sign}{delta:.1f}%</div>'
    return f"""
    <div class="kpi-card {tipo}">
      <div class="kpi-value">{value}</div>
      <div class="kpi-label">{label}</div>
      {delta_html}
    </div>
    """


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
def render_sidebar(datos):
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding: 1rem 0 1.5rem;">
          <div style="font-size:2rem;">🎓</div>
          <div style="font-weight:800; font-size:1.1rem; letter-spacing:-0.02em;">SIAC</div>
          <div style="font-size:0.7rem; opacity:0.7; margin-top:2px;">
            Sistema Inteligente de<br>Apertura de Cursos
          </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        pagina = st.radio(
            "Navegación",
            ["📊 Resumen Ejecutivo", 
             "🚦 Riesgo por Grupos",
             "👨‍🎓 Riesgo Estudiantil",
             "📈 Predicciones Monte Carlo",
             "🏛️ Vista por Programa",
             "📋 Tabla Completa"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown('<div style="font-size:0.75rem; opacity:0.7;">Filtros</div>', 
                    unsafe_allow_html=True)
        
        asignaturas = ["Todas"] + sorted(datos["df_sim"]["Asignatura"].dropna().unique().tolist())
        filtro_asig = st.selectbox("Asignatura", asignaturas)
        
        riesgo_opts = ["Todos", "ALTO", "MEDIO", "BAJO"]
        filtro_riesgo = st.selectbox("Nivel de riesgo", riesgo_opts)
        
        st.markdown("---")
        kpis = datos["kpis"]
        st.markdown(f"""
        <div style="font-size:0.7rem; opacity:0.7; margin-bottom:0.5rem;">PERÍODO: 202410</div>
        <div style="font-size:0.75rem;">
          🔴 {kpis['nrc_riesgo_alto']} grupos críticos<br>
          🟡 {kpis['nrc_riesgo_medio']} grupos en alerta<br>
          🟢 {kpis['nrc_riesgo_bajo']} grupos estables
        </div>
        """, unsafe_allow_html=True)
        
    return pagina, filtro_asig, filtro_riesgo


# ─────────────────────────────────────────────────────────────
# VISTAS
# ─────────────────────────────────────────────────────────────

def vista_resumen(datos):
    kpis = datos["kpis"]
    df_asig = datos["df_asig"]
    
    # ── KPIs ─────────────────────────────────────────────────
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1: st.markdown(kpi_card(kpis["total_nrc"], "Grupos abiertos", tipo="primary"), unsafe_allow_html=True)
    with c2: st.markdown(kpi_card(kpis["total_asignaturas"], "Asignaturas", tipo="primary"), unsafe_allow_html=True)
    with c3: st.markdown(kpi_card(kpis["total_matriculados"], "Estudiantes activos", tipo="primary"), unsafe_allow_html=True)
    with c4: st.markdown(kpi_card(kpis["nrc_riesgo_alto"], "Grupos críticos 🔴", tipo="danger"), unsafe_allow_html=True)
    with c5: st.markdown(kpi_card(f"{kpis['prob_aprobacion_global_pct']}%", "Prob. aprobación global", tipo="success"), unsafe_allow_html=True)
    with c6: st.markdown(kpi_card(f"{kpis['tasa_retiro_pct']}%", "Tasa de retiro", tipo="warning"), unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ── Gráficos fila 1 ──────────────────────────────────────
    col_izq, col_der = st.columns([3, 2])
    
    with col_izq:
        st.markdown('<div class="section-title">📊 Distribución de Riesgo por Asignatura</div>', unsafe_allow_html=True)
        df_plot = df_asig.dropna(subset=["Pct_Aprobacion"]).sort_values("Pct_Aprobacion")
        
        color_map = df_plot["Nivel_Riesgo_Asig"].map(RISK_COLORS)
        
        fig = go.Figure(go.Bar(
            x=df_plot["Pct_Aprobacion"],
            y=df_plot["Asignatura"],
            orientation="h",
            marker_color=list(color_map),
            text=[f"{v:.0f}%" for v in df_plot["Pct_Aprobacion"]],
            textposition="outside",
            customdata=np.column_stack([
                df_plot["Grupos"], df_plot["Total_Matriculados"], df_plot["Nivel_Riesgo_Asig"]
            ]),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Prob. Aprobación: %{x:.1f}%<br>"
                "Grupos: %{customdata[0]}<br>"
                "Matriculados: %{customdata[1]}<br>"
                "Riesgo: %{customdata[2]}<extra></extra>"
            )
        ))
        fig.add_vline(x=60, line_dash="dash", line_color=COLOR["warning"], 
                      annotation_text="Umbral 60%", annotation_position="top right")
        fig.update_layout(**plotly_theme(), height=420, 
                          xaxis=dict(range=[0, 115], title="% Probabilidad de Aprobación"),
                          yaxis=dict(title=""))
        st.plotly_chart(fig, use_container_width=True)
    
    with col_der:
        st.markdown('<div class="section-title">🎯 Semáforo de Riesgo</div>', unsafe_allow_html=True)
        
        # Donut chart
        conteos = {
            "Riesgo ALTO": kpis["nrc_riesgo_alto"],
            "Riesgo MEDIO": kpis["nrc_riesgo_medio"],
            "Riesgo BAJO": kpis["nrc_riesgo_bajo"]
        }
        fig_donut = go.Figure(go.Pie(
            labels=list(conteos.keys()),
            values=list(conteos.values()),
            hole=0.6,
            marker_colors=[COLOR["danger"], COLOR["warning"], COLOR["success"]],
            textinfo="percent+value",
            hovertemplate="%{label}: %{value} grupos (%{percent})<extra></extra>"
        ))
        fig_donut.add_annotation(
            text=f"<b>{kpis['total_nrc']}</b><br>grupos",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color=COLOR["text_dark"])
        )
        fig_donut.update_layout(**plotly_theme(), height=220, 
                                showlegend=True,
                                legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig_donut, use_container_width=True)
        
        # Alertas rápidas
        if kpis["nrc_riesgo_alto"] > 0:
            top_criticos = df_asig[df_asig["Nivel_Riesgo_Asig"] == "ALTO"].head(3)
            alertas = "<br>".join([
                f"⚠️ <b>{r['Asignatura'][:25]}...</b> ({r['Pct_Aprobacion']:.0f}% aprob.)" 
                if len(r['Asignatura']) > 25 
                else f"⚠️ <b>{r['Asignatura']}</b> ({r['Pct_Aprobacion']:.0f}% aprob.)"
                for _, r in top_criticos.iterrows()
            ])
            st.markdown(f'<div class="alert-box alert-danger">🔴 <b>GRUPOS CRÍTICOS</b><br>{alertas}</div>', 
                        unsafe_allow_html=True)
    
    # ── Fila 2: Embudo + Scatter ──────────────────────────────
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown('<div class="section-title">🔽 Embudo Académico Proyectado</div>', unsafe_allow_html=True)
        matricula_ini = int(datos["df_sim"]["Matricula_Inicial"].sum())
        activos = kpis["total_matriculados"]
        aprobaran = kpis["est_aprobaran"]
        
        fig_funnel = go.Figure(go.Funnel(
            y=["Matrícula Inicial", "Activos (sin retiro)", 
               "Proyectados a Aprobar", "En Riesgo"],
            x=[matricula_ini, activos, aprobaran, kpis["est_reprobaran"]],
            textinfo="value+percent initial",
            marker_color=[COLOR["accent"], COLOR["primary"], 
                          COLOR["success"], COLOR["danger"]],
        ))
        fig_funnel.update_layout(**plotly_theme(), height=280)
        st.plotly_chart(fig_funnel, use_container_width=True)
    
    with c2:
        st.markdown('<div class="section-title">📉 Nota P1 vs P2 por Asignatura</div>', unsafe_allow_html=True)
        df_scatter = df_asig.dropna(subset=["Prom_Nota_P1", "Prom_Nota_P2"])
        
        fig_sc = px.scatter(
            df_scatter, x="Prom_Nota_P1", y="Prom_Nota_P2",
            size="Total_Matriculados", color="Nivel_Riesgo_Asig",
            color_discrete_map={
                "ALTO": COLOR["danger"], "MEDIO": COLOR["warning"], 
                "BAJO": COLOR["success"], "Sin datos": COLOR["neutral"]
            },
            hover_name="Asignatura",
            hover_data={"Total_Matriculados": True, "Grupos": True},
            labels={"Prom_Nota_P1": "Promedio Parcial 1", "Prom_Nota_P2": "Promedio Parcial 2",
                    "Nivel_Riesgo_Asig": "Nivel de Riesgo"}
        )
        fig_sc.add_shape(type="line", x0=0, y0=0, x1=5, y1=5, 
                         line=dict(dash="dot", color=COLOR["neutral"], width=1))
        fig_sc.add_hline(y=3.0, line_dash="dash", line_color=COLOR["danger"], 
                         annotation_text="Mín. aprobación")
        fig_sc.add_vline(x=3.0, line_dash="dash", line_color=COLOR["danger"])
        fig_sc.update_layout(**plotly_theme(), height=280)
        st.plotly_chart(fig_sc, use_container_width=True)


def vista_riesgo_grupos(datos, filtro_asig, filtro_riesgo):
    st.markdown('<div class="section-title">🚦 Semáforo de Riesgo por Grupo (NRC)</div>', unsafe_allow_html=True)
    
    df = datos["df_sim"].copy()
    if filtro_asig != "Todas":
        df = df[df["Asignatura"] == filtro_asig]
    if filtro_riesgo != "Todos":
        df = df[df["Nivel_Riesgo"] == filtro_riesgo]
    
    df_show = df[[
        "NRC", "Asignatura", "Profesor", "Matriculados_Actuales", "Retirados",
        "Tasa_Retiro", "Prom_P1", "Prom_P2", "Nota_Final_Media",
        "Prob_Aprobacion", "Est_Aprobaran", "Est_Reprobaran", "Nivel_Riesgo"
    ]].copy()
    
    # Mapa de color en heatmap
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Tabla estilizada
        def colorear_riesgo(val):
            colores = {"ALTO": "#FEE2E2", "MEDIO": "#FEF9C3", "BAJO": "#D1FAE5"}
            return f"background-color: {colores.get(val, 'white')}"
        
        df_display = df_show.rename(columns={
            "NRC": "NRC",
            "Matriculados_Actuales": "Activos",
            "Retirados": "Retiros",
            "Tasa_Retiro": "% Retiro",
            "Prom_P1": "Prom P1",
            "Prom_P2": "Prom P2",
            "Nota_Final_Media": "Nota Est.",
            "Prob_Aprobacion": "P(Aprobación)",
            "Est_Aprobaran": "Aprob.",
            "Est_Reprobaran": "Reprobaran",
            "Nivel_Riesgo": "Riesgo"
        }).copy()
        
        df_display["% Retiro"] = (df_display["% Retiro"] * 100).round(1).astype(str) + "%"
        df_display["P(Aprobación)"] = (df_display["P(Aprobación)"] * 100).round(1).astype(str) + "%"
        df_display["Prom P1"] = df_display["Prom P1"].round(2)
        df_display["Prom P2"] = df_display["Prom P2"].round(2)
        df_display["Nota Est."] = df_display["Nota Est."].round(2)
        
        styled = df_display.style.map(
            colorear_riesgo, subset=["Riesgo"]
        ).format(na_rep="—")
        
        st.dataframe(styled, use_container_width=True, height=500)
    
    with col2:
        st.markdown('<div class="section-title">📊 Distribución Riesgo</div>', unsafe_allow_html=True)
        
        if not df["Nivel_Riesgo"].isna().all():
            cnt = df["Nivel_Riesgo"].value_counts()
            fig = go.Figure(go.Bar(
                x=list(cnt.index),
                y=list(cnt.values),
                marker_color=[RISK_COLORS.get(r, COLOR["neutral"]) for r in cnt.index],
                text=list(cnt.values),
                textposition="outside"
            ))
            fig.update_layout(**plotly_theme(), height=200, 
                              xaxis_title="", yaxis_title="Grupos",
                              showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('<div class="section-title">📉 Tasa de Retiro Top 10</div>', unsafe_allow_html=True)
        top_retiro = df.nlargest(10, "Tasa_Retiro")[["Asignatura", "NRC", "Tasa_Retiro"]].copy()
        top_retiro["Tasa_Retiro"] = (top_retiro["Tasa_Retiro"] * 100).round(1)
        top_retiro["Asignatura"] = top_retiro["Asignatura"].str[:20]
        
        fig2 = px.bar(top_retiro, x="Tasa_Retiro", y="Asignatura", 
                      orientation="h",
                      color="Tasa_Retiro",
                      color_continuous_scale=["#10B981", "#F59E0B", "#EF4444"],
                      labels={"Tasa_Retiro": "% Retiro", "Asignatura": ""})
        fig2.update_layout(**plotly_theme(), height=280, coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)


def vista_riesgo_estudiantil(datos):
    st.markdown('<div class="section-title">👨‍🎓 Estudiantes en Riesgo Académico</div>', unsafe_allow_html=True)
    
    df_r = datos["df_riesgo"].copy()
    
    # KPIs riesgo estudiantil
    c1, c2, c3, c4 = st.columns(4)
    criticos = int((df_r["Riesgo_Academico"] == "CRITICO").sum())
    altos = int((df_r["Riesgo_Academico"] == "ALTO").sum())
    medios = int((df_r["Riesgo_Academico"] == "MEDIO").sum())
    bajos = int((df_r["Riesgo_Academico"] == "BAJO").sum())
    
    with c1: st.markdown(kpi_card(criticos, "Casos Críticos (ambos P < 3)", tipo="danger"), unsafe_allow_html=True)
    with c2: st.markdown(kpi_card(altos, "Riesgo Alto", tipo="danger"), unsafe_allow_html=True)
    with c3: st.markdown(kpi_card(medios, "Riesgo Medio", tipo="warning"), unsafe_allow_html=True)
    with c4: st.markdown(kpi_card(bajos, "Sin Riesgo", tipo="success"), unsafe_allow_html=True)
    
    # Filtros
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        nivel_f = st.multiselect("Nivel de riesgo", ["CRITICO", "ALTO", "MEDIO", "BAJO"], 
                                  default=["CRITICO", "ALTO"])
    with col_f2:
        prog_f = st.multiselect("Programa", sorted(df_r["Programa"].dropna().unique()), 
                                  default=[])
    with col_f3:
        asig_f = st.multiselect("Asignatura", sorted(df_r["Asignatura"].dropna().unique()),
                                 default=[])
    
    df_filt = df_r.copy()
    if nivel_f:
        df_filt = df_filt[df_filt["Riesgo_Academico"].isin(nivel_f)]
    if prog_f:
        df_filt = df_filt[df_filt["Programa"].isin(prog_f)]
    if asig_f:
        df_filt = df_filt[df_filt["Asignatura"].isin(asig_f)]
    
    st.caption(f"Mostrando {len(df_filt):,} registros de {len(df_r):,} totales")
    
    col_tab, col_graf = st.columns([3, 2])
    
    with col_tab:
        def color_riesgo_est(val):
            c = {"CRITICO": "#EDE9FE", "ALTO": "#FEE2E2", 
                 "MEDIO": "#FEF9C3", "BAJO": "#D1FAE5"}
            return f"background-color: {c.get(val, 'white')}"
        
        df_show = df_filt[[
            "Estudiante", "ID_Estudiante", "Asignatura", "Programa",
            "Nota_P1", "Nota_P2", "Acum_P1_P2", "Tendencia", "Riesgo_Academico"
        ]].copy()
        df_show["Nota_P1"] = df_show["Nota_P1"].round(2)
        df_show["Nota_P2"] = df_show["Nota_P2"].round(2)
        df_show["Acum_P1_P2"] = df_show["Acum_P1_P2"].round(2)
        df_show["Tendencia"] = df_show["Tendencia"].round(2)
        
        styled = df_show.style.map(color_riesgo_est, subset=["Riesgo_Academico"])
        st.dataframe(styled, use_container_width=True, height=420)
    
    with col_graf:
        # Riesgo por programa
        st.markdown('<div class="section-title">Riesgo por Programa</div>', unsafe_allow_html=True)
        rp = df_r.groupby(["Programa", "Riesgo_Academico"]).size().reset_index(name="n")
        rp_top = rp[rp["Programa"].isin(
            df_r["Programa"].value_counts().head(8).index
        )]
        
        fig_rp = px.bar(
            rp_top, x="n", y="Programa", color="Riesgo_Academico",
            color_discrete_map=RISK_COLORS,
            orientation="h",
            barmode="stack",
            labels={"n": "Estudiantes", "Riesgo_Academico": "Riesgo"}
        )
        fig_rp.update_layout(**plotly_theme(), height=320)
        st.plotly_chart(fig_rp, use_container_width=True)
        
        # Distribución notas P1 vs P2
        st.markdown('<div class="section-title">Distribución Notas</div>', unsafe_allow_html=True)
        df_melt = df_r[["Nota_P1", "Nota_P2"]].melt(var_name="Corte", value_name="Nota")
        fig_hist = px.histogram(df_melt, x="Nota", color="Corte",
                                barmode="overlay", nbins=25,
                                color_discrete_map={"Nota_P1": COLOR["accent"], 
                                                     "Nota_P2": COLOR["success"]},
                                labels={"Nota": "Nota (0-5)", "count": "Frecuencia"})
        fig_hist.add_vline(x=3.0, line_dash="dash", line_color=COLOR["danger"],
                           annotation_text="Mínimo aprobación")
        fig_hist.update_layout(**plotly_theme(), height=250)
        st.plotly_chart(fig_hist, use_container_width=True)


def vista_monte_carlo(datos):
    st.markdown('<div class="section-title">🎲 Simulación Monte Carlo — Predicciones de Nota Final</div>', 
                unsafe_allow_html=True)
    
    st.info("""
    **¿Por qué Monte Carlo?** El Parcial 3 aún no ha ocurrido. En lugar de un pronóstico puntual, 
    la simulación genera 10,000 escenarios posibles basados en las notas históricas reales de cada grupo, 
    produciendo distribuciones de probabilidad con intervalos de confianza del 95%.
    """)
    
    df = datos["df_sim"].dropna(subset=["Prob_Aprobacion"]).copy()
    
    # ── Heatmap NRC x Métrica ─────────────────────────────────
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="section-title">Probabilidad de Aprobación por Grupo</div>', unsafe_allow_html=True)
        
        df_heat = df.sort_values("Prob_Aprobacion").copy()
        df_heat["Asig_corta"] = df_heat["Asignatura"].str[:22]
        df_heat["Label"] = df_heat["Asig_corta"] + " [" + df_heat["NRC"].astype(str) + "]"
        
        fig_heat = go.Figure(go.Bar(
            x=(df_heat["Prob_Aprobacion"] * 100).round(1),
            y=df_heat["Label"],
            orientation="h",
            marker=dict(
                color=(df_heat["Prob_Aprobacion"] * 100),
                colorscale=[[0, COLOR["danger"]], [0.4, COLOR["warning"]], [0.7, COLOR["success"]], [1, "#065F46"]],
                cmin=0, cmax=100,
                colorbar=dict(title="% Aprob.", thickness=12)
            ),
            text=(df_heat["Prob_Aprobacion"] * 100).round(0).astype(int).astype(str) + "%",
            textposition="outside",
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Prob. Aprobación: %{x:.1f}%<br>"
                "<extra></extra>"
            )
        ))
        fig_heat.add_vline(x=60, line_dash="dash", line_color=COLOR["warning"])
        fig_heat.update_layout(**plotly_theme(), height=600,
                                xaxis=dict(range=[0, 115], title="Probabilidad (%)"),
                                yaxis=dict(title=""))
        st.plotly_chart(fig_heat, use_container_width=True)
    
    with col2:
        st.markdown('<div class="section-title">Nota Final Proyectada con IC 95%</div>', unsafe_allow_html=True)
        
        df_ic = df.dropna(subset=["Nota_Final_P5", "Nota_Final_P95"]).sort_values("Nota_Final_Media")
        df_ic["Label"] = df_ic["Asignatura"].str[:18] + " [" + df_ic["NRC"].astype(str) + "]"
        
        fig_ic = go.Figure()
        
        # Barras de error (IC 95%)
        for _, row in df_ic.iterrows():
            color_r = RISK_COLORS.get(row["Nivel_Riesgo"], COLOR["neutral"])
            fig_ic.add_trace(go.Scatter(
                x=[row["Nota_Final_P5"], row["Nota_Final_P95"]],
                y=[row["Label"], row["Label"]],
                mode="lines",
                line=dict(color=color_r, width=3),
                showlegend=False,
                hoverinfo="skip"
            ))
        
        # Puntos medias
        fig_ic.add_trace(go.Scatter(
            x=df_ic["Nota_Final_Media"],
            y=df_ic["Label"],
            mode="markers",
            marker=dict(
                color=[RISK_COLORS.get(r, COLOR["neutral"]) for r in df_ic["Nivel_Riesgo"]],
                size=8, symbol="circle"
            ),
            hovertemplate="<b>%{y}</b><br>Nota media est.: %{x:.2f}<extra></extra>",
            showlegend=False
        ))
        
        fig_ic.add_vline(x=3.0, line_dash="dash", line_color=COLOR["danger"],
                         annotation_text="Mín. aprobación 3.0")
        fig_ic.update_layout(**plotly_theme(), height=600,
                              xaxis=dict(range=[0, 5.5], title="Nota Final Estimada"),
                              yaxis=dict(title=""))
        st.plotly_chart(fig_ic, use_container_width=True)
    
    # ── Detalle por NRC ───────────────────────────────────────
    st.markdown('<div class="section-title">🔍 Detalle Monte Carlo por Grupo</div>', unsafe_allow_html=True)
    
    nrc_sel = st.selectbox(
        "Seleccionar NRC para visualizar distribución simulada:",
        options=df["NRC"].unique(),
        format_func=lambda x: f"NRC {x} — {df[df['NRC']==x]['Asignatura'].values[0]}"
    )
    
    row_sel = df[df["NRC"] == nrc_sel].iloc[0]
    
    # Re-simular para el NRC seleccionado con semilla y graficar
    np.random.seed(42)
    p1 = row_sel.get("Prom_P1", 3.0) or 3.0
    p2 = row_sel.get("Prom_P2", p1) or p1
    desv_p1 = row_sel.get("Desv_P1", 1.0) or 1.0
    desv_p2 = row_sel.get("Desv_P2", 1.0) or 1.0
    
    media_p3 = 0.5 * p1 + 0.5 * p2
    sigma_p3 = max(desv_p1 if not np.isnan(desv_p1) else 1.0, 
                   desv_p2 if not np.isnan(desv_p2) else 1.0, 0.5)
    
    sim_p3 = np.clip(np.random.normal(media_p3, sigma_p3, 10000), 0, 5)
    nota_sim = np.clip(0.35*p1 + 0.35*p2 + 0.30*sim_p3, 0, 5)
    
    col_d1, col_d2, col_d3, col_d4 = st.columns(4)
    with col_d1: st.markdown(kpi_card(f"{row_sel['Prob_Aprobacion']*100:.1f}%", "P(Aprobar)", tipo="success"), unsafe_allow_html=True)
    with col_d2: st.markdown(kpi_card(f"{row_sel['Nota_Final_Media']:.2f}", "Nota Media Est.", tipo="primary"), unsafe_allow_html=True)
    with col_d3: st.markdown(kpi_card(f"[{row_sel['Nota_Final_P5']:.2f} – {row_sel['Nota_Final_P95']:.2f}]", "IC 90%", tipo="primary"), unsafe_allow_html=True)
    with col_d4: 
        nivel = row_sel.get("Nivel_Riesgo", "Sin datos")
        tipo_card = {"ALTO": "danger", "MEDIO": "warning", "BAJO": "success"}.get(nivel, "primary")
        st.markdown(kpi_card(nivel, "Nivel de Riesgo", tipo=tipo_card), unsafe_allow_html=True)
    
    # Histograma de distribución
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=nota_sim, nbinsx=40,
        marker_color=COLOR["accent"],
        opacity=0.75,
        name="Distribución simulada"
    ))
    fig_dist.add_vline(x=3.0, line_dash="dash", line_color=COLOR["danger"], line_width=2,
                       annotation_text="Aprobación (3.0)")
    fig_dist.add_vline(x=float(np.mean(nota_sim)), line_color=COLOR["primary"], line_width=2,
                       annotation_text=f"Media ({np.mean(nota_sim):.2f})")
    fig_dist.update_layout(
        **plotly_theme(), height=280,
        xaxis_title="Nota Final Estimada (0–5)",
        yaxis_title="Frecuencia (de 10.000 simulaciones)",
        title=f"Distribución Monte Carlo · NRC {nrc_sel} · {row_sel['Asignatura']}"
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    # ── Planeación del siguiente período ─────────────────────
    st.markdown("---")
    st.markdown('<div class="section-title">📐 Planeación del Siguiente Período — Grupos y Docentes Proyectados</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="alert-box alert-warning">
    <b>Metodología:</b> Los estudiantes proyectados a <b>reprobar</b> en el período actual
    se suman a la demanda esperada del siguiente período (repetición de materia).
    El número de grupos se calcula dividiendo esa demanda entre la capacidad máxima por grupo.
    Los docentes se estiman considerando carga máxima configurable.
    </div>
    """, unsafe_allow_html=True)

    # Parámetros configurables por el usuario
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        cap_grupo = st.slider("Capacidad máx. por grupo (estudiantes)", 20, 45, 40, 5)
    with col_p2:
        grupos_x_docente = st.slider("Grupos máx. por docente", 1, 5, 3, 1)
    with col_p3:
        factor_demanda = st.slider("Factor de demanda nueva (%)", 80, 130, 100, 5,
                                    help="100% = misma matrícula que el período actual. >100% = crecimiento esperado.")

    # Construcción de la tabla de planeación por asignatura
    df_plan = datos["df_asig"].copy()

    # Estudiantes que reprueban este período (de la simulación)
    df_plan["Reprobaran_Este_Periodo"] = df_plan["Est_Reprobaran"].fillna(0).astype(int)

    # Demanda nueva del siguiente período = matriculados actuales × factor + reprobados
    df_plan["Demanda_Proyectada"] = (
        (df_plan["Total_Matriculados"] * factor_demanda / 100).round(0).astype(int)
        + df_plan["Reprobaran_Este_Periodo"]
    )

    # Grupos necesarios (techo)
    import math
    df_plan["Grupos_Necesarios"] = df_plan["Demanda_Proyectada"].apply(
        lambda x: math.ceil(x / cap_grupo) if x > 0 else 1
    )

    # Grupos actuales
    df_plan["Grupos_Actuales"] = df_plan["Grupos"].astype(int)

    # Delta
    df_plan["Delta_Grupos"] = df_plan["Grupos_Necesarios"] - df_plan["Grupos_Actuales"]

    # Docentes necesarios (techo)
    df_plan["Docentes_Necesarios"] = df_plan["Grupos_Necesarios"].apply(
        lambda x: math.ceil(x / grupos_x_docente)
    )

    # Prioridad de apertura según riesgo
    prioridad_map = {"ALTO": "🔴 Urgente", "MEDIO": "🟡 Prioritario", "BAJO": "🟢 Normal", "Sin datos": "⚪ Sin datos"}
    df_plan["Prioridad"] = df_plan["Nivel_Riesgo_Asig"].map(prioridad_map)

    # KPIs de planeación
    total_grupos_sig = int(df_plan["Grupos_Necesarios"].sum())
    total_docentes_sig = int(df_plan["Docentes_Necesarios"].sum())
    total_demanda_sig = int(df_plan["Demanda_Proyectada"].sum())
    grupos_nuevos = int(df_plan[df_plan["Delta_Grupos"] > 0]["Delta_Grupos"].sum())
    grupos_reducir = int(df_plan[df_plan["Delta_Grupos"] < 0]["Delta_Grupos"].abs().sum())

    kc1, kc2, kc3, kc4, kc5 = st.columns(5)
    with kc1: st.markdown(kpi_card(f"{total_demanda_sig:,}", "Estudiantes proyectados", tipo="primary"), unsafe_allow_html=True)
    with kc2: st.markdown(kpi_card(total_grupos_sig, "Grupos a abrir", tipo="primary"), unsafe_allow_html=True)
    with kc3: st.markdown(kpi_card(total_docentes_sig, "Docentes necesarios", tipo="accent" if False else "primary"), unsafe_allow_html=True)
    with kc4: st.markdown(kpi_card(f"+{grupos_nuevos}", "Grupos adicionales", tipo="warning"), unsafe_allow_html=True)
    with kc5: st.markdown(kpi_card(f"-{grupos_reducir}", "Grupos a reducir", tipo="success"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Gráfico comparativo grupos actuales vs necesarios
    col_g1, col_g2 = st.columns([3, 2])

    with col_g1:
        st.markdown('<div class="section-title">Grupos Actuales vs. Necesarios por Asignatura</div>', unsafe_allow_html=True)

        df_plot_plan = df_plan.sort_values("Grupos_Necesarios", ascending=True)

        fig_plan = go.Figure()
        fig_plan.add_trace(go.Bar(
            name="Grupos actuales",
            y=df_plot_plan["Asignatura"].str[:28],
            x=df_plot_plan["Grupos_Actuales"],
            orientation="h",
            marker_color=COLOR["neutral"],
            opacity=0.7
        ))
        fig_plan.add_trace(go.Bar(
            name="Grupos proyectados",
            y=df_plot_plan["Asignatura"].str[:28],
            x=df_plot_plan["Grupos_Necesarios"],
            orientation="h",
            marker_color=COLOR["accent"],
            opacity=0.85
        ))
        fig_plan.update_layout(
            **plotly_theme(), barmode="overlay", height=480,
            xaxis_title="Número de grupos",
            yaxis_title="",
            legend=dict(orientation="h", y=1.05)
        )
        st.plotly_chart(fig_plan, use_container_width=True)

    with col_g2:
        st.markdown('<div class="section-title">Docentes Necesarios por Asignatura</div>', unsafe_allow_html=True)

        df_doc = df_plan.sort_values("Docentes_Necesarios", ascending=True)
        fig_doc = px.bar(
            df_doc,
            x="Docentes_Necesarios",
            y=df_doc["Asignatura"].str[:22],
            orientation="h",
            color="Docentes_Necesarios",
            color_continuous_scale=[[0, COLOR["success"]], [0.5, COLOR["warning"]], [1, COLOR["danger"]]],
            text="Docentes_Necesarios",
            labels={"Docentes_Necesarios": "Docentes", "y": ""}
        )
        fig_doc.update_traces(textposition="outside")
        fig_doc.update_layout(**plotly_theme(), height=480,
                               coloraxis_showscale=False,
                               xaxis_title="Docentes requeridos")
        st.plotly_chart(fig_doc, use_container_width=True)

    # Tabla detallada de planeación
    st.markdown('<div class="section-title">📋 Tabla Detallada de Planeación por Asignatura</div>', unsafe_allow_html=True)

    cols_tabla = [
        "Asignatura", "Prioridad", "Total_Matriculados", "Reprobaran_Este_Periodo",
        "Demanda_Proyectada", "Grupos_Actuales", "Grupos_Necesarios",
        "Delta_Grupos", "Docentes_Necesarios", "Pct_Aprobacion"
    ]
    df_tabla_plan = df_plan[cols_tabla].copy().sort_values("Grupos_Necesarios", ascending=False)
    df_tabla_plan = df_tabla_plan.rename(columns={
        "Total_Matriculados":      "Matriculados Hoy",
        "Reprobaran_Este_Periodo": "Reprobaran (MC)",
        "Demanda_Proyectada":      "Demanda Sig. Período",
        "Grupos_Actuales":         "Grupos Hoy",
        "Grupos_Necesarios":       "Grupos a Abrir",
        "Delta_Grupos":            "Δ Grupos",
        "Docentes_Necesarios":     "Docentes",
        "Pct_Aprobacion":          "% Aprob. Est."
    })
    df_tabla_plan["% Aprob. Est."] = df_tabla_plan["% Aprob. Est."].round(1).astype(str) + "%"

    def color_delta(val):
        try:
            v = int(val)
            if v > 0:  return "background-color: #FEF9C3; color: #92400E;"
            if v < 0:  return "background-color: #D1FAE5; color: #065F46;"
            return ""
        except:
            return ""

    def color_prioridad(val):
        if "Urgente"    in str(val): return "background-color: #FEE2E2;"
        if "Prioritario" in str(val): return "background-color: #FEF9C3;"
        if "Normal"     in str(val): return "background-color: #D1FAE5;"
        return ""

    styled_plan = (
        df_tabla_plan.style
        .map(color_delta, subset=["Δ Grupos"])
        .map(color_prioridad, subset=["Prioridad"])
    )
    st.dataframe(styled_plan, use_container_width=True, height=520)

    # Descarga
    csv_plan = df_tabla_plan.to_csv(index=False, encoding="utf-8-sig")
    st.download_button(
        "⬇️ Descargar plan de apertura (CSV)",
        data=csv_plan,
        file_name="siac_plan_apertura_siguiente_periodo.csv",
        mime="text/csv"
    )


def vista_programa(datos):
    st.markdown('<div class="section-title">🏛️ Análisis por Programa Académico</div>', unsafe_allow_html=True)
    
    df_prog = datos["df_programa"].copy()
    df_notas = datos["df_notas"].copy()
    
    c1, c2 = st.columns([3, 2])
    
    with c1:
        fig = px.treemap(
            df_prog, path=["Programa"], values="Estudiantes",
            color="Nota_Prom",
            color_continuous_scale=[[0, COLOR["danger"]], [0.5, COLOR["warning"]], [1, COLOR["success"]]],
            color_continuous_midpoint=3.0,
            hover_data={"NRCs": True, "Asignaturas": True, "Nota_Prom": ":.2f"},
            title="Distribución de Estudiantes por Programa (color = nota promedio)"
        )
        fig.update_layout(**plotly_theme(), height=400,
                          coloraxis_colorbar=dict(title="Nota Prom."))
        st.plotly_chart(fig, use_container_width=True)
    
    with c2:
        st.markdown('<div class="section-title">Nota Promedio por Programa</div>', unsafe_allow_html=True)
        fig2 = px.bar(
            df_prog.sort_values("Nota_Prom"), 
            x="Nota_Prom", y="Programa",
            orientation="h",
            color="Nota_Prom",
            color_continuous_scale=[[0, COLOR["danger"]], [0.6, COLOR["warning"]], [1, COLOR["success"]]],
            text="Nota_Prom",
            labels={"Nota_Prom": "Nota Promedio", "Programa": ""}
        )
        fig2.add_vline(x=3.0, line_dash="dash", line_color=COLOR["danger"])
        fig2.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        fig2.update_layout(**plotly_theme(), height=400, 
                            coloraxis_showscale=False,
                            xaxis=dict(range=[0, 5.5]))
        st.plotly_chart(fig2, use_container_width=True)
    
    # Heatmap programa x asignatura
    st.markdown('<div class="section-title">🗺️ Mapa de Calor: Nota Promedio por Programa × Asignatura</div>', unsafe_allow_html=True)
    
    pivot_hm = df_notas[df_notas["Corte_Num"] == 1].pivot_table(
        index="Programa", columns="Asignatura", values="Nota", aggfunc="mean"
    )
    
    top_progs = df_prog.head(10)["Programa"].tolist()
    pivot_hm = pivot_hm[pivot_hm.index.isin(top_progs)]
    
    fig_hm = px.imshow(
        pivot_hm,
        color_continuous_scale=[[0, COLOR["danger"]], [0.4, COLOR["warning"]], [0.6, "#FFF"], [1, COLOR["success"]]],
        zmin=0, zmax=5,
        aspect="auto",
        title="Parcial 1 — Nota Promedio por Programa × Asignatura (Top 10 programas)"
    )
    fig_hm.update_layout(**plotly_theme(), height=380,
                          coloraxis_colorbar=dict(title="Nota"))
    st.plotly_chart(fig_hm, use_container_width=True)


def vista_tabla_completa(datos):
    st.markdown('<div class="section-title">📋 Tabla Maestra — Todos los Grupos</div>', unsafe_allow_html=True)
    
    df = datos["df_sim"].copy()
    
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        buscar = st.text_input("🔍 Buscar asignatura o profesor", "")
    with col_f2:
        cols_show = st.multiselect(
            "Columnas a mostrar",
            ["NRC", "Asignatura", "Profesor", "Matriculados_Actuales", "Retirados",
             "Prom_P1", "Prom_P2", "Nota_Final_Media", "Prob_Aprobacion",
             "Est_Aprobaran", "Est_Reprobaran", "Nivel_Riesgo"],
            default=["NRC", "Asignatura", "Profesor", "Matriculados_Actuales", 
                     "Prom_P1", "Prom_P2", "Prob_Aprobacion", "Nivel_Riesgo"]
        )
    
    if buscar:
        mask = (
            df["Asignatura"].str.contains(buscar, case=False, na=False) |
            df["Profesor"].str.contains(buscar, case=False, na=False)
        )
        df = df[mask]
    
    if cols_show:
        df = df[cols_show]
    
    # Formato
    for col in ["Prom_P1", "Prom_P2", "Nota_Final_Media"]:
        if col in df.columns:
            df[col] = df[col].round(2)
    
    if "Prob_Aprobacion" in df.columns:
        df["Prob_Aprobacion"] = (df["Prob_Aprobacion"] * 100).round(1).astype(str) + "%"
    
    st.caption(f"{len(df)} grupos encontrados")
    
    def color_riesgo(val):
        c = {"ALTO": "#FEE2E2", "MEDIO": "#FEF9C3", "BAJO": "#D1FAE5"}
        return f"background-color: {c.get(val, 'white')}" if "RIESGO" not in val.upper() else ""
    
    if "Nivel_Riesgo" in df.columns:
        st.dataframe(
            df.style.map(color_riesgo, subset=["Nivel_Riesgo"]),
            use_container_width=True, height=600
        )
    else:
        st.dataframe(df, use_container_width=True, height=600)
    
    # Botón de descarga
    csv = datos["df_sim"].to_csv(index=False, encoding="utf-8-sig")
    st.download_button(
        "⬇️ Descargar tabla completa (CSV)",
        data=csv,
        file_name="siac_resultados_completos.csv",
        mime="text/csv"
    )


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    # Header institucional
    st.markdown("""
    <div class="siac-header">
      <h1>🎓 SIAC — Sistema Inteligente de Apertura de Cursos</h1>
      <p>Dashboard Ejecutivo de Planeación Académica · Período 202410 · 
         Dpto. Matemáticas y Estadística &nbsp;|&nbsp; 
         Simulación Monte Carlo con 10,000 iteraciones por grupo</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Cargar datos
    with st.spinner("⏳ Procesando datos y ejecutando simulación Monte Carlo..."):
        datos = cargar_datos()
    
    # Sidebar y navegación
    pagina, filtro_asig, filtro_riesgo = render_sidebar(datos)
    
    # Renderizar vista
    if pagina == "📊 Resumen Ejecutivo":
        vista_resumen(datos)
    elif pagina == "🚦 Riesgo por Grupos":
        vista_riesgo_grupos(datos, filtro_asig, filtro_riesgo)
    elif pagina == "👨‍🎓 Riesgo Estudiantil":
        vista_riesgo_estudiantil(datos)
    elif pagina == "📈 Predicciones Monte Carlo":
        vista_monte_carlo(datos)
    elif pagina == "🏛️ Vista por Programa":
        vista_programa(datos)
    elif pagina == "📋 Tabla Completa":
        vista_tabla_completa(datos)


if __name__ == "__main__":
    main()
