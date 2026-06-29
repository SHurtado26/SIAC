"""
SIAC - Sistema Inteligente de Apertura de Cursos
Dashboard Ejecutivo Institucional — v2.0

Tecnología: Streamlit + Plotly + OpenAI/Gemini
Diseño: Dark Mode · Glassmorphism · Animaciones
"""

import json
import math
import os
import sys

# ── Fix Windows encoding (cp1252 → UTF-8) ────────────────────
# Necesario para que los emoji en print() de data_processor no
# rompan en Windows donde el codec por defecto es cp1252.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Path ──────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from data_processor import (
    procesar_todo,
    NOTA_APROBACION, NOTA_MIN, NOTA_MAX, N_SIMULACIONES,
    PESO_PARCIAL_1, PESO_PARCIAL_2, PESO_FINAL_EST,
    UMBRAL_RIESGO_ALTO, UMBRAL_RIESGO_MEDIO,
)
from chatbot import chat_stream as chatbot_chat_stream, get_suggested_questions

# ─────────────────────────────────────────────────────────────
# CONFIGURACIÓN DE PÁGINA
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SIAC · Sistema Inteligente de Apertura de Cursos",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# PALETA DARK MODE
# ─────────────────────────────────────────────────────────────
C = {
    "primary":    "#6366F1",   # Indigo vibrante
    "secondary":  "#8B5CF6",   # Púrpura
    "accent":     "#22D3EE",   # Cyan
    "success":    "#10B981",   # Verde esmeralda
    "warning":    "#F59E0B",   # Ámbar
    "danger":     "#EF4444",   # Rojo
    "neutral":    "#94A3B8",   # Gris azulado
    "bg":         "#0A0E1A",   # Fondo principal
    "bg2":        "#0F1629",   # Fondo secundario
    "card":       "rgba(15,22,41,0.85)",
    "border":     "rgba(255,255,255,0.09)",
    "text":       "#E2E8F0",
    "muted":      "#64748B",
}

RISK_COLORS = {
    "ALTO":      C["danger"],
    "MEDIO":     C["warning"],
    "BAJO":      C["success"],
    "CRITICO":   C["secondary"],
    "Sin datos": C["neutral"],
}

# ─────────────────────────────────────────────────────────────
# CSS — DARK GLASSMORPHISM
# ─────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@600;700;800&display=swap');

/* ── Base ── */
html, body, [class*="css"] {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}}

/* ── Animations ── */
@keyframes gradientShift {{
    0%   {{ background-position: 0% 50%; }}
    50%  {{ background-position: 100% 50%; }}
    100% {{ background-position: 0% 50%; }}
}}
@keyframes fadeSlideUp {{
    from {{ opacity: 0; transform: translateY(18px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}
@keyframes subtlePulse {{
    0%, 100% {{ opacity: 0.6; transform: scale(1); }}
    50%       {{ opacity: 1;   transform: scale(1.03); }}
}}
@keyframes shimmer {{
    from {{ background-position: -200% center; }}
    to   {{ background-position: 200% center; }}
}}
@keyframes blink {{
    0%, 80%, 100% {{ opacity: 0; }}
    40%            {{ opacity: 1; }}
}}

/* ── Header institucional animado ── */
.siac-header {{
    background: linear-gradient(135deg, #1a1b6b 0%, #312e81 25%, #1e3a8a 55%, #0e7490 80%, #065f46 100%);
    background-size: 300% 300%;
    animation: gradientShift 8s ease infinite;
    padding: 2rem 2.5rem;
    border-radius: 20px;
    color: white;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    border: 1px solid rgba(99,102,241,0.35);
    box-shadow: 0 8px 40px rgba(0,0,0,0.5);
}}
.siac-header::before {{
    content: '';
    position: absolute;
    top: -40%; left: -20%;
    width: 140%; height: 200%;
    background: radial-gradient(ellipse, rgba(99,102,241,0.12) 0%, transparent 60%);
    animation: subtlePulse 5s ease-in-out infinite;
    pointer-events: none;
}}
.siac-header h1 {{
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 1.85rem;
    font-weight: 800;
    margin: 0;
    letter-spacing: -0.03em;
    position: relative;
    text-shadow: 0 2px 10px rgba(0,0,0,0.3);
}}
.siac-header p {{
    font-size: 0.84rem;
    margin: 0.5rem 0 0;
    opacity: 0.78;
    position: relative;
}}

/* ── KPI Cards glassmorphism ── */
.kpi-card {{
    background: {C['card']};
    border: 1px solid {C['border']};
    border-radius: 16px;
    padding: 1.4rem 1.5rem;
    position: relative;
    overflow: hidden;
    transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
    animation: fadeSlideUp 0.45s ease forwards;
    margin-bottom: 0.8rem;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
}}
.kpi-card::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 16px 16px 0 0;
}}
.kpi-card:hover {{
    transform: translateY(-3px);
    box-shadow: 0 16px 48px rgba(0,0,0,0.45);
    border-color: rgba(99,102,241,0.35);
}}
.kpi-card.primary::before {{ background: linear-gradient(90deg, {C['primary']}, {C['secondary']}); }}
.kpi-card.success::before {{ background: linear-gradient(90deg, {C['success']}, #34d399); }}
.kpi-card.danger::before  {{ background: linear-gradient(90deg, {C['danger']},  #f87171); }}
.kpi-card.warning::before {{ background: linear-gradient(90deg, {C['warning']}, #fcd34d); }}
.kpi-card.accent::before  {{ background: linear-gradient(90deg, {C['accent']},  {C['primary']}); }}

.kpi-value {{
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 2.1rem;
    font-weight: 800;
    color: #f1f5f9;
    line-height: 1;
    letter-spacing: -0.02em;
}}
.kpi-label {{
    font-size: 0.7rem;
    color: {C['muted']};
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.4rem;
    font-weight: 500;
}}
.kpi-delta {{ font-size: 0.78rem; margin-top: 0.25rem; font-weight: 600; }}
.kpi-delta.pos {{ color: {C['success']}; }}
.kpi-delta.neg {{ color: {C['danger']}; }}

/* ── Section titles ── */
.section-title {{
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 0.97rem;
    font-weight: 700;
    color: #c7d2fe;
    border-bottom: 2px solid rgba(99,102,241,0.35);
    padding-bottom: 0.45rem;
    margin: 1.4rem 0 0.9rem;
    letter-spacing: -0.01em;
}}

/* ── Badges ── */
.badge-alto    {{ background:rgba(239,68,68,.18);  color:#fca5a5; padding:3px 12px; border-radius:20px; font-size:.72rem; font-weight:600; border:1px solid rgba(239,68,68,.4);   white-space:nowrap; }}
.badge-medio   {{ background:rgba(245,158,11,.18); color:#fcd34d; padding:3px 12px; border-radius:20px; font-size:.72rem; font-weight:600; border:1px solid rgba(245,158,11,.4);  white-space:nowrap; }}
.badge-bajo    {{ background:rgba(16,185,129,.18); color:#6ee7b7; padding:3px 12px; border-radius:20px; font-size:.72rem; font-weight:600; border:1px solid rgba(16,185,129,.4);  white-space:nowrap; }}
.badge-critico {{ background:rgba(139,92,246,.18); color:#c4b5fd; padding:3px 12px; border-radius:20px; font-size:.72rem; font-weight:600; border:1px solid rgba(139,92,246,.4); white-space:nowrap; }}

/* ── Alert boxes ── */
.alert-box {{
    padding: 0.85rem 1.1rem;
    border-radius: 12px;
    margin: 0.7rem 0;
    border-left: 4px solid;
    font-size: 0.87rem;
}}
.alert-danger  {{ background:rgba(239,68,68,.1);  border-color:{C['danger']};  color:#fca5a5; }}
.alert-warning {{ background:rgba(245,158,11,.1); border-color:{C['warning']}; color:#fcd34d; }}
.alert-success {{ background:rgba(16,185,129,.1); border-color:{C['success']}; color:#6ee7b7; }}
.alert-info    {{ background:rgba(99,102,241,.1);  border-color:{C['primary']}; color:#c7d2fe; }}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #06080f 0%, {C['bg']} 100%);
    border-right: 1px solid rgba(255,255,255,0.05);
}}

/* ── Botones ── */
.stButton > button {{
    background: linear-gradient(135deg, #4338ca, {C['primary']}) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
    font-family: 'Inter', sans-serif !important;
}}
.stButton > button:hover {{
    background: linear-gradient(135deg, #3730a3, #4f46e5) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(99,102,241,0.4) !important;
}}
.stDownloadButton > button {{
    background: rgba(16,185,129,0.12) !important;
    color: #34d399 !important;
    border: 1px solid rgba(16,185,129,0.35) !important;
}}
.stDownloadButton > button:hover {{
    background: rgba(16,185,129,0.22) !important;
}}

/* ── Streamlit info/warning/error overrides ── */
.stAlert {{
    border-radius: 12px !important;
}}

/* ── Chat bubbles ── */
.chat-user-msg {{
    background: linear-gradient(135deg, #4338ca, {C['primary']});
    color: white;
    padding: 0.75rem 1.1rem;
    border-radius: 18px 18px 4px 18px;
    margin: 0.4rem 0;
    max-width: 82%;
    margin-left: auto;
    font-size: 0.88rem;
    line-height: 1.5;
    box-shadow: 0 4px 15px rgba(99,102,241,0.3);
    word-wrap: break-word;
}}
.chat-bot-msg {{
    background: rgba(255,255,255,0.05);
    border: 1px solid {C['border']};
    color: {C['text']};
    padding: 0.75rem 1.1rem;
    border-radius: 18px 18px 18px 4px;
    margin: 0.4rem 0;
    max-width: 88%;
    font-size: 0.88rem;
    line-height: 1.55;
    backdrop-filter: blur(8px);
    word-wrap: break-word;
}}
.chat-bot-msg strong {{ color: #c7d2fe; }}
.chat-bot-msg code {{
    background: rgba(99,102,241,0.15);
    padding: 1px 6px;
    border-radius: 4px;
    font-size: 0.82rem;
    color: {C['accent']};
}}
.typing-dot {{
    display: inline-block;
    width: 7px; height: 7px;
    border-radius: 50%;
    background: {C['primary']};
    animation: blink 1.4s infinite ease-in-out;
}}
.typing-dot:nth-child(2) {{ animation-delay: 0.2s; }}
.typing-dot:nth-child(3) {{ animation-delay: 0.4s; }}

/* ── Plotly border ── */
.js-plotly-plot {{ border-radius: 14px; }}

/* ── Scrollbar ── */
::-webkit-scrollbar {{ width: 5px; height: 5px; }}
::-webkit-scrollbar-track {{ background: rgba(255,255,255,0.02); }}
::-webkit-scrollbar-thumb {{ background: rgba(99,102,241,0.45); border-radius: 4px; }}
::-webkit-scrollbar-thumb:hover {{ background: {C['primary']}; }}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header {{ visibility: hidden; }}

/* ── Empty state ── */
.empty-state {{
    text-align: center;
    padding: 3rem;
    color: {C['muted']};
    font-size: 0.9rem;
    border: 1px dashed rgba(255,255,255,0.1);
    border-radius: 16px;
    margin: 1rem 0;
}}
.empty-state .icon {{ font-size: 2.5rem; margin-bottom: 0.7rem; display: block; }}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# CARGA DE DATOS
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def cargar_datos():
    base = os.path.join(os.path.dirname(__file__), "..", "data")
    return procesar_todo(
        path_matriculas=os.path.join(base, "matricula_retiros.xlsx"),
        path_notas=os.path.join(base, "Notas_parciales_abril.xlsx"),
    )


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def badge(nivel: str) -> str:
    cls = {"ALTO": "alto", "MEDIO": "medio", "BAJO": "bajo", "CRITICO": "critico"}.get(nivel, "bajo")
    return f'<span class="badge-{cls}">{nivel}</span>'


def plotly_theme(**extra) -> dict:
    base = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.02)",
        font=dict(family="Inter, sans-serif", color=C["text"], size=12),
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.05)",
            zerolinecolor="rgba(255,255,255,0.1)",
            tickfont=dict(color=C["muted"]),
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.05)",
            zerolinecolor="rgba(255,255,255,0.1)",
            tickfont=dict(color=C["muted"]),
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(255,255,255,0.1)",
            font=dict(color=C["text"]),
        ),
    )
    base.update(extra)
    return base


def kpi_card(value, label: str, delta=None, tipo: str = "primary") -> str:
    delta_html = ""
    if delta is not None:
        sign = "+" if delta >= 0 else ""
        cls  = "pos" if delta >= 0 else "neg"
        delta_html = f'<div class="kpi-delta {cls}">{sign}{delta:.1f}%</div>'
    return f"""
    <div class="kpi-card {tipo}">
      <div class="kpi-value">{value}</div>
      <div class="kpi-label">{label}</div>
      {delta_html}
    </div>"""


def empty_state(icon: str, msg: str) -> None:
    st.markdown(
        f'<div class="empty-state"><span class="icon">{icon}</span>{msg}</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
def render_sidebar(datos) -> tuple:
    with st.sidebar:
        # Logo / título
        st.markdown("""
        <div style="text-align:center; padding:1.5rem 0 1.2rem;">
          <div style="font-size:2.4rem; margin-bottom:0.3rem;">🎓</div>
          <div style="font-family:'Plus Jakarta Sans',sans-serif; font-weight:800;
                      font-size:1.2rem; letter-spacing:-0.02em; color:#e2e8f0;">SIAC</div>
          <div style="font-size:0.68rem; color:#64748b; margin-top:3px; line-height:1.4;">
            Sistema Inteligente de<br>Apertura de Cursos
          </div>
          <div style="margin-top:0.6rem; font-size:0.65rem; color:#6366f1;
                      background:rgba(99,102,241,0.1); padding:3px 10px;
                      border-radius:10px; display:inline-block; border:1px solid rgba(99,102,241,0.3);">
            Período 202410
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<hr style="border-color:rgba(255,255,255,0.08); margin:0 0 1rem;">', unsafe_allow_html=True)

        pagina = st.radio(
            "Navegación",
            [
                "📊 Resumen Ejecutivo",
                "🚦 Riesgo por Grupos",
                "👨‍🎓 Riesgo Estudiantil",
                "📈 Predicciones Monte Carlo",
                "🏛️ Vista por Programa",
                "📋 Tabla Completa",
                "🧮 Metodología",
            ],
            label_visibility="collapsed",
        )

        st.markdown('<hr style="border-color:rgba(255,255,255,0.08); margin:1rem 0;">', unsafe_allow_html=True)

        # Filtros (solo relevantes para ciertas vistas)
        st.markdown('<div style="font-size:0.7rem; color:#64748b; font-weight:600; text-transform:uppercase; letter-spacing:.07em; margin-bottom:.5rem;">Filtros globales</div>', unsafe_allow_html=True)

        asignaturas = ["Todas"] + sorted(datos["df_sim"]["Asignatura"].dropna().unique().tolist())
        filtro_asig = st.selectbox("Asignatura", asignaturas, key="sb_asig")

        riesgo_opts = ["Todos", "ALTO", "MEDIO", "BAJO"]
        filtro_riesgo = st.selectbox("Nivel de riesgo", riesgo_opts, key="sb_riesgo")

        st.markdown('<hr style="border-color:rgba(255,255,255,0.08); margin:1rem 0;">', unsafe_allow_html=True)

        # Mini-KPIs semáforo
        kpis = datos["kpis"]
        pct_alto  = round(kpis["nrc_riesgo_alto"]  / max(kpis["total_nrc"], 1) * 100, 0)
        pct_medio = round(kpis["nrc_riesgo_medio"] / max(kpis["total_nrc"], 1) * 100, 0)
        pct_bajo  = round(kpis["nrc_riesgo_bajo"]  / max(kpis["total_nrc"], 1) * 100, 0)

        st.markdown(f"""
        <div style="font-size:.68rem; color:#64748b; margin-bottom:.5rem; font-weight:600; text-transform:uppercase; letter-spacing:.07em;">Semáforo global</div>
        <div style="display:flex; flex-direction:column; gap:.3rem;">
          <div style="display:flex; align-items:center; gap:.5rem;">
            <div style="width:8px; height:8px; border-radius:50%; background:{C['danger']}; flex-shrink:0; box-shadow:0 0 6px {C['danger']};"></div>
            <div style="font-size:.76rem; color:#e2e8f0; flex:1;">ALTO</div>
            <div style="font-size:.76rem; color:{C['danger']}; font-weight:700;">{kpis['nrc_riesgo_alto']} <span style="color:#64748b;font-weight:400;">({pct_alto:.0f}%)</span></div>
          </div>
          <div style="display:flex; align-items:center; gap:.5rem;">
            <div style="width:8px; height:8px; border-radius:50%; background:{C['warning']}; flex-shrink:0; box-shadow:0 0 6px {C['warning']};"></div>
            <div style="font-size:.76rem; color:#e2e8f0; flex:1;">MEDIO</div>
            <div style="font-size:.76rem; color:{C['warning']}; font-weight:700;">{kpis['nrc_riesgo_medio']} <span style="color:#64748b;font-weight:400;">({pct_medio:.0f}%)</span></div>
          </div>
          <div style="display:flex; align-items:center; gap:.5rem;">
            <div style="width:8px; height:8px; border-radius:50%; background:{C['success']}; flex-shrink:0; box-shadow:0 0 6px {C['success']};"></div>
            <div style="font-size:.76rem; color:#e2e8f0; flex:1;">BAJO</div>
            <div style="font-size:.76rem; color:{C['success']}; font-weight:700;">{kpis['nrc_riesgo_bajo']} <span style="color:#64748b;font-weight:400;">({pct_bajo:.0f}%)</span></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    return pagina, filtro_asig, filtro_riesgo


# ─────────────────────────────────────────────────────────────
# VISTA: RESUMEN EJECUTIVO
# ─────────────────────────────────────────────────────────────
def vista_resumen(datos):
    kpis   = datos["kpis"]
    df_asig = datos["df_asig"]

    # ── KPIs ──────────────────────────────────────────────────
    cols = st.columns(6)
    tarjetas = [
        (kpis["total_nrc"],             "Grupos activos",          "primary"),
        (kpis["total_asignaturas"],      "Asignaturas",             "primary"),
        (f'{kpis["total_matriculados"]:,}', "Estudiantes activos",  "accent"),
        (kpis["nrc_riesgo_alto"],        "Grupos críticos 🔴",       "danger"),
        (f'{kpis["prob_aprobacion_global_pct"]}%', "Prob. aprobación global", "success"),
        (f'{kpis["tasa_retiro_pct"]}%',  "Tasa de retiro",          "warning"),
    ]
    for col, (val, lbl, tipo) in zip(cols, tarjetas):
        with col:
            st.markdown(kpi_card(val, lbl, tipo=tipo), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Gráficos fila 1 ───────────────────────────────────────
    col_izq, col_der = st.columns([3, 2])

    with col_izq:
        st.markdown('<div class="section-title">📊 Probabilidad de Aprobación por Asignatura</div>', unsafe_allow_html=True)
        df_plot = df_asig.dropna(subset=["Pct_Aprobacion"]).sort_values("Pct_Aprobacion")
        color_map = df_plot["Nivel_Riesgo_Asig"].map(RISK_COLORS)

        fig = go.Figure(go.Bar(
            x=df_plot["Pct_Aprobacion"],
            y=df_plot["Asignatura"],
            orientation="h",
            marker=dict(color=list(color_map), opacity=0.85),
            text=[f"{v:.0f}%" for v in df_plot["Pct_Aprobacion"]],
            textposition="outside",
            textfont=dict(color=C["text"], size=10),
            customdata=np.column_stack([
                df_plot["Grupos"], df_plot["Total_Matriculados"], df_plot["Nivel_Riesgo_Asig"]
            ]),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Prob. Aprobación: %{x:.1f}%<br>"
                "Grupos: %{customdata[0]}<br>"
                "Matriculados: %{customdata[1]}<br>"
                "Riesgo: %{customdata[2]}<extra></extra>"
            ),
        ))
        fig.add_vline(x=60, line_dash="dash", line_color=C["warning"], line_width=1.5,
                      annotation_text="Umbral 60%", annotation_font_color=C["warning"])
        fig.update_layout(
            **plotly_theme(height=440, xaxis=dict(range=[0, 120], title="% Probabilidad de Aprobación",
                                                   gridcolor="rgba(255,255,255,0.05)"),
                           yaxis=dict(title=""))
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_der:
        st.markdown('<div class="section-title">🎯 Semáforo de Riesgo Global</div>', unsafe_allow_html=True)

        conteos = {
            "Riesgo ALTO":  kpis["nrc_riesgo_alto"],
            "Riesgo MEDIO": kpis["nrc_riesgo_medio"],
            "Riesgo BAJO":  kpis["nrc_riesgo_bajo"],
        }
        fig_donut = go.Figure(go.Pie(
            labels=list(conteos.keys()),
            values=list(conteos.values()),
            hole=0.65,
            marker=dict(
                colors=[C["danger"], C["warning"], C["success"]],
                line=dict(color=C["bg"], width=2),
            ),
            textinfo="percent+value",
            textfont=dict(color=C["text"]),
            hovertemplate="%{label}: %{value} grupos (%{percent})<extra></extra>",
        ))
        fig_donut.add_annotation(
            text=f"<b>{kpis['total_nrc']}</b><br><span style='font-size:11px'>grupos</span>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color=C["text"]),
        )
        fig_donut.update_layout(
            **plotly_theme(height=240, showlegend=True,
                           legend=dict(orientation="h", y=-0.12, font=dict(size=11)))
        )
        st.plotly_chart(fig_donut, use_container_width=True)

        # Alertas de asignaturas críticas
        criticas = df_asig[df_asig["Nivel_Riesgo_Asig"] == "ALTO"].head(3)
        if not criticas.empty:
            alertas_html = "<br>".join([
                f"⚠️ <b>{r['Asignatura'][:28]}{'...' if len(r['Asignatura']) > 28 else ''}</b> "
                f"({r['Pct_Aprobacion']:.0f}% aprob.)"
                for _, r in criticas.iterrows()
            ])
            st.markdown(
                f'<div class="alert-box alert-danger">🔴 <b>ASIGNATURAS CRÍTICAS</b><br>{alertas_html}</div>',
                unsafe_allow_html=True
            )

    # ── Fila 2: Embudo + Scatter ──────────────────────────────
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section-title">🔽 Embudo Académico Proyectado</div>', unsafe_allow_html=True)
        matricula_ini = int(datos["df_sim"]["Matricula_Inicial"].fillna(0).sum())
        activos       = kpis["total_matriculados"]
        aprobaran     = kpis["est_aprobaran"]
        reprobaran    = kpis["est_reprobaran"]

        fig_funnel = go.Figure(go.Funnel(
            y=["Matrícula Inicial", "Activos (sin retiro)", "Proyectados a Aprobar", "En Riesgo de Reprobar"],
            x=[matricula_ini, activos, aprobaran, reprobaran],
            textinfo="value+percent initial",
            textfont=dict(color="white"),
            marker=dict(color=[C["primary"], C["accent"], C["success"], C["danger"]]),
            connector=dict(line=dict(color="rgba(255,255,255,0.1)", width=1)),
        ))
        fig_funnel.update_layout(**plotly_theme(height=300))
        st.plotly_chart(fig_funnel, use_container_width=True)

    with c2:
        st.markdown('<div class="section-title">📉 Nota P1 vs P2 por Asignatura</div>', unsafe_allow_html=True)
        df_sc = df_asig.dropna(subset=["Prom_Nota_P1", "Prom_Nota_P2"])

        fig_sc = px.scatter(
            df_sc, x="Prom_Nota_P1", y="Prom_Nota_P2",
            size="Total_Matriculados",
            color="Nivel_Riesgo_Asig",
            color_discrete_map={
                "ALTO": C["danger"], "MEDIO": C["warning"],
                "BAJO": C["success"], "Sin datos": C["neutral"],
            },
            hover_name="Asignatura",
            hover_data={"Total_Matriculados": True, "Grupos": True},
            labels={
                "Prom_Nota_P1": "Promedio Parcial 1",
                "Prom_Nota_P2": "Promedio Parcial 2",
                "Nivel_Riesgo_Asig": "Nivel de Riesgo",
            },
        )
        fig_sc.add_shape(type="line", x0=0, y0=0, x1=5, y1=5,
                         line=dict(dash="dot", color=C["neutral"], width=1))
        fig_sc.add_hline(y=3.0, line_dash="dash", line_color=C["danger"], line_width=1,
                         annotation_text="Mín. aprobación 3.0", annotation_font_color=C["danger"])
        fig_sc.add_vline(x=3.0, line_dash="dash", line_color=C["danger"], line_width=1)
        fig_sc.update_layout(**plotly_theme(height=300))
        st.plotly_chart(fig_sc, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# VISTA: RIESGO POR GRUPOS
# ─────────────────────────────────────────────────────────────
def vista_riesgo_grupos(datos, filtro_asig: str, filtro_riesgo: str):
    st.markdown('<div class="section-title">🚦 Semáforo de Riesgo por Grupo (NRC)</div>', unsafe_allow_html=True)

    df = datos["df_sim"].copy()

    # Aplicar filtros
    if filtro_asig != "Todas":
        df = df[df["Asignatura"] == filtro_asig]
    if filtro_riesgo != "Todos":
        df = df[df["Nivel_Riesgo"] == filtro_riesgo]

    if df.empty:
        empty_state("🔍", "No hay grupos que coincidan con los filtros seleccionados.")
        return

    col1, col2 = st.columns([2, 1])

    with col1:
        df_display = df[[
            "NRC", "Asignatura", "Profesor",
            "Matriculados_Actuales", "Retirados", "Tasa_Retiro",
            "Prom_P1", "Prom_P2", "Nota_Final_Media",
            "Prob_Aprobacion", "Est_Aprobaran", "Est_Reprobaran", "Nivel_Riesgo",
        ]].copy().rename(columns={
            "Matriculados_Actuales": "Activos",
            "Retirados": "Retiros",
            "Tasa_Retiro": "% Retiro",
            "Prom_P1": "Prom P1",
            "Prom_P2": "Prom P2",
            "Nota_Final_Media": "Nota Est.",
            "Prob_Aprobacion": "P(Aprob.)",
            "Est_Aprobaran": "Aprob.",
            "Est_Reprobaran": "Reprobar.",
            "Nivel_Riesgo": "Riesgo",
        })

        df_display["% Retiro"]  = (df_display["% Retiro"]  * 100).round(1).astype(str) + "%"
        df_display["P(Aprob.)"] = (df_display["P(Aprob.)"] * 100).round(1).astype(str) + "%"
        df_display["Prom P1"]   = df_display["Prom P1"].round(2)
        df_display["Prom P2"]   = df_display["Prom P2"].round(2)
        df_display["Nota Est."] = df_display["Nota Est."].round(2)

        BG_RISK = {
            "ALTO":  "background-color:#3f1515; color:#fca5a5;",
            "MEDIO": "background-color:#3d2c08; color:#fcd34d;",
            "BAJO":  "background-color:#0f2e1e; color:#6ee7b7;",
        }

        def style_riesgo(val):
            return BG_RISK.get(str(val), "")

        styled = df_display.style.map(style_riesgo, subset=["Riesgo"]).format(na_rep="—")
        st.caption(f"Mostrando **{len(df)}** grupos")
        st.dataframe(styled, use_container_width=True, height=500)

    with col2:
        st.markdown('<div class="section-title">📊 Distribución de Riesgo</div>', unsafe_allow_html=True)

        cnt = df["Nivel_Riesgo"].value_counts()
        if not cnt.empty:
            fig = go.Figure(go.Bar(
                x=list(cnt.index),
                y=list(cnt.values),
                marker=dict(
                    color=[RISK_COLORS.get(r, C["neutral"]) for r in cnt.index],
                    opacity=0.85,
                ),
                text=list(cnt.values),
                textposition="outside",
                textfont=dict(color=C["text"]),
            ))
            fig.update_layout(**plotly_theme(height=210,
                                              xaxis_title="", yaxis_title="Grupos",
                                              showlegend=False))
            st.plotly_chart(fig, use_container_width=True)
        else:
            empty_state("📊", "Sin datos para esta selección.")

        st.markdown('<div class="section-title">📉 Top 10: Mayor Tasa de Retiro</div>', unsafe_allow_html=True)
        top_retiro = (
            df.nlargest(10, "Tasa_Retiro")[["Asignatura", "NRC", "Tasa_Retiro"]]
            .copy()
        )
        top_retiro["Tasa_Retiro"] = (top_retiro["Tasa_Retiro"] * 100).round(1)
        top_retiro["Label"] = top_retiro["Asignatura"].str[:20] + " [" + top_retiro["NRC"].astype(str) + "]"

        if not top_retiro.empty:
            fig2 = px.bar(
                top_retiro, x="Tasa_Retiro", y="Label",
                orientation="h",
                color="Tasa_Retiro",
                color_continuous_scale=[[0, C["success"]], [0.5, C["warning"]], [1, C["danger"]]],
                labels={"Tasa_Retiro": "% Retiro", "Label": ""},
            )
            fig2.update_layout(**plotly_theme(height=310, coloraxis_showscale=False,
                                               xaxis_title="% Tasa de Retiro"))
            st.plotly_chart(fig2, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# VISTA: RIESGO ESTUDIANTIL
# ─────────────────────────────────────────────────────────────
def vista_riesgo_estudiantil(datos):
    st.markdown('<div class="section-title">👨‍🎓 Estudiantes en Riesgo Académico</div>', unsafe_allow_html=True)

    df_r = datos["df_riesgo"].copy()

    # ── KPIs ──────────────────────────────────────────────────
    criticos = int((df_r["Riesgo_Academico"] == "CRITICO").sum())
    altos    = int((df_r["Riesgo_Academico"] == "ALTO").sum())
    medios   = int((df_r["Riesgo_Academico"] == "MEDIO").sum())
    bajos    = int((df_r["Riesgo_Academico"] == "BAJO").sum())

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(kpi_card(criticos, "Casos Críticos (P1 y P2 < 3)", tipo="danger"),  unsafe_allow_html=True)
    with c2: st.markdown(kpi_card(altos,    "Riesgo Alto",                   tipo="danger"),  unsafe_allow_html=True)
    with c3: st.markdown(kpi_card(medios,   "Riesgo Medio",                  tipo="warning"), unsafe_allow_html=True)
    with c4: st.markdown(kpi_card(bajos,    "Sin Riesgo",                    tipo="success"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Filtros ───────────────────────────────────────────────
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        nivel_f = st.multiselect("Nivel de riesgo",
                                 ["CRITICO", "ALTO", "MEDIO", "BAJO"],
                                 default=["CRITICO", "ALTO"])
    with col_f2:
        progs_disp = sorted(df_r["Programa"].dropna().unique())
        prog_f = st.multiselect("Programa", progs_disp, default=[])
    with col_f3:
        asigs_disp = sorted(df_r["Asignatura"].dropna().unique())
        asig_f = st.multiselect("Asignatura", asigs_disp, default=[])

    df_filt = df_r.copy()
    if nivel_f:
        df_filt = df_filt[df_filt["Riesgo_Academico"].isin(nivel_f)]
    if prog_f:
        df_filt = df_filt[df_filt["Programa"].isin(prog_f)]
    if asig_f:
        df_filt = df_filt[df_filt["Asignatura"].isin(asig_f)]

    st.caption(f"Mostrando **{len(df_filt):,}** registros de **{len(df_r):,}** totales")

    if df_filt.empty:
        empty_state("🎯", "No hay estudiantes en riesgo con los filtros seleccionados.")
        return

    col_tab, col_graf = st.columns([3, 2])

    with col_tab:
        BG_RIESGO_EST = {
            "CRITICO": "background-color:#2a0a2e; color:#c4b5fd;",
            "ALTO":    "background-color:#3f1515; color:#fca5a5;",
            "MEDIO":   "background-color:#3d2c08; color:#fcd34d;",
            "BAJO":    "background-color:#0f2e1e; color:#6ee7b7;",
        }

        def style_riesgo_est(val):
            return BG_RIESGO_EST.get(str(val), "")

        df_show = df_filt[[
            "Estudiante", "ID_Estudiante", "Asignatura", "Programa",
            "Nota_P1", "Nota_P2", "Acum_P1_P2", "Tendencia", "Riesgo_Academico",
        ]].copy()
        for col in ["Nota_P1", "Nota_P2", "Acum_P1_P2", "Tendencia"]:
            df_show[col] = df_show[col].round(2)

        styled = df_show.style.map(style_riesgo_est, subset=["Riesgo_Academico"]).format(na_rep="—")
        st.dataframe(styled, use_container_width=True, height=430)

    with col_graf:
        # Riesgo por programa (top 8)
        st.markdown('<div class="section-title">Riesgo por Programa</div>', unsafe_allow_html=True)
        rp = df_r.groupby(["Programa", "Riesgo_Academico"]).size().reset_index(name="n")
        top_progs = df_r["Programa"].value_counts().head(8).index
        rp_top = rp[rp["Programa"].isin(top_progs)]

        if not rp_top.empty:
            fig_rp = px.bar(
                rp_top, x="n", y="Programa",
                color="Riesgo_Academico",
                color_discrete_map=RISK_COLORS,
                orientation="h",
                barmode="stack",
                labels={"n": "Estudiantes", "Riesgo_Academico": "Riesgo"},
            )
            fig_rp.update_layout(**plotly_theme(height=330))
            st.plotly_chart(fig_rp, use_container_width=True)

        # Distribución de notas P1 vs P2
        st.markdown('<div class="section-title">Distribución de Notas</div>', unsafe_allow_html=True)
        df_melt = df_r[["Nota_P1", "Nota_P2"]].melt(var_name="Corte", value_name="Nota").dropna()
        if not df_melt.empty:
            fig_hist = px.histogram(
                df_melt, x="Nota", color="Corte",
                barmode="overlay", nbins=30, opacity=0.75,
                color_discrete_map={"Nota_P1": C["primary"], "Nota_P2": C["accent"]},
                labels={"Nota": "Nota (0–5)", "count": "Frecuencia"},
            )
            fig_hist.add_vline(x=3.0, line_dash="dash", line_color=C["danger"], line_width=1.5,
                               annotation_text="Mín. 3.0", annotation_font_color=C["danger"])
            fig_hist.update_layout(**plotly_theme(height=260))
            st.plotly_chart(fig_hist, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# VISTA: MONTE CARLO
# ─────────────────────────────────────────────────────────────
def vista_monte_carlo(datos):
    st.markdown('<div class="section-title">🎲 Simulación Monte Carlo — Predicciones de Nota Final</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="alert-box alert-info">
    <b>¿Por qué Monte Carlo?</b> El Parcial 3 aún no ha ocurrido. La simulación genera <b>10,000 escenarios</b>
    posibles para cada grupo usando la media y desviación estándar reales de P1 y P2,
    produciendo distribuciones de probabilidad con intervalos de confianza del 95%.
    </div>
    """, unsafe_allow_html=True)

    df = datos["df_sim"].dropna(subset=["Prob_Aprobacion"]).copy()

    if df.empty:
        empty_state("🎲", "No hay datos de simulación disponibles.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">Probabilidad de Aprobación por Grupo</div>', unsafe_allow_html=True)

        df_heat = df.sort_values("Prob_Aprobacion").copy()
        df_heat["Label"] = df_heat["Asignatura"].str[:22] + " [" + df_heat["NRC"].astype(str) + "]"

        fig_heat = go.Figure(go.Bar(
            x=(df_heat["Prob_Aprobacion"] * 100).round(1),
            y=df_heat["Label"],
            orientation="h",
            marker=dict(
                color=(df_heat["Prob_Aprobacion"] * 100),
                colorscale=[
                    [0,   C["danger"]],
                    [0.4, C["warning"]],
                    [0.7, C["success"]],
                    [1,   "#065F46"],
                ],
                cmin=0, cmax=100,
                colorbar=dict(title="% Aprob.", thickness=12, tickfont=dict(color=C["muted"])),
            ),
            text=(df_heat["Prob_Aprobacion"] * 100).round(0).astype(int).astype(str) + "%",
            textposition="outside",
            textfont=dict(color=C["text"], size=9),
            hovertemplate="<b>%{y}</b><br>Prob. Aprobación: %{x:.1f}%<extra></extra>",
        ))
        fig_heat.add_vline(x=60, line_dash="dash", line_color=C["warning"], line_width=1,
                           annotation_text="60%", annotation_font_color=C["warning"])
        fig_heat.update_layout(**plotly_theme(height=620,
                                               xaxis=dict(range=[0, 120], title="Probabilidad (%)"),
                                               yaxis=dict(title="")))
        st.plotly_chart(fig_heat, use_container_width=True)

    with col2:
        st.markdown('<div class="section-title">Nota Final Proyectada con IC 95%</div>', unsafe_allow_html=True)

        df_ic = df.dropna(subset=["Nota_Final_P5", "Nota_Final_P95"]).sort_values("Nota_Final_Media")
        df_ic["Label"] = df_ic["Asignatura"].str[:18] + " [" + df_ic["NRC"].astype(str) + "]"

        fig_ic = go.Figure()
        for _, row in df_ic.iterrows():
            color_r = RISK_COLORS.get(row["Nivel_Riesgo"], C["neutral"])
            fig_ic.add_trace(go.Scatter(
                x=[row["Nota_Final_P5"], row["Nota_Final_P95"]],
                y=[row["Label"], row["Label"]],
                mode="lines",
                line=dict(color=color_r, width=2.5),
                showlegend=False,
                hoverinfo="skip",
            ))

        fig_ic.add_trace(go.Scatter(
            x=df_ic["Nota_Final_Media"],
            y=df_ic["Label"],
            mode="markers",
            marker=dict(
                color=[RISK_COLORS.get(r, C["neutral"]) for r in df_ic["Nivel_Riesgo"]],
                size=7, symbol="circle",
            ),
            hovertemplate="<b>%{y}</b><br>Nota media est.: %{x:.2f}<extra></extra>",
            showlegend=False,
        ))
        fig_ic.add_vline(x=3.0, line_dash="dash", line_color=C["danger"], line_width=1.5,
                         annotation_text="Mín. 3.0", annotation_font_color=C["danger"])
        fig_ic.update_layout(**plotly_theme(height=620,
                                             xaxis=dict(range=[0, 5.5], title="Nota Final Estimada"),
                                             yaxis=dict(title="")))
        st.plotly_chart(fig_ic, use_container_width=True)

    # ── Detalle por NRC ───────────────────────────────────────
    st.markdown('<div class="section-title">🔍 Detalle Monte Carlo por Grupo</div>', unsafe_allow_html=True)

    nrcs_disp = df["NRC"].unique()
    if len(nrcs_disp) == 0:
        empty_state("🔍", "No hay NRCs disponibles.")
        return

    nrc_sel = st.selectbox(
        "Seleccionar NRC para visualizar distribución simulada:",
        options=nrcs_disp,
        format_func=lambda x: (
            f"NRC {x} — {df[df['NRC']==x]['Asignatura'].values[0]}"
            if len(df[df["NRC"] == x]) > 0 else f"NRC {x}"
        ),
    )

    matches = df[df["NRC"] == nrc_sel]
    if matches.empty:
        empty_state("⚠️", "No se encontraron datos para este NRC.")
        return

    row_sel = matches.iloc[0]

    # Re-simular con valores seguros ante NaN
    p1      = float(row_sel.get("Prom_P1", 3.0) or 3.0)
    p2_val  = row_sel.get("Prom_P2", None)
    p2      = float(p2_val) if pd.notna(p2_val) else p1

    desv_p1_val = row_sel.get("Desv_P1", None)
    desv_p2_val = row_sel.get("Desv_P2", None)
    desv_p1 = float(desv_p1_val) if pd.notna(desv_p1_val) and desv_p1_val > 0 else 1.0
    desv_p2 = float(desv_p2_val) if pd.notna(desv_p2_val) and desv_p2_val > 0 else 1.0

    media_p3 = 0.5 * p1 + 0.5 * p2
    sigma_p3 = max(desv_p1, desv_p2, 0.5)

    np.random.seed(42)
    sim_p3    = np.clip(np.random.normal(media_p3, sigma_p3, 10_000), 0, 5)
    nota_sim  = np.clip(0.35 * p1 + 0.35 * p2 + 0.30 * sim_p3, 0, 5)

    # KPIs del NRC seleccionado
    col_d1, col_d2, col_d3, col_d4 = st.columns(4)
    prob_aprob = float(row_sel.get("Prob_Aprobacion", np.mean(nota_sim >= 3.0)) or 0)
    nota_media = float(row_sel.get("Nota_Final_Media", np.mean(nota_sim)) or np.mean(nota_sim))
    p5_val     = float(row_sel.get("Nota_Final_P5",  np.percentile(nota_sim, 5))  or 0)
    p95_val    = float(row_sel.get("Nota_Final_P95", np.percentile(nota_sim, 95)) or 0)
    nivel      = str(row_sel.get("Nivel_Riesgo", "Sin datos") or "Sin datos")
    tipo_card  = {"ALTO": "danger", "MEDIO": "warning", "BAJO": "success"}.get(nivel, "primary")

    with col_d1: st.markdown(kpi_card(f"{prob_aprob*100:.1f}%", "P(Aprobar)",     tipo="success"), unsafe_allow_html=True)
    with col_d2: st.markdown(kpi_card(f"{nota_media:.2f}",      "Nota Media Est.", tipo="primary"), unsafe_allow_html=True)
    with col_d3: st.markdown(kpi_card(f"[{p5_val:.2f}–{p95_val:.2f}]", "IC 90%", tipo="accent"),  unsafe_allow_html=True)
    with col_d4: st.markdown(kpi_card(nivel,                    "Nivel de Riesgo", tipo=tipo_card), unsafe_allow_html=True)

    # Histograma de distribución simulada
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=nota_sim, nbinsx=45,
        marker=dict(color=C["primary"], opacity=0.75,
                    line=dict(color=C["secondary"], width=0.5)),
        name="Distribución simulada",
    ))
    fig_dist.add_vline(x=3.0, line_dash="dash", line_color=C["danger"], line_width=2,
                       annotation_text="Aprobación (3.0)", annotation_font_color=C["danger"])
    fig_dist.add_vline(x=float(np.mean(nota_sim)), line_color=C["accent"], line_width=2,
                       annotation_text=f"Media ({np.mean(nota_sim):.2f})",
                       annotation_font_color=C["accent"])
    fig_dist.update_layout(
        **plotly_theme(height=290),
        xaxis_title="Nota Final Estimada (0–5)",
        yaxis_title="Frecuencia (de 10,000 simulaciones)",
        title=dict(
            text=f"Distribución Monte Carlo · NRC {nrc_sel} · {row_sel.get('Asignatura', '')}",
            font=dict(color=C["text"], size=13),
        ),
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    # ── Planeación del siguiente período ──────────────────────
    st.markdown("---")
    st.markdown('<div class="section-title">📐 Planeación del Siguiente Período — Grupos y Docentes Proyectados</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="alert-box alert-warning">
    <b>Metodología:</b> Los estudiantes proyectados a <b>reprobar</b> en el período actual se suman a la
    demanda del siguiente período. Los grupos se calculan dividiendo esa demanda entre la capacidad
    máxima configurada. Los docentes se estiman con la carga máxima configurable.
    </div>
    """, unsafe_allow_html=True)

    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        cap_grupo = st.slider("Capacidad máx. por grupo (estudiantes)", 20, 45, 40, 5)
    with col_p2:
        grupos_x_docente = st.slider("Grupos máx. por docente", 1, 5, 3, 1)
    with col_p3:
        factor_demanda = st.slider("Factor de demanda nueva (%)", 80, 130, 100, 5,
                                   help="100% = misma matrícula. >100% = crecimiento esperado.")

    df_plan = datos["df_asig"].copy()
    df_plan["Reprobaran_Este_Periodo"] = df_plan["Est_Reprobaran"].fillna(0).astype(int)
    df_plan["Demanda_Proyectada"] = (
        (df_plan["Total_Matriculados"] * factor_demanda / 100).round(0).astype(int)
        + df_plan["Reprobaran_Este_Periodo"]
    )
    df_plan["Grupos_Necesarios"] = df_plan["Demanda_Proyectada"].apply(
        lambda x: math.ceil(x / cap_grupo) if x > 0 else 1
    )
    df_plan["Grupos_Actuales"] = df_plan["Grupos"].astype(int)
    df_plan["Delta_Grupos"] = df_plan["Grupos_Necesarios"] - df_plan["Grupos_Actuales"]
    df_plan["Docentes_Necesarios"] = df_plan["Grupos_Necesarios"].apply(
        lambda x: math.ceil(x / grupos_x_docente)
    )
    prioridad_map = {
        "ALTO": "🔴 Urgente", "MEDIO": "🟡 Prioritario",
        "BAJO": "🟢 Normal",  "Sin datos": "⚪ Sin datos",
    }
    df_plan["Prioridad"] = df_plan["Nivel_Riesgo_Asig"].map(prioridad_map).fillna("⚪ Sin datos")

    total_grupos_sig  = int(df_plan["Grupos_Necesarios"].sum())
    total_docentes_sig = int(df_plan["Docentes_Necesarios"].sum())
    total_demanda_sig = int(df_plan["Demanda_Proyectada"].sum())
    grupos_nuevos     = int(df_plan[df_plan["Delta_Grupos"] > 0]["Delta_Grupos"].sum())
    grupos_reducir    = int(df_plan[df_plan["Delta_Grupos"] < 0]["Delta_Grupos"].abs().sum())

    kc1, kc2, kc3, kc4, kc5 = st.columns(5)
    with kc1: st.markdown(kpi_card(f"{total_demanda_sig:,}", "Estudiantes proyectados", tipo="primary"), unsafe_allow_html=True)
    with kc2: st.markdown(kpi_card(total_grupos_sig,         "Grupos a abrir",          tipo="primary"), unsafe_allow_html=True)
    with kc3: st.markdown(kpi_card(total_docentes_sig,       "Docentes necesarios",     tipo="accent"),  unsafe_allow_html=True)
    with kc4: st.markdown(kpi_card(f"+{grupos_nuevos}",      "Grupos adicionales",      tipo="warning"), unsafe_allow_html=True)
    with kc5: st.markdown(kpi_card(f"-{grupos_reducir}",     "Grupos a reducir",        tipo="success"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_g1, col_g2 = st.columns([3, 2])

    with col_g1:
        st.markdown('<div class="section-title">Grupos Actuales vs. Necesarios por Asignatura</div>', unsafe_allow_html=True)
        df_pp = df_plan.sort_values("Grupos_Necesarios", ascending=True)
        fig_plan = go.Figure()
        fig_plan.add_trace(go.Bar(
            name="Grupos actuales",
            y=df_pp["Asignatura"].str[:28],
            x=df_pp["Grupos_Actuales"],
            orientation="h",
            marker=dict(color=C["neutral"], opacity=0.55),
        ))
        fig_plan.add_trace(go.Bar(
            name="Grupos proyectados",
            y=df_pp["Asignatura"].str[:28],
            x=df_pp["Grupos_Necesarios"],
            orientation="h",
            marker=dict(color=C["primary"], opacity=0.85),
        ))
        fig_plan.update_layout(**plotly_theme(barmode="overlay", height=500,
                                               xaxis_title="Número de grupos",
                                               legend=dict(orientation="h", y=1.05)))
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
            color_continuous_scale=[[0, C["success"]], [0.5, C["warning"]], [1, C["danger"]]],
            text="Docentes_Necesarios",
            labels={"Docentes_Necesarios": "Docentes", "y": ""},
        )
        fig_doc.update_traces(textposition="outside", textfont=dict(color=C["text"]))
        fig_doc.update_layout(**plotly_theme(height=500, coloraxis_showscale=False,
                                              xaxis_title="Docentes requeridos"))
        st.plotly_chart(fig_doc, use_container_width=True)

    # Tabla detallada de planeación
    st.markdown('<div class="section-title">📋 Tabla Detallada de Planeación por Asignatura</div>', unsafe_allow_html=True)

    cols_tabla = [
        "Asignatura", "Prioridad", "Total_Matriculados", "Reprobaran_Este_Periodo",
        "Demanda_Proyectada", "Grupos_Actuales", "Grupos_Necesarios",
        "Delta_Grupos", "Docentes_Necesarios", "Pct_Aprobacion",
    ]
    df_tabla = df_plan[cols_tabla].copy().sort_values("Grupos_Necesarios", ascending=False)
    df_tabla = df_tabla.rename(columns={
        "Total_Matriculados":      "Matriculados Hoy",
        "Reprobaran_Este_Periodo": "Reprobaran (MC)",
        "Demanda_Proyectada":      "Demanda Sig. Período",
        "Grupos_Actuales":         "Grupos Hoy",
        "Grupos_Necesarios":       "Grupos a Abrir",
        "Delta_Grupos":            "Δ Grupos",
        "Docentes_Necesarios":     "Docentes",
        "Pct_Aprobacion":          "% Aprob. Est.",
    })
    df_tabla["% Aprob. Est."] = df_tabla["% Aprob. Est."].round(1).astype(str) + "%"

    def color_delta(val):
        try:
            v = int(val)
            if v > 0: return "background-color:#3d2c08; color:#fcd34d;"
            if v < 0: return "background-color:#0f2e1e; color:#6ee7b7;"
        except Exception:
            pass
        return ""

    def color_prioridad(val):
        s = str(val)
        if "Urgente"     in s: return "background-color:#3f1515; color:#fca5a5;"
        if "Prioritario" in s: return "background-color:#3d2c08; color:#fcd34d;"
        if "Normal"      in s: return "background-color:#0f2e1e; color:#6ee7b7;"
        return ""

    styled_plan = (
        df_tabla.style
        .map(color_delta,      subset=["Δ Grupos"])
        .map(color_prioridad,  subset=["Prioridad"])
    )
    st.dataframe(styled_plan, use_container_width=True, height=520)

    csv_plan = df_tabla.to_csv(index=False, encoding="utf-8-sig")
    st.download_button(
        "⬇️ Descargar plan de apertura (CSV)",
        data=csv_plan,
        file_name="siac_plan_apertura_siguiente_periodo.csv",
        mime="text/csv",
    )


# ─────────────────────────────────────────────────────────────
# VISTA: ANÁLISIS POR PROGRAMA
# ─────────────────────────────────────────────────────────────
def vista_programa(datos):
    st.markdown('<div class="section-title">🏛️ Análisis por Programa Académico</div>', unsafe_allow_html=True)

    df_prog  = datos["df_programa"].copy()
    df_notas = datos["df_notas"].copy()

    c1, c2 = st.columns([3, 2])

    with c1:
        fig = px.treemap(
            df_prog, path=["Programa"], values="Estudiantes",
            color="Nota_Prom",
            color_continuous_scale=[[0, C["danger"]], [0.5, C["warning"]], [1, C["success"]]],
            color_continuous_midpoint=3.0,
            hover_data={"NRCs": True, "Asignaturas": True, "Nota_Prom": ":.2f"},
            title="Distribución de Estudiantes por Programa (color = nota promedio P1)",
        )
        fig.update_layout(**plotly_theme(height=420,
                                          coloraxis_colorbar=dict(title="Nota Prom.",
                                                                   tickfont=dict(color=C["muted"]))))
        fig.update_traces(textfont=dict(color="white"))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<div class="section-title">Nota Promedio por Programa</div>', unsafe_allow_html=True)
        fig2 = px.bar(
            df_prog.sort_values("Nota_Prom"),
            x="Nota_Prom", y="Programa",
            orientation="h",
            color="Nota_Prom",
            color_continuous_scale=[[0, C["danger"]], [0.6, C["warning"]], [1, C["success"]]],
            text="Nota_Prom",
            labels={"Nota_Prom": "Nota Promedio", "Programa": ""},
        )
        fig2.add_vline(x=3.0, line_dash="dash", line_color=C["danger"], line_width=1)
        fig2.update_traces(texttemplate="%{text:.2f}", textposition="outside",
                           textfont=dict(color=C["text"]))
        fig2.update_layout(**plotly_theme(height=420, coloraxis_showscale=False,
                                           xaxis=dict(range=[0, 5.5])))
        st.plotly_chart(fig2, use_container_width=True)

    # Heatmap programa × asignatura
    st.markdown('<div class="section-title">🗺️ Mapa de Calor — Nota P1 por Programa × Asignatura (Top 10 programas)</div>',
                unsafe_allow_html=True)

    pivot_hm = df_notas[df_notas["Corte_Num"] == 1].pivot_table(
        index="Programa", columns="Asignatura", values="Nota", aggfunc="mean"
    )
    top_progs = df_prog.head(10)["Programa"].tolist()
    pivot_hm = pivot_hm[pivot_hm.index.isin(top_progs)]

    if not pivot_hm.empty:
        fig_hm = px.imshow(
            pivot_hm,
            color_continuous_scale=[[0, C["danger"]], [0.4, C["warning"]], [0.7, "#E2E8F0"], [1, C["success"]]],
            zmin=0, zmax=5,
            aspect="auto",
        )
        fig_hm.update_layout(**plotly_theme(height=390,
                                             coloraxis_colorbar=dict(title="Nota",
                                                                      tickfont=dict(color=C["muted"]))))
        fig_hm.update_traces(hoverongaps=False)
        st.plotly_chart(fig_hm, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# VISTA: TABLA COMPLETA
# ─────────────────────────────────────────────────────────────
def vista_tabla_completa(datos):
    st.markdown('<div class="section-title">📋 Tabla Maestra — Todos los Grupos</div>', unsafe_allow_html=True)

    df = datos["df_sim"].copy()

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        buscar = st.text_input("🔍 Buscar asignatura o profesor", "")
    with col_f2:
        todas_cols = [
            "NRC", "Asignatura", "Profesor", "Matriculados_Actuales", "Retirados",
            "Prom_P1", "Prom_P2", "Nota_Final_Media",
            "Prob_Aprobacion", "Est_Aprobaran", "Est_Reprobaran", "Nivel_Riesgo",
        ]
        cols_show = st.multiselect(
            "Columnas a mostrar",
            todas_cols,
            default=["NRC", "Asignatura", "Profesor", "Matriculados_Actuales",
                     "Prom_P1", "Prom_P2", "Prob_Aprobacion", "Nivel_Riesgo"],
        )

    if buscar:
        mask = (
            df["Asignatura"].str.contains(buscar, case=False, na=False)
            | df["Profesor"].str.contains(buscar, case=False, na=False)
        )
        df = df[mask]

    if cols_show:
        # Filtrar solo las columnas que existen en el dataframe
        cols_validas = [c for c in cols_show if c in df.columns]
        df = df[cols_validas]

    # Formateo numérico
    for col in ["Prom_P1", "Prom_P2", "Nota_Final_Media"]:
        if col in df.columns:
            df[col] = df[col].round(2)
    if "Prob_Aprobacion" in df.columns:
        df["Prob_Aprobacion"] = (df["Prob_Aprobacion"] * 100).round(1).astype(str) + "%"

    st.caption(f"**{len(df)}** grupos encontrados")

    if df.empty:
        empty_state("🔍", "No se encontraron grupos con esa búsqueda.")
        return

    # Coloreado de riesgo — FIXED: manejo seguro de NaN y lógica correcta
    BG_RISK_TABLE = {
        "ALTO":  "background-color:#3f1515; color:#fca5a5;",
        "MEDIO": "background-color:#3d2c08; color:#fcd34d;",
        "BAJO":  "background-color:#0f2e1e; color:#6ee7b7;",
    }

    def color_riesgo(val):
        if not isinstance(val, str):
            return ""
        return BG_RISK_TABLE.get(val, "")

    if "Nivel_Riesgo" in df.columns:
        st.dataframe(
            df.style.map(color_riesgo, subset=["Nivel_Riesgo"]).format(na_rep="—"),
            use_container_width=True, height=600,
        )
    else:
        st.dataframe(df.style.format(na_rep="—"), use_container_width=True, height=600)

    csv = datos["df_sim"].to_csv(index=False, encoding="utf-8-sig")
    st.download_button(
        "⬇️ Descargar tabla completa (CSV)",
        data=csv,
        file_name="siac_resultados_completos.csv",
        mime="text/csv",
    )


# ─────────────────────────────────────────────────────────────
# VISTA: METODOLOGÍA ESTADÍSTICA (mapa conceptual interactivo)
# ─────────────────────────────────────────────────────────────

def _resolver_inputs_nrc(p1, p2, desv_p1, desv_p2):
    """
    Replica EXACTAMENTE la lógica de respaldo de
    `data_processor.simular_monte_carlo_nrc` para que los valores mostrados
    en el mapa conceptual coincidan con los que realmente se usaron en la
    simulación real de ese grupo.
    """
    if pd.isna(p1) and pd.isna(p2):
        p1 = p2 = 3.0
    else:
        p1 = p1 if not pd.isna(p1) else (p2 if not pd.isna(p2) else 3.0)
        p2 = p2 if not pd.isna(p2) else p1
    desv_p1 = desv_p1 if not pd.isna(desv_p1) and desv_p1 > 0 else 1.0
    desv_p2 = desv_p2 if not pd.isna(desv_p2) and desv_p2 > 0 else 1.0
    return p1, p2, desv_p1, desv_p2


def _ejemplo_real_para_metodologia(datos: dict) -> dict:
    """
    Selecciona el NRC real con mayor riesgo (menor probabilidad de
    aprobación) para usarlo como ejemplo numérico concreto en cada paso
    del mapa conceptual de metodología. Todos los cálculos reutilizan
    exactamente las mismas fórmulas y constantes que `data_processor.py`.
    """
    df_sim = datos["df_sim"]
    candidatos = df_sim.dropna(subset=["Prob_Aprobacion"])
    fila = candidatos.sort_values("Prob_Aprobacion").iloc[0]

    p1_orig, p2_orig = fila.get("Prom_P1", np.nan), fila.get("Prom_P2", np.nan)
    desv_p1_orig, desv_p2_orig = fila.get("Desv_P1", 1.0), fila.get("Desv_P2", 1.0)
    p1, p2, desv_p1, desv_p2 = _resolver_inputs_nrc(p1_orig, p2_orig, desv_p1_orig, desv_p2_orig)

    media_p3 = 0.5 * p1 + 0.5 * p2
    sigma_p3 = max(desv_p1, desv_p2, 0.5)

    # Regenerar EXACTAMENTE el mismo arreglo simulado (misma semilla=42 y
    # mismos parámetros) solo para poder dibujar el histograma real.
    np.random.seed(42)
    sim_p3 = np.clip(np.random.normal(media_p3, sigma_p3, size=(N_SIMULACIONES,)), NOTA_MIN, NOTA_MAX)
    nota_final_sim = np.clip(
        PESO_PARCIAL_1 * p1 + PESO_PARCIAL_2 * p2 + PESO_FINAL_EST * sim_p3,
        NOTA_MIN, NOTA_MAX,
    )
    counts, edges = np.histogram(nota_final_sim, bins=22, range=(NOTA_MIN, NOTA_MAX))

    kpis = datos["kpis"]

    return {
        "nrc": int(fila["NRC"]),
        "asignatura": str(fila["Asignatura"]),
        "profesor": str(fila.get("Profesor", "—")),
        "matriculados": int(fila["Matriculados_Actuales"]),
        "p1_tiene_dato": not pd.isna(p1_orig),
        "p2_tiene_dato": not pd.isna(p2_orig),
        "p1": p1, "p2": p2, "desv_p1": desv_p1, "desv_p2": desv_p2,
        "media_p3": media_p3, "sigma_p3": sigma_p3,
        "nota_media": float(fila["Nota_Final_Media"]),
        "p5": float(fila["Nota_Final_P5"]), "p95": float(fila["Nota_Final_P95"]),
        "prob_aprob": float(fila["Prob_Aprobacion"]) * 100,
        "prob_reprob": (1 - float(fila["Prob_Aprobacion"])) * 100,
        "ic_lo": float(fila["IC_95_Lower"]) * 100,
        "ic_hi": float(fila["IC_95_Upper"]) * 100,
        "nivel_riesgo": str(fila["Nivel_Riesgo"]),
        "est_aprobaran_grupo": int(fila["Est_Aprobaran"]),
        "est_reprobaran_grupo": int(fila["Est_Reprobaran"]),
        "counts": counts.tolist(),
        "edges": edges.tolist(),
        "kpis": kpis,
    }


def _svg_histograma_montecarlo(counts, edges, umbral=3.0, width=560, height=130) -> str:
    """Genera un histograma SVG real (sin librerías externas) a partir de
    los conteos de la simulación Monte Carlo, coloreando en rojo las barras
    bajo el umbral de aprobación y en verde las que aprueban."""
    max_c = max(counts) if counts and max(counts) > 0 else 1
    n = len(counts)
    bar_w = width / n
    bars = []
    for i, c in enumerate(counts):
        h = (c / max_c) * (height - 6)
        x = i * bar_w
        y = height - h
        mid = (edges[i] + edges[i + 1]) / 2
        color = "#34d399" if mid >= umbral else "#f87171"
        bars.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(bar_w - 1.4, 0.5):.1f}" '
            f'height="{h:.1f}" fill="{color}" opacity="0.88" rx="1.5"/>'
        )
    umbral_x = (umbral / NOTA_MAX) * width
    bars.append(
        f'<line x1="{umbral_x:.1f}" y1="2" x2="{umbral_x:.1f}" y2="{height-2}" '
        f'stroke="#fbbf24" stroke-width="1.6" stroke-dasharray="4,3"/>'
    )
    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" '
        f'xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="none">{"".join(bars)}</svg>'
    )


def _construir_html_mapa_conceptual(pasos: list) -> str:
    """
    Construye el componente HTML/CSS/JS autocontenido (mapa conceptual
    animado e interactivo) para la vista de Metodología. Se renderiza en
    un iframe aislado vía `st.iframe`, por lo que todo su CSS
    y JS vive en este único bloque, sin dependencias externas (sin CDNs).
    """
    steps_payload = [
        {"icono": p["icono"], "titulo": p["titulo"], "resumen": p["resumen"], "detalle": p["detalle"]}
        for p in pasos
    ]
    steps_json = json.dumps(steps_payload, ensure_ascii=False)

    nodos_html = ""
    for i, p in enumerate(pasos):
        nodos_html += (
            f'<div class="meto-node" data-idx="{i}" onclick="selectStep({i})">'
            f'  <div class="meto-node-circle">{p["icono"]}</div>'
            f'  <div class="meto-node-label">PASO {i+1}<br><span>{p["titulo"]}</span></div>'
            f'</div>'
        )
        if i < len(pasos) - 1:
            nodos_html += '<div class="meto-connector"><div class="meto-connector-fill"></div></div>'

    progreso_html = "".join(
        f'<div class="meto-dot" data-idx="{i}" onclick="selectStep({i})"></div>'
        for i in range(len(pasos))
    )

    template = r"""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
  * { box-sizing: border-box; }
  html, body {
    margin: 0; padding: 0;
    background: transparent;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    color: #E2E8F0;
  }
  .meto-wrap {
    background: rgba(15,22,41,0.85);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 20px;
    padding: 1.6rem 1.6rem 1.8rem;
    backdrop-filter: blur(10px);
  }
  /* ── Toolbar ── */
  .meto-toolbar {
    display: flex; align-items: center; justify-content: space-between;
    gap: 1rem; margin-bottom: 1.4rem; flex-wrap: wrap;
  }
  .meto-play-btn {
    background: linear-gradient(135deg, #4338ca, #6366F1);
    color: white; border: none; border-radius: 10px;
    padding: 0.55rem 1.1rem; font-weight: 600; font-size: 0.85rem;
    cursor: pointer; transition: all 0.2s ease;
    font-family: inherit;
  }
  .meto-play-btn:hover { transform: translateY(-1px); box-shadow: 0 6px 18px rgba(99,102,241,0.45); }
  .meto-progress { display: flex; gap: 6px; }
  .meto-dot {
    width: 9px; height: 9px; border-radius: 50%;
    background: rgba(255,255,255,0.15); cursor: pointer;
    transition: all 0.25s ease;
  }
  .meto-dot.active { background: #22D3EE; transform: scale(1.35); box-shadow: 0 0 8px rgba(34,211,238,0.7); }
  .meto-dot.done { background: rgba(34,211,238,0.4); }

  /* ── Flow row ── */
  .meto-flow {
    display: flex; align-items: center; flex-wrap: wrap;
    gap: 0; margin-bottom: 1.6rem; row-gap: 1.6rem;
  }
  .meto-node {
    display: flex; flex-direction: column; align-items: center;
    cursor: pointer; width: 92px; flex-shrink: 0; position: relative;
  }
  .meto-node-circle {
    width: 56px; height: 56px; border-radius: 50%;
    background: rgba(255,255,255,0.05);
    border: 2px solid rgba(255,255,255,0.14);
    display: flex; align-items: center; justify-content: center;
    font-size: 1.5rem; margin-bottom: 0.5rem;
    transition: all 0.3s ease;
  }
  .meto-node-label {
    font-size: 0.62rem; text-align: center; color: #64748B;
    text-transform: uppercase; letter-spacing: 0.04em; font-weight: 700;
    line-height: 1.3;
  }
  .meto-node-label span {
    display: block; font-size: 0.66rem; text-transform: none;
    color: #94A3B8; font-weight: 500; margin-top: 2px; letter-spacing: 0;
  }
  .meto-node.active .meto-node-circle {
    background: linear-gradient(135deg, #6366F1, #8B5CF6);
    border-color: rgba(255,255,255,0.4);
    box-shadow: 0 0 0 5px rgba(99,102,241,0.18), 0 8px 24px rgba(99,102,241,0.5);
    animation: metoPulse 2.2s ease-in-out infinite;
    transform: scale(1.08);
  }
  .meto-node.active .meto-node-label { color: #c7d2fe; }
  .meto-node.active .meto-node-label span { color: #e0e7ff; }
  .meto-node.done .meto-node-circle {
    background: rgba(34,211,238,0.12); border-color: rgba(34,211,238,0.5);
  }
  .meto-node.done .meto-node-label { color: #67e8f9; }
  @keyframes metoPulse {
    0%, 100% { box-shadow: 0 0 0 5px rgba(99,102,241,0.18), 0 8px 24px rgba(99,102,241,0.5); }
    50%      { box-shadow: 0 0 0 9px rgba(99,102,241,0.08), 0 8px 30px rgba(99,102,241,0.65); }
  }

  .meto-connector {
    flex: 1; min-width: 24px; height: 3px;
    background: rgba(255,255,255,0.08);
    margin: 0 4px; position: relative; overflow: hidden;
    border-radius: 2px; align-self: flex-start; margin-top: 27px;
  }
  .meto-connector-fill {
    position: absolute; top: 0; left: 0; height: 100%; width: 200%;
    background: repeating-linear-gradient(
      90deg,
      #6366F1 0px, #6366F1 10px,
      transparent 10px, transparent 20px
    );
    animation: metoFlow 1.1s linear infinite;
    opacity: 0.85;
  }
  @keyframes metoFlow { from { transform: translateX(-20px); } to { transform: translateX(0); } }

  /* ── Detail panel ── */
  .meto-detail {
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.3rem 1.5rem;
    min-height: 250px;
    max-height: 400px;
    overflow-y: auto;
    transition: opacity 0.18s ease;
  }
  .meto-detail.fading { opacity: 0; }
  .meto-detail h3 {
    margin: 0 0 0.7rem; font-size: 1.05rem; color: #f1f5f9;
    font-weight: 700; display: flex; align-items: center; gap: 0.5rem;
  }
  .meto-detail p { font-size: 0.86rem; line-height: 1.6; color: #cbd5e1; margin: 0.5rem 0; }
  .meto-formula {
    background: rgba(99,102,241,0.1);
    border: 1px solid rgba(99,102,241,0.3);
    border-left: 3px solid #6366F1;
    border-radius: 8px;
    padding: 0.65rem 0.9rem;
    font-family: 'SFMono-Regular', Consolas, Menlo, monospace;
    font-size: 0.78rem;
    color: #c7d2fe;
    margin: 0.7rem 0;
    overflow-x: auto;
  }
  .meto-ejemplo-tag {
    font-size: 0.78rem; color: #94A3B8; margin-top: 1rem; margin-bottom: 0.4rem;
  }
  .meto-stats { display: flex; gap: 0.7rem; flex-wrap: wrap; margin-top: 0.5rem; }
  .meto-stat {
    background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.09);
    border-radius: 10px; padding: 0.55rem 0.9rem; min-width: 100px;
    display: flex; flex-direction: column;
  }
  .meto-stat .v { font-size: 1.05rem; font-weight: 800; color: #f1f5f9; font-family: 'Plus Jakarta Sans', sans-serif; }
  .meto-stat .l { font-size: 0.62rem; color: #64748B; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 2px; }
  .meto-stat.alto .v  { color: #fca5a5; }
  .meto-stat.medio .v { color: #fcd34d; }
  .meto-stat.bajo .v  { color: #6ee7b7; }
  .meto-stat.success .v { color: #34d399; }
  .meto-hist { margin: 0.8rem 0; border-radius: 10px; overflow: hidden; background: rgba(0,0,0,0.15); padding: 6px; }

  .meto-nav-buttons { display:flex; justify-content: space-between; margin-top: 1.1rem; }
  .meto-nav-buttons button {
    background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.12);
    color: #cbd5e1; border-radius: 8px; padding: 0.4rem 0.9rem; cursor: pointer;
    font-size: 0.78rem; font-weight: 600; font-family: inherit; transition: all 0.2s ease;
  }
  .meto-nav-buttons button:hover { background: rgba(255,255,255,0.12); color: white; }

  @media (max-width: 640px) {
    .meto-node { width: 70px; }
    .meto-node-circle { width: 46px; height: 46px; font-size: 1.2rem; }
    .meto-connector { margin-top: 22px; }
  }
</style>
</head>
<body>
  <div class="meto-wrap">
    <div class="meto-toolbar">
      <button id="meto-play-btn" class="meto-play-btn" onclick="metoTogglePlay()">▶ Reproducir recorrido</button>
      <div class="meto-progress" id="meto-progress">__PROGRESO__</div>
    </div>
    <div class="meto-flow" id="meto-flow">__NODOS__</div>
    <div class="meto-detail" id="meto-detail"></div>
    <div class="meto-nav-buttons">
      <button onclick="metoStep(-1)">← Anterior</button>
      <button onclick="metoStep(1)">Siguiente →</button>
    </div>
  </div>

  <script id="meto-steps-data" type="application/json">__STEPS_JSON__</script>
  <script>
    const METO_STEPS = JSON.parse(document.getElementById('meto-steps-data').textContent);
    let metoCurrent = 0;
    let metoPlaying = false;
    let metoTimer = null;

    function metoRender(idx) {
      const detail = document.getElementById('meto-detail');
      detail.classList.add('fading');
      setTimeout(() => {
        const s = METO_STEPS[idx];
        detail.innerHTML = '<h3>' + s.icono + ' &nbsp;' + s.titulo + '</h3>' + s.detalle;
        detail.classList.remove('fading');
      }, 140);

      document.querySelectorAll('.meto-node').forEach((el) => {
        const i = parseInt(el.getAttribute('data-idx'));
        el.classList.toggle('active', i === idx);
        el.classList.toggle('done', i < idx);
      });
      document.querySelectorAll('.meto-dot').forEach((el) => {
        const i = parseInt(el.getAttribute('data-idx'));
        el.classList.toggle('active', i === idx);
        el.classList.toggle('done', i < idx);
      });
    }

    function selectStep(idx) {
      metoCurrent = idx;
      metoRender(idx);
      metoPause();
    }

    function metoStep(delta) {
      metoCurrent = (metoCurrent + delta + METO_STEPS.length) % METO_STEPS.length;
      metoRender(metoCurrent);
      metoPause();
    }

    function metoTogglePlay() {
      if (metoPlaying) { metoPause(); } else { metoPlay(); }
    }
    function metoPlay() {
      metoPlaying = true;
      document.getElementById('meto-play-btn').textContent = '⏸ Pausar recorrido';
      metoTimer = setInterval(() => {
        metoCurrent = (metoCurrent + 1) % METO_STEPS.length;
        metoRender(metoCurrent);
      }, 4200);
    }
    function metoPause() {
      metoPlaying = false;
      document.getElementById('meto-play-btn').textContent = '▶ Reproducir recorrido';
      if (metoTimer) { clearInterval(metoTimer); metoTimer = null; }
    }

    metoRender(0);
  </script>
</body>
</html>
"""
    html = (
        template
        .replace("__NODOS__", nodos_html)
        .replace("__PROGRESO__", progreso_html)
        .replace("__STEPS_JSON__", steps_json)
    )
    return html


def _render_iframe_html(html: str, height: int) -> None:
    """
    Renderiza un bloque HTML/CSS/JS autocontenido en un iframe, de forma
    compatible con versiones nuevas y antiguas de Streamlit:
    - Streamlit reciente (>=1.58 aprox.): usa el comando moderno `st.iframe`.
    - Streamlit más antiguo: usa `streamlit.components.v1.html` (más
      ampliamente compatible, aunque esté marcado como obsoleto en versiones
      nuevas).
    """
    if hasattr(st, "iframe"):
        st.iframe(html, height=height)
    else:
        import streamlit.components.v1 as components
        components.html(html, height=height, scrolling=True)


def vista_metodologia(datos: dict) -> None:
    """
    Mapa conceptual animado e interactivo: explica, paso a paso, el proceso
    estadístico completo que usa SIAC (carga de datos → tabla maestra →
    modelo del Parcial 3 → simulación Monte Carlo → clasificación de
    riesgo → intervalo de confianza de Wilson → agregación institucional),
    mostrando en cada paso las fórmulas reales y valores reales calculados
    a partir de los datos del período 202410.
    """
    st.markdown('<div class="section-title">🧮 Metodología Estadística — Cómo funciona SIAC</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="alert-box alert-info" style="margin-bottom:1rem;">
    Recorrido interactivo del proceso estadístico completo. Haz clic en cada paso o presiona
    <b>▶ Reproducir recorrido</b> para una explicación animada y automática. Todos los valores
    mostrados son <b>reales</b>, calculados a partir de los datos del período 202410.
    </div>
    """, unsafe_allow_html=True)

    ej = _ejemplo_real_para_metodologia(datos)
    kpis = ej["kpis"]
    hist_svg = _svg_histograma_montecarlo(ej["counts"], ej["edges"])

    p2_nota = f'{ej["p2"]:.2f}' if ej["p2_tiene_dato"] else f'{ej["p2"]:.2f} (sin dato → se copió de P1)'

    pasos = [
        {
            "icono": "📥", "titulo": "Carga y limpieza de datos",
            "resumen": "Dos fuentes Excel se combinan y limpian.",
            "detalle": f"""
                <p>SIAC parte de <b>dos archivos reales</b>: uno con estadísticas de matrícula y
                retiros por grupo (NRC), y otro con las notas individuales de cada estudiante en
                Parcial 1 y Parcial 2. Se limpian tipos de datos, se recortan notas al rango
                [0,5] y se calculan tasas de retiro por grupo.</p>
                <div class="meto-stats">
                    <div class="meto-stat"><span class="v">{kpis['total_matriculados']:,}</span><span class="l">estudiantes</span></div>
                    <div class="meto-stat"><span class="v">{kpis['total_nrc']}</span><span class="l">grupos (NRC)</span></div>
                    <div class="meto-stat"><span class="v">{kpis['total_asignaturas']}</span><span class="l">asignaturas</span></div>
                </div>
            """,
        },
        {
            "icono": "🗂️", "titulo": "Tabla maestra por NRC",
            "resumen": "Se agregan las notas individuales por grupo.",
            "detalle": f"""
                <p>Las notas individuales se agrupan por NRC para obtener el <b>promedio</b> y la
                <b>desviación estándar</b> de cada corte evaluado:</p>
                <div class="meto-formula">μ_P1 = (1/n) · Σ nota_i &nbsp;&nbsp;|&nbsp;&nbsp; σ_P1 = desviación estándar de esas notas</div>
                <p class="meto-ejemplo-tag">📌 Ejemplo real — NRC {ej['nrc']} ({ej['asignatura']}, {ej['matriculados']} estudiantes)</p>
                <div class="meto-stats">
                    <div class="meto-stat"><span class="v">{ej['p1']:.2f}</span><span class="l">μ Parcial 1</span></div>
                    <div class="meto-stat"><span class="v">±{ej['desv_p1']:.2f}</span><span class="l">σ Parcial 1</span></div>
                    <div class="meto-stat"><span class="v">{p2_nota}</span><span class="l">Parcial 2</span></div>
                </div>
            """,
        },
        {
            "icono": "🎯", "titulo": "Modelo del Parcial 3 (futuro)",
            "resumen": "El corte 3 aún no existe: se modela con incertidumbre.",
            "detalle": f"""
                <p>El Parcial 3 todavía no ha ocurrido. SIAC lo modela como una variable
                aleatoria con <b>regresión a la media</b> de P1 y P2, y una incertidumbre
                conservadora basada en la variabilidad histórica observada:</p>
                <div class="meto-formula">P3_sim ~ Normal(μ = 0.5·P1 + 0.5·P2,&nbsp; σ = max(σ_P1, σ_P2, 0.5)) &nbsp;recortado a [0, 5]</div>
                <p class="meto-ejemplo-tag">📌 Para el NRC {ej['nrc']}:</p>
                <div class="meto-stats">
                    <div class="meto-stat"><span class="v">{ej['media_p3']:.2f}</span><span class="l">μ estimada P3</span></div>
                    <div class="meto-stat"><span class="v">±{ej['sigma_p3']:.2f}</span><span class="l">σ estimada P3</span></div>
                </div>
            """,
        },
        {
            "icono": "🎲", "titulo": "Simulación Monte Carlo (10.000 iteraciones)",
            "resumen": "Se generan 10.000 escenarios posibles de nota final.",
            "detalle": f"""
                <p>Se generan <b>10.000 valores posibles</b> de P3_sim y, para cada uno, se
                calcula la nota final del estudiante promedio del grupo:</p>
                <div class="meto-formula">Nota_Final = 0.35·P1 + 0.35·P2 + 0.30·P3_sim</div>
                <p class="meto-ejemplo-tag">📊 Histograma real de las 10.000 simulaciones — NRC {ej['nrc']}
                (línea punteada = nota mínima de aprobación, 3.0)</p>
                <div class="meto-hist">{hist_svg}</div>
                <div class="meto-stats">
                    <div class="meto-stat"><span class="v">{ej['nota_media']:.2f}</span><span class="l">nota media simulada</span></div>
                    <div class="meto-stat"><span class="v">{ej['p5']:.2f} – {ej['p95']:.2f}</span><span class="l">rango P5–P95</span></div>
                </div>
            """,
        },
        {
            "icono": "🚦", "titulo": "Clasificación de riesgo por grupo",
            "resumen": "Cada grupo se etiqueta ALTO / MEDIO / BAJO.",
            "detalle": f"""
                <p>La probabilidad de reprobación (1 − P(aprobación)) se compara contra dos
                umbrales institucionales:</p>
                <div class="meto-formula">Riesgo = ALTO si P(reprobar) ≥ 40%&nbsp; |&nbsp; MEDIO si 20%–40%&nbsp; |&nbsp; BAJO si &lt; 20%</div>
                <p class="meto-ejemplo-tag">📌 El NRC {ej['nrc']} tiene <b>{ej['prob_reprob']:.1f}%</b> de probabilidad de
                reprobación → riesgo <b>{ej['nivel_riesgo']}</b></p>
                <div class="meto-stats">
                    <div class="meto-stat alto"><span class="v">{kpis['nrc_riesgo_alto']}</span><span class="l">grupos ALTO</span></div>
                    <div class="meto-stat medio"><span class="v">{kpis['nrc_riesgo_medio']}</span><span class="l">grupos MEDIO</span></div>
                    <div class="meto-stat bajo"><span class="v">{kpis['nrc_riesgo_bajo']}</span><span class="l">grupos BAJO</span></div>
                </div>
            """,
        },
        {
            "icono": "📐", "titulo": "Intervalo de confianza de Wilson (95%)",
            "resumen": "Cuantifica la incertidumbre de cada estimación.",
            "detalle": f"""
                <p>Como la probabilidad de aprobación se estima a partir de una muestra
                (10.000 simulaciones), se calcula su <b>intervalo de Wilson</b> al 95% de
                confianza — más preciso que el intervalo normal clásico cuando p está cerca
                de 0 o de 1:</p>
                <div class="meto-formula">IC = ( p + z²/2n ± z·√(p(1−p)/n + z²/4n²) ) / (1 + z²/n) &nbsp;,&nbsp; z = 1.96</div>
                <p class="meto-ejemplo-tag">📌 Para el NRC {ej['nrc']}, P(aprobación) = {ej['prob_aprob']:.1f}%</p>
                <div class="meto-stats">
                    <div class="meto-stat"><span class="v">[{ej['ic_lo']:.1f}%, {ej['ic_hi']:.1f}%]</span><span class="l">IC 95% de aprobación</span></div>
                </div>
            """,
        },
        {
            "icono": "🏛️", "titulo": "Agregación institucional y decisión",
            "resumen": "Los resultados por grupo se consolidan en KPIs.",
            "detalle": f"""
                <p>Finalmente, los resultados de los {kpis['total_nrc']} grupos se agregan en
                indicadores institucionales que alimentan la decisión de apertura de cursos del
                siguiente período:</p>
                <div class="meto-stats">
                    <div class="meto-stat success"><span class="v">{kpis['est_aprobaran']:,}</span><span class="l">aprobarán (estimado)</span></div>
                    <div class="meto-stat alto"><span class="v">{kpis['est_reprobaran']:,}</span><span class="l">reprobarán (estimado)</span></div>
                    <div class="meto-stat"><span class="v">{kpis['docentes_requeridos']}</span><span class="l">docentes activos</span></div>
                    <div class="meto-stat"><span class="v">{kpis['prob_aprobacion_global_pct']}%</span><span class="l">prob. aprobación global</span></div>
                </div>
            """,
        },
    ]

    html = _construir_html_mapa_conceptual(pasos)
    _render_iframe_html(html, height=820)


# ─────────────────────────────────────────────────────────────
# VISTA: ASISTENTE IA (CHATBOT) — vive en un diálogo flotante
# ─────────────────────────────────────────────────────────────
_MODELOS_POR_PROVEEDOR = {
    "groq":   {"env_var": "GROQ_MODEL",   "default": "openai/gpt-oss-120b"},
    "openai": {"env_var": "OPENAI_MODEL", "default": "gpt-4o-mini"},
    "gemini": {"env_var": "GEMINI_MODEL", "default": "gemini-1.5-flash"},
}

_ENV_VARS_POR_PROVEEDOR = {
    "groq":   "GROQ_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
}


def _autoload_chat_credentials() -> None:
    """
    Carga automáticamente la API key desde el archivo .env la primera vez
    que se abre el chat en la sesión. Prioridad: Groq → OpenAI → Gemini.

    Ya no hay panel de configuración en la interfaz: la clave se gestiona
    exclusivamente vía variables de entorno (.env), nunca se pide al usuario.
    """
    if st.session_state.get("_chat_env_loaded"):
        return
    st.session_state["_chat_env_loaded"] = True

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    for provider, env_var in _ENV_VARS_POR_PROVEEDOR.items():
        env_key = os.getenv(env_var)
        if env_key:
            cfg = _MODELOS_POR_PROVEEDOR[provider]
            st.session_state["api_key"]  = env_key
            st.session_state["provider"] = provider
            st.session_state["model"]    = os.getenv(cfg["env_var"], cfg["default"])
            break


def _render_mensaje_chat(role: str, content: str) -> None:
    if role == "user":
        with st.chat_message("user"):
            st.markdown(content)
    else:
        with st.chat_message("assistant", avatar="🎓"):
            st.markdown(content)


def vista_chatbot(datos) -> None:
    """
    Cuerpo del SIAC Assistant, renderizado dentro de un `st.dialog`.

    IMPORTANTE — por qué nunca llamamos a `st.rerun()` aquí:
    Un `st.rerun()` normal (scope="app") cierra el diálogo, porque el
    diálogo solo se vuelve a abrir si el botón flotante que lo dispara
    fue presionado en ESE mismo rerun. Por eso, toda mutación de estado
    (limpiar historial, fijar la pregunta a procesar) ocurre ANTES de
    pintar el historial en esta misma pasada del script, en vez de mutar
    estado y des­pués forzar un rerun.
    """
    _autoload_chat_credentials()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # ── Input del usuario (el valor ya está disponible en esta pasada) ──
    user_input = st.chat_input("Escribe tu pregunta sobre SIAC...")
    pregunta_a_procesar = user_input

    # ── Barra superior: sugerencias + botón de limpiar ───────────
    # (Mutamos el estado del botón "Limpiar" ANTES de pintar el historial
    # para que el cambio se vea de inmediato, sin necesidad de un rerun.)
    if st.session_state.chat_history:
        col_tip, col_clear = st.columns([6, 1])
        with col_tip:
            st.caption("🤖 Pregúntame sobre los datos reales del período 202410.")
        with col_clear:
            if st.button("🗑️", help="Limpiar conversación", use_container_width=True):
                st.session_state.chat_history = []

    if len(st.session_state.chat_history) < 3 and not pregunta_a_procesar:
        sugerencias = get_suggested_questions(datos)
        st.markdown("**💡 Preguntas sugeridas:**")
        cols_sug = st.columns(2)
        for i, sug in enumerate(sugerencias[:6]):
            with cols_sug[i % 2]:
                if st.button(sug, key=f"sug_{i}", use_container_width=True):
                    pregunta_a_procesar = sug

    # ── Historial de mensajes (caja con scroll) ───────────────────
    try:
        historial_box = st.container(height=420, border=False, autoscroll=True)
    except TypeError:
        # Versiones de Streamlit anteriores a 1.56 no soportan `autoscroll`
        historial_box = st.container(height=420, border=False)
    with historial_box:
        if not st.session_state.chat_history and not pregunta_a_procesar:
            st.markdown("""
            <div style="text-align:center; padding:2rem; color:#64748b;">
              <div style="font-size:2rem; margin-bottom:0.5rem;">🎓</div>
              <div style="font-size:0.9rem;">Hola, soy <b style="color:#c7d2fe;">SIAC Assistant</b>.<br>
              Pregúntame sobre los datos académicos del período 202410.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            for msg in st.session_state.chat_history:
                _render_mensaje_chat(msg["role"], msg["content"])

            if pregunta_a_procesar:
                _render_mensaje_chat("user", pregunta_a_procesar)
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": pregunta_a_procesar,
                })

                api_key_actual  = st.session_state.get("api_key", "")
                provider_actual = st.session_state.get("provider", "groq")
                model_actual    = st.session_state.get("model", "openai/gpt-oss-120b")

                with st.chat_message("assistant", avatar="🎓"):
                    respuesta_completa = st.write_stream(
                        chatbot_chat_stream(
                            pregunta=pregunta_a_procesar,
                            historial=st.session_state.chat_history[:-1],
                            datos=datos,
                            api_key=api_key_actual,
                            provider=provider_actual,
                            model=model_actual,
                        )
                    )

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": respuesta_completa,
                })


def abrir_chat_flotante(datos) -> None:
    """Abre el SIAC Assistant en un diálogo modal, disparado por el botón flotante."""

    @st.dialog("💬 SIAC Assistant", width="large")
    def _dialogo_chat():
        vista_chatbot(datos)

    _dialogo_chat()


def render_boton_flotante_chat(datos) -> None:
    """
    Botón de acción flotante (FAB), fijo en la esquina inferior derecha,
    visible en todas las vistas del dashboard. Al hacer clic abre el
    SIAC Assistant en un diálogo modal.
    """
    st.html("""
    <style>
    .st-key-fab_chat_btn button {
        position: fixed;
        bottom: 2rem;
        right: 2rem;
        width: 64px;
        height: 64px;
        min-width: 64px;
        border-radius: 50%;
        background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%);
        color: white !important;
        border: 1px solid rgba(255,255,255,0.25);
        box-shadow: 0 8px 28px rgba(99,102,241,0.55);
        z-index: 999999;
        padding: 0;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .st-key-fab_chat_btn button:hover {
        transform: scale(1.08) translateY(-2px);
        box-shadow: 0 12px 36px rgba(99,102,241,0.7);
        border-color: rgba(255,255,255,0.4);
    }
    .st-key-fab_chat_btn button p {
        font-size: 1.7rem !important;
        line-height: 1;
    }
    </style>
    """)
    if st.button("💬", key="fab_chat_btn", help="Abrir SIAC Assistant"):
        abrir_chat_flotante(datos)
    """Abre el SIAC Assistant en un diálogo modal, disparado por el botón flotante."""

    @st.dialog("💬 SIAC Assistant", width="large")
    def _dialogo_chat():
        vista_chatbot(datos)

    _dialogo_chat()


def render_boton_flotante_chat(datos) -> None:
    """
    Botón de acción flotante (FAB), fijo en la esquina inferior derecha,
    visible en todas las vistas del dashboard. Al hacer clic abre el
    SIAC Assistant en un diálogo modal.
    """
    st.html("""
    <style>
    .st-key-fab_chat_btn button {
        position: fixed;
        bottom: 2rem;
        right: 2rem;
        width: 64px;
        height: 64px;
        min-width: 64px;
        border-radius: 50%;
        background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%);
        color: white !important;
        border: 1px solid rgba(255,255,255,0.25);
        box-shadow: 0 8px 28px rgba(99,102,241,0.55);
        z-index: 999999;
        padding: 0;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .st-key-fab_chat_btn button:hover {
        transform: scale(1.08) translateY(-2px);
        box-shadow: 0 12px 36px rgba(99,102,241,0.7);
        border-color: rgba(255,255,255,0.4);
    }
    .st-key-fab_chat_btn button p {
        font-size: 1.7rem !important;
        line-height: 1;
    }
    </style>
    """)
    if st.button("💬", key="fab_chat_btn", help="Abrir SIAC Assistant"):
        abrir_chat_flotante(datos)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    # Header animado
    st.markdown("""
    <div class="siac-header">
      <h1>🎓 SIAC — Sistema Inteligente de Apertura de Cursos</h1>
      <p>Dashboard Ejecutivo de Planeación Académica &nbsp;·&nbsp; Período 202410
         &nbsp;·&nbsp; Dpto. Matemáticas y Estadística
         &nbsp;|&nbsp; Simulación Monte Carlo · 10,000 iteraciones por grupo</p>
    </div>
    """, unsafe_allow_html=True)

    # Carga de datos con spinner
    with st.spinner("⏳ Procesando datos y ejecutando simulación Monte Carlo..."):
        try:
            datos = cargar_datos()
        except Exception as e:
            st.error(f"❌ **Error al cargar los datos**: {e}")
            st.info("Verifica que los archivos `data/matricula_retiros.xlsx` y `data/Notas_parciales_abril.xlsx` existan.")
            st.stop()

    # Sidebar y navegación
    pagina, filtro_asig, filtro_riesgo = render_sidebar(datos)

    # Despachar vista
    dispatch = {
        "📊 Resumen Ejecutivo":         lambda: vista_resumen(datos),
        "🚦 Riesgo por Grupos":         lambda: vista_riesgo_grupos(datos, filtro_asig, filtro_riesgo),
        "👨‍🎓 Riesgo Estudiantil":        lambda: vista_riesgo_estudiantil(datos),
        "📈 Predicciones Monte Carlo":   lambda: vista_monte_carlo(datos),
        "🏛️ Vista por Programa":         lambda: vista_programa(datos),
        "📋 Tabla Completa":            lambda: vista_tabla_completa(datos),
        "🧮 Metodología":               lambda: vista_metodologia(datos),
    }

    vista_fn = dispatch.get(pagina)
    if vista_fn:
        vista_fn()
    else:
        st.error("Vista no reconocida. Selecciona una opción del menú lateral.")

    # Botón flotante del SIAC Assistant, visible en cualquier vista
    render_boton_flotante_chat(datos)


if __name__ == "__main__":
    main()
