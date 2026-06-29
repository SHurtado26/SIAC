"""
SIAC — Chatbot Contextual
Motor de IA restringido al dominio del Sistema Inteligente de Apertura de Cursos.

Soporta: Groq (openai/gpt-oss-120b, openai/gpt-oss-20b, qwen/qwen3.6-27b — recomendado,
gratis y muy rápido) | OpenAI (gpt-4o-mini, gpt-4o) | Google Gemini (gemini-1.5-flash, gemini-1.5-pro)
"""

import json
import numpy as np
import pandas as pd
from typing import Optional

# ─────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────

DOMAIN_KEYWORDS = [
    # Entidades del sistema
    "siac", "asignatura", "curso", "grupo", "nrc", "estudiante", "docente",
    "profesor", "programa", "departamento", "matematicas", "estadistica",
    # Métricas académicas
    "nota", "parcial", "p1", "p2", "corte", "aprobacion", "reprobacion",
    "matricula", "retiro", "periodo", "202410", "semestre",
    # Análisis y predicción
    "riesgo", "monte carlo", "simulacion", "prediccion", "probabilidad",
    "promedio", "tendencia", "critico", "alto", "medio", "bajo",
    # Planeación
    "planeacion", "apertura", "grupos", "docentes", "demanda", "capacidad",
    # KPIs
    "kpi", "dashboard", "tasa", "porcentaje", "total", "cuantos", "cuales",
    # Preguntas naturales relacionadas
    "aprueba", "reprueba", "mejora", "empeora", "necesita", "proyecta",
    "intervenir", "intervención", "alertas", "criticos",
]

CONVERSATION_OPENERS = [
    "hola", "buenos días", "buenas tardes", "buenas noches", "buen día",
    "gracias", "muchas gracias", "de nada", "ok", "ayuda", "help",
    "qué puedes", "que puedes", "cómo funciona", "como funciona",
    "qué eres", "que eres", "quién eres", "quien eres",
]

SYSTEM_PROMPT_TEMPLATE = """Eres SIAC Assistant, un asistente de inteligencia artificial especializado exclusivamente en el análisis académico del Sistema Inteligente de Apertura de Cursos (SIAC) del Departamento de Matemáticas y Estadística.

═══════════════════════════════════════════════
TU ÚNICO PROPÓSITO es responder preguntas sobre:
═══════════════════════════════════════════════
• Los grupos (NRCs), asignaturas y profesores del departamento en el período 202410
• El análisis de riesgo académico (ALTO, MEDIO, BAJO, CRÍTICO)
• Las predicciones de la simulación Monte Carlo (10,000 iteraciones por grupo)
• La tasa de aprobación y reprobación por grupo y asignatura
• Los estudiantes en riesgo académico y sus indicadores
• La planeación de apertura de grupos para el siguiente período
• Los KPIs institucionales visibles en el dashboard
• La metodología estadística utilizada (Monte Carlo, regresión a la media, IC 95%)

═══════════════════════════════════════════════
DATOS ACTUALES DEL SISTEMA — PERÍODO 202410:
═══════════════════════════════════════════════
{context}

═══════════════════════════════════════════════
INSTRUCCIONES DE COMPORTAMIENTO:
═══════════════════════════════════════════════
1. Responde SIEMPRE en español (colombiano, formal pero accesible).
2. Sé preciso con los números — usa los datos del sistema, no inventes valores.
3. Usa listas, negritas y estructura cuando la respuesta tenga múltiples puntos.
4. Si te piden algo que NO está en los datos disponibles, dilo claramente y sugiere cómo encontrarlo en el dashboard.
5. Si la pregunta está FUERA del dominio de SIAC (política, entretenimiento, programación general, etc.), responde EXACTAMENTE:
   "Lo siento, solo puedo ayudarte con análisis académicos del sistema SIAC. ¿Tienes alguna pregunta sobre los grupos, estudiantes o predicciones del período 202410? 🎓"
6. Nunca inventes datos que no estén en el contexto proporcionado.
7. Para la metodología Monte Carlo, puedes explicarla detalladamente si se pregunta.
"""


# ─────────────────────────────────────────────────────────────
# CONSTRUCCIÓN DEL CONTEXTO
# ─────────────────────────────────────────────────────────────

def build_context(datos: dict) -> str:
    """
    Genera un resumen JSON estructurado del estado actual del dashboard.
    Este contexto se inyecta en el system prompt del LLM.
    """
    kpis = datos["kpis"]
    df_sim = datos["df_sim"].copy()
    df_asig = datos["df_asig"].copy()
    df_riesgo = datos["df_riesgo"].copy()

    # ── Top grupos en riesgo ALTO ──────────────────────────────
    grupos_alto = (
        df_sim[df_sim["Nivel_Riesgo"] == "ALTO"]
        .sort_values("Prob_Aprobacion")
        [["NRC", "Asignatura", "Prob_Aprobacion", "Matriculados_Actuales", "Profesor"]]
        .head(8)
    )

    # ── Asignaturas más críticas ───────────────────────────────
    asig_criticas = (
        df_asig[df_asig["Nivel_Riesgo_Asig"] == "ALTO"]
        .sort_values("Pct_Aprobacion")
        [["Asignatura", "Pct_Aprobacion", "Grupos", "Total_Matriculados", "Est_Reprobaran"]]
        .head(8)
    )

    # ── Asignaturas con mayor tasa de retiro ──────────────────
    top_retiro = (
        df_sim.groupby("Asignatura")["Tasa_Retiro"]
        .mean()
        .sort_values(ascending=False)
        .head(5)
        .reset_index()
    )
    top_retiro["Tasa_Retiro"] = (top_retiro["Tasa_Retiro"] * 100).round(1)

    # ── Riesgo estudiantil ─────────────────────────────────────
    criticos_est = int((df_riesgo["Riesgo_Academico"] == "CRITICO").sum())
    altos_est    = int((df_riesgo["Riesgo_Academico"] == "ALTO").sum())
    medios_est   = int((df_riesgo["Riesgo_Academico"] == "MEDIO").sum())
    bajos_est    = int((df_riesgo["Riesgo_Academico"] == "BAJO").sum())

    # ── Programa con mayor riesgo ──────────────────────────────
    if "Programa" in df_riesgo.columns:
        prog_critico = (
            df_riesgo[df_riesgo["Riesgo_Academico"].isin(["CRITICO", "ALTO"])]
            .groupby("Programa").size()
            .sort_values(ascending=False)
            .head(3)
            .to_dict()
        )
    else:
        prog_critico = {}

    # ── Construir dict de contexto ─────────────────────────────
    context = {
        "resumen_periodo": {
            "periodo_academico": "202410",
            "departamento": "Matemáticas y Estadística",
        },
        "kpis_globales": {
            "total_grupos_nrc": kpis["total_nrc"],
            "total_asignaturas": kpis["total_asignaturas"],
            "total_estudiantes_activos": kpis["total_matriculados"],
            "grupos_en_riesgo_ALTO": kpis["nrc_riesgo_alto"],
            "grupos_en_riesgo_MEDIO": kpis["nrc_riesgo_medio"],
            "grupos_en_riesgo_BAJO": kpis["nrc_riesgo_bajo"],
            "probabilidad_aprobacion_global_pct": kpis["prob_aprobacion_global_pct"],
            "tasa_retiro_global_pct": kpis["tasa_retiro_pct"],
            "estudiantes_proyectados_a_APROBAR": kpis["est_aprobaran"],
            "estudiantes_proyectados_a_REPROBAR": kpis["est_reprobaran"],
        },
        "riesgo_estudiantil": {
            "casos_CRITICOS_ambos_parciales_bajo_3": criticos_est,
            "riesgo_ALTO": altos_est,
            "riesgo_MEDIO": medios_est,
            "sin_riesgo_BAJO": bajos_est,
            "total_registros_estudiante_asignatura": len(df_riesgo),
            "top_programas_con_mas_riesgo": prog_critico,
        },
        "grupos_riesgo_alto_peores": grupos_alto.to_dict("records") if not grupos_alto.empty else [],
        "asignaturas_mas_criticas": asig_criticas.to_dict("records") if not asig_criticas.empty else [],
        "asignaturas_mayor_tasa_retiro_pct": top_retiro.to_dict("records"),
        "metodologia_monte_carlo": {
            "descripcion": "Simulación estocástica del Parcial 3 (aún no ocurrido).",
            "iteraciones_por_grupo": 10000,
            "formula_nota_final": "Nota_Final = 0.35×P1 + 0.35×P2 + 0.30×P3_simulado",
            "distribucion_P3": "Normal(media=0.5×P1+0.5×P2, sigma=max(Desv_P1,Desv_P2,0.5)) clipeado [0,5]",
            "umbral_aprobacion": 3.0,
            "clasificacion_riesgo_grupo": {
                "ALTO": "P(Reprobación) ≥ 40%",
                "MEDIO": "20% ≤ P(Reprobación) < 40%",
                "BAJO": "P(Reprobación) < 20%",
            },
            "intervalo_confianza": "IC 95% via método de Wilson para proporción",
        },
    }

    return json.dumps(context, ensure_ascii=False, indent=2, default=str)


# ─────────────────────────────────────────────────────────────
# DETECCIÓN DE DOMINIO
# ─────────────────────────────────────────────────────────────

def is_in_domain(pregunta: str) -> bool:
    """
    Verifica si la pregunta pertenece al dominio SIAC.
    Primera línea de defensa antes de llamar al LLM.
    Retorna True si la pregunta parece relevante o si es ambigua (el LLM decidirá).
    """
    pregunta_lower = pregunta.lower().strip()

    # Saludos y openers de conversación → siempre aceptar
    for opener in CONVERSATION_OPENERS:
        if opener in pregunta_lower:
            return True

    # Palabras clave del dominio
    for kw in DOMAIN_KEYWORDS:
        if kw in pregunta_lower:
            return True

    # Preguntas muy cortas (posibles seguimientos conversacionales, ej. "sí", "y eso?")
    # → dejar al LLM decidir. El umbral se mantiene bajo para no dejar pasar
    # preguntas fuera de dominio que casualmente tienen pocas palabras
    # (ej. "dame una receta de cocina" tiene 5 palabras y NO es del dominio SIAC).
    if len(pregunta.split()) <= 2:
        return True

    # Preguntas que empiezan con "cuántos", "cuáles", "qué", "cuál" → probable dominio
    interrogativos = ["cuántos", "cuantos", "cuáles", "cuales", "qué", "que ", "cuál", "cual",
                      "cómo", "como ", "por qué", "por que", "explica", "describe", "muestra"]
    for inter in interrogativos:
        if pregunta_lower.startswith(inter):
            return True

    return False


# ─────────────────────────────────────────────────────────────
# PREGUNTAS SUGERIDAS
# ─────────────────────────────────────────────────────────────

def get_suggested_questions(datos: dict) -> list:
    """
    Genera preguntas sugeridas dinámicas según el estado actual de los datos.
    """
    kpis = datos["kpis"]
    df_sim = datos["df_sim"]
    df_asig = datos["df_asig"]

    # Asignatura con mayor riesgo
    asig_critica = "una asignatura crítica"
    grupos_alto = df_sim[df_sim["Nivel_Riesgo"] == "ALTO"]
    if not grupos_alto.empty:
        asig_critica = grupos_alto["Asignatura"].value_counts().index[0]
        asig_critica = asig_critica[:35] if len(asig_critica) > 35 else asig_critica

    n_alto = kpis["nrc_riesgo_alto"]
    prob_global = kpis["prob_aprobacion_global_pct"]

    return [
        f"¿Cuántos grupos están en riesgo ALTO este período?",
        f"¿Por qué {n_alto} grupos tienen riesgo ALTO?",
        f"¿Cómo funciona la simulación Monte Carlo?",
        f"¿Cuáles son las asignaturas con mayor probabilidad de reprobación?",
        f"¿Qué estudiantes necesitan intervención urgente?",
        f"¿Cuántos grupos debería abrir el departamento el próximo período?",
        f"¿Cuál es la tasa de retiro global y qué significa?",
        f"¿Qué programa académico tiene más estudiantes en riesgo?",
        f"Explícame el intervalo de confianza del 95% en las predicciones",
        f"¿Cómo se calcula la nota final estimada con Monte Carlo?",
    ]


# ─────────────────────────────────────────────────────────────
# FUNCIÓN PRINCIPAL DEL CHAT
# ─────────────────────────────────────────────────────────────

def chat(
    pregunta: str,
    historial: list,
    datos: dict,
    api_key: str,
    provider: str = "groq",
    model: str = "openai/gpt-oss-120b",
) -> str:
    """
    Genera una respuesta contextual del chatbot SIAC.

    Args:
        pregunta  : Mensaje del usuario
        historial : Lista de dicts [{"role": "user"/"assistant", "content": "..."}, ...]
        datos     : Dict con todos los DataFrames del dashboard (df_sim, df_asig, etc.)
        api_key   : Clave de API del proveedor
        provider  : "groq" | "openai" | "gemini"
        model     : Nombre del modelo (ej: "openai/gpt-oss-120b", "gpt-4o-mini", "gemini-1.5-flash")

    Returns:
        str: Respuesta del asistente en Markdown
    """
    # ── Validar API key ────────────────────────────────────────
    if not api_key or not api_key.strip():
        return (
            "⚠️ **API no configurada aún**\n\n"
            "Para activar el asistente, ingresa tu clave de API en el panel de configuración de arriba.\n\n"
            "**¿Qué proveedor usar?**\n"
            "- **Groq** → [console.groq.com/keys](https://console.groq.com/keys) "
            "(recomendado: `openai/gpt-oss-120b`, gratuito y extremadamente rápido)\n"
            "- **OpenAI** → [platform.openai.com/api-keys](https://platform.openai.com/api-keys) "
            "(recomendado: `gpt-4o-mini`, económico y preciso)\n"
            "- **Google Gemini** → [aistudio.google.com](https://aistudio.google.com/app/apikey) "
            "(recomendado: `gemini-1.5-flash`, gratuito con cuota)\n\n"
            "Una vez configurada, puedo responder cualquier pregunta sobre los datos de SIAC 🎓"
        )

    # ── Chequeo de dominio (defensa rápida) ───────────────────
    if not is_in_domain(pregunta):
        return (
            "Lo siento, solo puedo ayudarte con análisis académicos del sistema SIAC. "
            "¿Tienes alguna pregunta sobre los grupos, estudiantes o predicciones del período 202410? 🎓"
        )

    # ── Construir contexto y system prompt ────────────────────
    messages = _construir_mensajes(pregunta, historial, datos)

    # ── Llamada a la API ───────────────────────────────────────
    try:
        if provider == "groq":
            return _call_groq(messages, api_key, model)
        elif provider == "openai":
            return _call_openai(messages, api_key, model)
        elif provider == "gemini":
            return _call_gemini(messages, api_key, model)
        else:
            return f"❌ Proveedor '{provider}' no reconocido. Usa **groq**, **openai** o **gemini**."

    except ImportError as e:
        pkg = "google-generativeai" if provider == "gemini" else "openai"
        return (
            f"❌ **Paquete no instalado**: `{pkg}`\n\n"
            f"Ejecuta en la terminal:\n```\npip install {pkg}\n```"
        )
    except Exception as e:
        return _handle_api_error(e, provider)


def chat_stream(
    pregunta: str,
    historial: list,
    datos: dict,
    api_key: str,
    provider: str = "groq",
    model: str = "openai/gpt-oss-120b",
):
    """
    Versión en streaming de `chat()`: un generador que va produciendo la
    respuesta del asistente en fragmentos de texto a medida que el modelo
    los genera (pensado para usarse con `st.write_stream`).

    Si la pregunta no requiere llamar al LLM (sin API key, fuera de dominio),
    el generador produce el mensaje completo de una sola vez.

    Args:
        Igual que `chat()`.

    Yields:
        str: fragmentos de texto de la respuesta, en orden.
    """
    # ── Validar API key ────────────────────────────────────────
    if not api_key or not api_key.strip():
        yield (
            "⚠️ **API no configurada aún**\n\n"
            "Para activar el asistente, ingresa tu clave de API en el panel de configuración de arriba.\n\n"
            "**¿Qué proveedor usar?**\n"
            "- **Groq** → [console.groq.com/keys](https://console.groq.com/keys) "
            "(recomendado: `openai/gpt-oss-120b`, gratuito y extremadamente rápido)\n"
            "- **OpenAI** → [platform.openai.com/api-keys](https://platform.openai.com/api-keys) "
            "(recomendado: `gpt-4o-mini`, económico y preciso)\n"
            "- **Google Gemini** → [aistudio.google.com](https://aistudio.google.com/app/apikey) "
            "(recomendado: `gemini-1.5-flash`, gratuito con cuota)\n\n"
            "Una vez configurada, puedo responder cualquier pregunta sobre los datos de SIAC 🎓"
        )
        return

    # ── Chequeo de dominio (defensa rápida) ───────────────────
    if not is_in_domain(pregunta):
        yield (
            "Lo siento, solo puedo ayudarte con análisis académicos del sistema SIAC. "
            "¿Tienes alguna pregunta sobre los grupos, estudiantes o predicciones del período 202410? 🎓"
        )
        return

    messages = _construir_mensajes(pregunta, historial, datos)

    try:
        if provider in ("groq", "openai"):
            yield from _stream_openai_like(messages, api_key, model, provider)
        elif provider == "gemini":
            yield from _stream_gemini(messages, api_key, model)
        else:
            yield f"❌ Proveedor '{provider}' no reconocido. Usa **groq**, **openai** o **gemini**."
    except ImportError:
        pkg = "google-generativeai" if provider == "gemini" else "openai"
        yield (
            f"❌ **Paquete no instalado**: `{pkg}`\n\n"
            f"Ejecuta en la terminal:\n```\npip install {pkg}\n```"
        )
    except Exception as e:
        yield _handle_api_error(e, provider)


def _construir_mensajes(pregunta: str, historial: list, datos: dict) -> list:
    """Construye la lista de mensajes (system + historial + pregunta) compartida
    entre `chat()` y `chat_stream()`."""
    try:
        context = build_context(datos)
    except Exception:
        context = "{}"

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context)
    messages = [{"role": "system", "content": system_prompt}]

    # Agregar historial (máx. últimos 10 turnos para no exceder tokens)
    for msg in historial[-10:]:
        if isinstance(msg, dict) and "role" in msg and "content" in msg:
            messages.append({"role": msg["role"], "content": msg["content"]})

    # Agregar pregunta actual
    messages.append({"role": "user", "content": pregunta})
    return messages


# ─────────────────────────────────────────────────────────────
# CLIENTES ESPECÍFICOS POR PROVEEDOR
# ─────────────────────────────────────────────────────────────

def _call_groq(messages: list, api_key: str, model: str) -> str:
    """
    Llama a la API de Groq.

    Groq expone una API compatible con el estándar de OpenAI (mismo esquema de
    request/response), por lo que reutilizamos el SDK `openai` apuntando a su
    endpoint. Esto evita una dependencia adicional en requirements.txt.
    """
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
    response = client.chat.completions.create(
        model=model or "openai/gpt-oss-120b",
        messages=messages,
        max_tokens=1500,
        temperature=0.25,  # Más determinista para datos académicos
    )
    return response.choices[0].message.content or "Sin respuesta del modelo."


def _call_openai(messages: list, api_key: str, model: str) -> str:
    """Llama a la API de OpenAI."""
    from openai import OpenAI, AuthenticationError, RateLimitError, NotFoundError

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model or "gpt-4o-mini",
        messages=messages,
        max_tokens=1500,
        temperature=0.25,  # Más determinista para datos académicos
    )
    return response.choices[0].message.content or "Sin respuesta del modelo."


def _call_gemini(messages: list, api_key: str, model: str) -> str:
    """Llama a la API de Google Gemini."""
    import google.generativeai as genai

    genai.configure(api_key=api_key)

    # Separar system prompt del historial
    system_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
    user_messages = [m for m in messages if m["role"] != "system"]

    # Construir historial en formato Gemini
    gemini_history = []
    for i, msg in enumerate(user_messages[:-1]):  # Todos excepto el último
        role = "user" if msg["role"] == "user" else "model"
        gemini_history.append({
            "role": role,
            "parts": [{"text": msg["content"]}]
        })

    model_obj = genai.GenerativeModel(
        model_name=model or "gemini-1.5-flash",
        system_instruction=system_msg,
    )

    if gemini_history:
        chat_session = model_obj.start_chat(history=gemini_history)
        response = chat_session.send_message(user_messages[-1]["content"])
    else:
        response = model_obj.generate_content(user_messages[-1]["content"])

    return response.text or "Sin respuesta del modelo."


# ─────────────────────────────────────────────────────────────
# CLIENTES ESPECÍFICOS POR PROVEEDOR — VERSIÓN STREAMING
# ─────────────────────────────────────────────────────────────

def _stream_openai_like(messages: list, api_key: str, model: str, provider: str):
    """
    Generador de streaming para Groq y OpenAI (ambos comparten el mismo
    esquema de API/SDK; solo cambia la base_url).
    """
    from openai import OpenAI

    base_url = "https://api.groq.com/openai/v1" if provider == "groq" else None
    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
    default_model = "openai/gpt-oss-120b" if provider == "groq" else "gpt-4o-mini"

    stream = client.chat.completions.create(
        model=model or default_model,
        messages=messages,
        max_tokens=1500,
        temperature=0.25,
        stream=True,
    )
    produjo_contenido = False
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        if delta:
            produjo_contenido = True
            yield delta
    if not produjo_contenido:
        yield "Sin respuesta del modelo."


def _stream_gemini(messages: list, api_key: str, model: str):
    """Generador de streaming para Google Gemini."""
    import google.generativeai as genai

    genai.configure(api_key=api_key)

    system_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
    user_messages = [m for m in messages if m["role"] != "system"]

    gemini_history = []
    for msg in user_messages[:-1]:
        role = "user" if msg["role"] == "user" else "model"
        gemini_history.append({
            "role": role,
            "parts": [{"text": msg["content"]}]
        })

    model_obj = genai.GenerativeModel(
        model_name=model or "gemini-1.5-flash",
        system_instruction=system_msg,
    )

    if gemini_history:
        chat_session = model_obj.start_chat(history=gemini_history)
        response = chat_session.send_message(user_messages[-1]["content"], stream=True)
    else:
        response = model_obj.generate_content(user_messages[-1]["content"], stream=True)

    produjo_contenido = False
    for chunk in response:
        if chunk.text:
            produjo_contenido = True
            yield chunk.text
    if not produjo_contenido:
        yield "Sin respuesta del modelo."


def _handle_api_error(error: Exception, provider: str) -> str:
    """Convierte errores de API en mensajes amigables."""
    error_str = str(error).lower()

    if any(kw in error_str for kw in ["authentication", "api_key", "401", "invalid_api_key", "incorrect api key"]):
        return (
            "❌ **API Key inválida**\n\n"
            "Verifica que tu clave sea correcta. Asegúrate de copiarla completa "
            "sin espacios al inicio o al final."
        )
    elif any(kw in error_str for kw in ["rate_limit", "429", "quota", "resource_exhausted"]):
        return (
            "⏳ **Límite de solicitudes alcanzado**\n\n"
            "Espera unos segundos e intenta de nuevo. "
            "Si el problema persiste, puede que hayas alcanzado el límite de tu plan."
        )
    elif any(kw in error_str for kw in ["model_not_found", "model", "404", "does not exist"]):
        return (
            "❌ **Modelo no disponible**\n\n"
            "El modelo seleccionado no está disponible en tu cuenta. "
            "Prueba con `openai/gpt-oss-120b` (Groq), `gpt-4o-mini` (OpenAI) o `gemini-1.5-flash` (Gemini)."
        )
    elif any(kw in error_str for kw in ["connection", "timeout", "network"]):
        return (
            "🌐 **Error de conexión**\n\n"
            "No se pudo conectar con el servicio de IA. "
            "Verifica tu conexión a internet e intenta de nuevo."
        )
    else:
        # Error genérico — mostrar primeros 200 chars
        snippet = str(error)[:200]
        return (
            f"❌ **Error inesperado**\n\n"
            f"```\n{snippet}\n```\n\n"
            f"Si el problema persiste, verifica tu configuración o contacta soporte."
        )
