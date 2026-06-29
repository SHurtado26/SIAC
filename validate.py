import sys, os
sys.path.insert(0, 'src')

from data_processor import procesar_todo
datos = procesar_todo(
    path_matriculas='data/matricula_retiros.xlsx',
    path_notas='data/Notas_parciales_abril.xlsx'
)
kpis = datos['kpis']
print("NRCs:", kpis["total_nrc"])
print("Asignaturas:", kpis["total_asignaturas"])
print("Matriculados:", kpis["total_matriculados"])
print("Riesgo ALTO:", kpis["nrc_riesgo_alto"])
print("Prob. aprobacion:", kpis["prob_aprobacion_global_pct"], "%")
print("data_processor: OK")

from chatbot import build_context, get_suggested_questions, is_in_domain
ctx = build_context(datos)
sugs = get_suggested_questions(datos)
print("Contexto:", len(ctx), "chars")
print("Sugerencias:", len(sugs), "preguntas")
print("domain-check cuantos grupos:", is_in_domain("cuantos grupos en riesgo"))
print("domain-check receta cocina:", is_in_domain("dame una receta de cocina"))
print("chatbot: OK")
print()
print("=== VALIDACION COMPLETA: TODO OK ===")
