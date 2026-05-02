from flask import Flask, request, jsonify, render_template
import pandas as pd
from openai import OpenAI
import os
from dotenv import load_dotenv

# Cargar variables de entorno desde el archivo .env
load_dotenv()

# Configuramos Flask para que busque el HTML en la carpeta "frontend"
template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'frontend'))
app = Flask(__name__, template_folder=template_dir)

# 🔥 CONFIGURACIÓN GROQ
client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

# Cargar dataset
# Usamos una ruta relativa que funcione tanto localmente como en Vercel
base_path = os.path.dirname(__file__)
csv_path = os.path.join(base_path, "..", "datasetLimpio", "peliculas_total_limpio.csv")
df = pd.read_csv(csv_path)

import json

# ---------------------------
# EXTRAER PREFERENCIAS (USANDO IA)
# ---------------------------
def conversar_o_extraer(texto, historial=[]):
    prompt_sistema = """
    Eres un asistente conversacional recomendador de películas. 
    Tu objetivo es cumplir ESTRICTAMENTE con estas reglas:
    1. ANALIZA EL HISTORIAL: Revisa los mensajes anteriores para entender el contexto de la conversación.
    2. SOLO SALUDOS: Si el usuario SÓLO saluda, preséntate brevemente.
    3. SIN CRITERIOS: Si pide recomendaciones sin detalles, pregunta por género, duración, tono, año o calificación.
    4. CON CRITERIOS: Si da preferencias, devuelve ÚNICAMENTE un JSON con: "genero" (en inglés), "duracion" (corta, media, larga), "tono" (serio, emocionante, neutral, divertido, oscuro), "anio", "calificacion".
    5. PREGUNTAS SOBRE RESULTADOS: Si el usuario hace una pregunta sobre películas ya recomendadas (ej: "¿cuál es la mejor de esas?"), responde directamente como charla usando el contexto del historial.
    6. Solo responde con texto, por favor no des JSON ni csv para el usuario, a menos que estés extrayendo preferencias
    7. No agregues explicaciones muy largas sobre las peliculas, podrias darle una lista de peliculas y el usuario decide sobre cual quiere profundizar.
    8. Al preguntar sobre una pelicula en especial, habla solo al respecto de esa pelicula, no te extiendas a otras a menos que el usuario lo pida.
    9. Manten la estructura de la conversacion, no es necesario que muestres el JSON al usuario, solo úsalo para filtrar y luego responde de forma natural con las recomendaciones o respuestas a preguntas sobre las recomendaciones.
    10. Centrate en solo responder preguntas sobre las peliculas, no hables mucho al respecto de actores, directores o cosas asi a menos que el usuario lo pida explícitamente.
    11. Puedes recomendar peliculas por directores o actores si el usuario lo pide, pero no es tu función principal, solo hazlo si el usuario lo solicita explícitamente.
    """

    messages = [{"role": "system", "content": prompt_sistema}]
    # Añadimos historial (limitado a los últimos 6 mensajes para no saturar al modelo)
    for msg in historial[-6:]:
        role = "user" if msg["sender"] == "usuario" else "assistant"
        messages.append({"role": role, "content": msg["texto"]})
    
    messages.append({"role": "user", "content": texto})
    
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        temperature=0.3
    )
    
    contenido = response.choices[0].message.content.strip()
    
    # Intentar parsear como JSON por si extrajo entidades
    try:
        if "{" in contenido and "}" in contenido:
            # extraer el pedazo json
            start = contenido.find('{')
            end = contenido.rfind('}') + 1
            datos_json = json.loads(contenido[start:end])
            return {"tipo": "busqueda", "datos": datos_json}
    except Exception:
        pass
        
    return {"tipo": "charla", "respuesta": contenido}

# ---------------------------
# FILTRAR
# ---------------------------
def filtrar_peliculas(pref):
    resultados = df.copy()

    # Aplicamos filtros de forma "inteligente". Si un filtro deja la lista vacía, lo ignoramos 
    # para asegurar que siempre haya recomendaciones cercanas.

    g = pref.get("genero")
    if g:
        temp = resultados[resultados["genero"].str.contains(str(g), case=False, na=False)]
        if not temp.empty: resultados = temp

    d = pref.get("duracion")
    if d:
        temp = resultados[resultados["duracion"] == d]
        if not temp.empty: resultados = temp

    t = pref.get("tono")
    if t:
        temp = resultados[resultados["tono"] == t]
        if not temp.empty: resultados = temp
        
    c = pref.get("calificacion")
    if c:
        try:
            temp = resultados[resultados["calificacion"] >= float(c)]
            if not temp.empty: resultados = temp
        except:
            pass

    a = pref.get("anio")
    if a:
        try:
            temp = resultados.copy()
            temp["anio_num"] = pd.to_numeric(temp["anio"].astype(str).str.extract(r'(\d+)')[0], errors='coerce')
            temp_filtrado = temp[temp["anio_num"] >= int(a)]
            if not temp_filtrado.empty:
                resultados = temp_filtrado.drop(columns=["anio_num"])
        except Exception as e:
            print("Error filtrando año:", e)
            pass

    return resultados.head(5)

# ---------------------------
# RESPUESTA CON GROQ
# ---------------------------
def generar_respuesta(usuario, peliculas):

    lista = "\n".join([
        f"{row['titulo']} ({row['anio']}) - Rating {row['calificacion']}"
        for _, row in peliculas.iterrows()
    ])

    prompt = f"""
    Usuario quiere: {usuario}

    Películas encontradas (puede que no cumplan el 100% de criterios estrictos, pero son sugerencias afines):
    {lista}

    Explica de forma natural y conversacional por qué son buenas opciones basadas en lo que pidió. Si no cumplen un año exacto o algo similar, menciónalo sutilmente pero enfatiza lo que sí cumple.
    """

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",  # Actualizado a un modelo vigente en Groq
        messages=[
            {"role": "system", "content": "Eres un recomendador experto"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )

    return response.choices[0].message.content

# ---------------------------
# ENDPOINTS
# ---------------------------
@app.route("/", methods=["GET"])
def index():
    return render_template("pantallaInicio.html")

@app.route("/chat", methods=["POST"])
def chat():
    datos = request.json
    mensaje = datos.get("mensaje")
    historial = datos.get("historial", [])

    analisis = conversar_o_extraer(mensaje, historial)

    if analisis["tipo"] == "charla":
        return jsonify({"respuesta": analisis["respuesta"]})
    
    preferencias = analisis["datos"]
    peliculas = filtrar_peliculas(preferencias)

    if peliculas.empty:
        return jsonify({"respuesta": "No encontré películas con esos criterios. ¿Por qué no pruebas con otros detalles o ampliar mas en estos?"})

    respuesta = generar_respuesta(mensaje, peliculas)

    return jsonify({"respuesta": respuesta})


if __name__ == "__main__":
    app.run(debug=True)