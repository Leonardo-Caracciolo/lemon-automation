import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, request, jsonify
from rag.build_vector_db import cargar_docs, buscar_contexto
from transformers import pipeline
import joblib
import re

# Inicializar Flask
app = Flask(__name__)

# Cargar modelos y recursos NLP
modelo_email = joblib.load("models/email_classifier.joblib")
modelo_intent = joblib.load("models/intent_classifier.joblib")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
qa_model = pipeline("text-generation", model="gpt2")
docs_rag = cargar_docs()

# 🔁 Endpoint RAG
@app.route("/consulta_help", methods=["POST"])
def consulta_help():
    pregunta = request.json["pregunta"]
    contexto = buscar_contexto(pregunta, docs_rag)
    prompt = f"Pregunta: {pregunta}\nContexto: {contexto}\nRespuesta:"
    respuesta = qa_model(prompt, max_length=150)[0]["generated_text"]
    return jsonify({"respuesta": respuesta})

# ✉️ Endpoint para clasificar email
@app.route("/clasificar_email", methods=["POST"])
def clasificar_email():
    data = request.json
    asunto = data.get("Asunto", "")
    cuerpo = data.get("Cuerpo", "")
    texto = asunto + " " + cuerpo
    categoria = modelo_email.predict([texto])[0]
    resumen = summarizer(cuerpo, max_length=150, min_length=20, do_sample=False)[0]["summary_text"]
    cvu = ""
    if categoria == "Consultas de Banking":
        match = re.search(r'\b\d{22}\b', cuerpo)
        if match:
            cvu = match.group(0)
    return jsonify({
        "categoria": categoria,
        "resumen": resumen,
        "CVU": cvu
    })

# 💬 Endpoint para clasificar intención
@app.route("/clasificar_intencion", methods=["POST"])
def clasificar_intencion():
    pregunta = request.json.get("pregunta", "")
    prediccion = modelo_intent.predict([pregunta])[0]
    return jsonify({
        "intencion": int(prediccion)
    })

# 🚀 Ejecutar servidor Flask
if __name__ == "__main__":
    app.run(debug=True)
