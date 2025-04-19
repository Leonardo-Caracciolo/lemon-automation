from transformers import pipeline
from rag.build_vector_db import cargar_docs, buscar_contexto

question = input("Pregunta: ")
docs = cargar_docs()
context = buscar_contexto(question, docs)

qa = pipeline("text-generation", model="gpt2")
respuesta = qa(f"Pregunta: {question}\nContexto: {context}\nRespuesta:", max_length=100)

print("✅ Respuesta:\n", respuesta[0]["generated_text"])
