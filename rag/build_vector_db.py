import difflib
import os

def cargar_docs():
    with open("rag/help_articles.txt", "r", encoding="utf-8") as f:
        return f.read().split("\n---\n")

def buscar_contexto(pregunta, docs):
    return difflib.get_close_matches(pregunta, docs, n=1, cutoff=0.1)[0]

if __name__ == "__main__":
    docs = cargar_docs()
    pregunta = input("Pregunta del usuario: ")
    contexto = buscar_contexto(pregunta, docs)
    print("\n🔎 Contexto más relevante:\n")
    print(contexto)
