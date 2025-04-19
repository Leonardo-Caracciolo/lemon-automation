import pandas as pd
import re
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
import joblib
from transformers import pipeline
import os

# ------------------- CONFIG -------------------
INPUT_PATH = "data/emails.csv"
OUTPUT_PATH = "data/emails_output.csv"
MODEL_PATH = "models/email_classifier.joblib"
MAX_SUMMARY_WORDS = 150

# ------------------- CATEGORIZADOR -------------------
def entrenar_modelo(df):
    df['texto_completo'] = df['Asunto'] + " " + df['Cuerpo']
    X_train, X_test, y_train, y_test = train_test_split(df['texto_completo'], df['Categoria'], test_size=0.2, random_state=42)

    pipeline_modelo = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', MultinomialNB())
    ])

    pipeline_modelo.fit(X_train, y_train)
    pred = pipeline_modelo.predict(X_test)
    print(f"Accuracy: {metrics.accuracy_score(y_test, pred)}")

    joblib.dump(pipeline_modelo, MODEL_PATH)
    print(f"✅ Modelo guardado en {MODEL_PATH}")
    return pipeline_modelo

# ------------------- CVU EXTRACTOR -------------------
def extraer_cvu(texto):
    match = re.search(r'\b\d{22}\b', texto)
    return match.group(0) if match else ""

# ------------------- SUMMARIZER -------------------
def generar_resumen(texto):
    try:
        resumen = summarizer(texto, max_length=150, min_length=20, do_sample=False)
        return resumen[0]['summary_text']
    except:
        return texto[:150]

# ------------------- MAIN -------------------
if __name__ == "__main__":
    if not os.path.exists(INPUT_PATH):
        print(f"❌ No se encontró {INPUT_PATH}")
        exit()

    df = pd.read_csv(INPUT_PATH)

    if 'Categoria' not in df.columns:
        print("❌ El dataset necesita una columna 'Categoria' para entrenar")
        exit()

    print("🔄 Cargando modelo de resumen (esto tarda un poco)...")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    print("⚙️ Entrenando modelo de clasificación...")
    modelo = entrenar_modelo(df)

    print("📥 Clasificando y generando resumen...")
    resultados = []

    for _, fila in df.iterrows():
        texto_completo = fila['Asunto'] + " " + fila['Cuerpo']
        prediccion = modelo.predict([texto_completo])[0]
        resumen = generar_resumen(fila['Cuerpo'])
        cvu = extraer_cvu(fila['Cuerpo']) if prediccion == "Consultas de Banking" else ""
        resultados.append({
            "ID del cliente": fila["ID del cliente"],
            "Asunto": fila["Asunto"],
            "Categoría": prediccion,
            "Resumen": resumen,
            "CVU": cvu
        })

    df_salida = pd.DataFrame(resultados)
    df_salida.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Resultado guardado en {OUTPUT_PATH}")
