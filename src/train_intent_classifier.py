import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import os

# ------------------- CONFIG -------------------
INPUT_PATH = "data/intents.csv"
MODEL_PATH = "models/intent_classifier.joblib"

# ------------------- CARGAR DATOS -------------------
if not os.path.exists(INPUT_PATH):
    print(f"❌ No se encontró el archivo {INPUT_PATH}")
    exit()

df = pd.read_csv(INPUT_PATH)

if 'Pregunta' not in df.columns or 'Intencion' not in df.columns:
    print("❌ El archivo debe tener columnas: 'Pregunta' y 'Intencion'")
    exit()

# ------------------- ENCODING -------------------
X = df["Pregunta"]
y_labels = df["Intencion"]
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_labels)

# Guardar el mapeo de intenciones
label_map_path = "models/intent_labels.txt"
with open(label_map_path, "w", encoding="utf-8") as f:
    for idx, label in enumerate(label_encoder.classes_):
        f.write(f"{idx}: {label}\n")

# ------------------- ENTRENAMIENTO -------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(max_iter=200))
])

pipeline.fit(X_train, y_train)
joblib.dump(pipeline, MODEL_PATH)

print(f"✅ Modelo de intenciones guardado en: {MODEL_PATH}")
print(f"📄 Mapeo de etiquetas guardado en: {label_map_path}")
