# Lemon Automation Challenge

## Cómo correr

1. Crear entorno virtual
2. Instalar dependencias:
   pip install -r requirements.txt

3. Entrenar modelos:
   python src/train_email_classifier.py
   python src/train_intent_classifier.py

4. Construir sistema RAG:
   python rag/scraper.py
   python rag/build_vector_db.py

5. Correr API:
   python api/main.py

## Endpoints

- POST /clasificar_email
  - JSON: { "Asunto": "...", "Cuerpo": "..." }

- POST /clasificar_intencion
  - JSON: { "pregunta": "..." }

- POST /consulta_help
  - JSON: { "pregunta": "..." }

## Decisiones técnicas

- Clasificadores: scikit-learn con Tfidf + NaiveBayes / LogisticRegression
- RAG: ChromaDB + SentenceTransformers + GPT2
- Resumen: facebook/bart-large-cnn
## ✅ Funcionalidades implementadas

### 1. Automatización de correos

Automaticé la lectura y clasificación de correos a partir de un archivo `.csv` que contiene:

- ID del cliente
- Asunto
- Cuerpo del mensaje

Los correos se clasifican en:

- Consultas de Crypto
- Consultas de Banking
- Consultas de Tarjeta
- Otro

Además, para cada correo:
- Se genera un resumen automático (máx. 150 palabras) utilizando `facebook/bart-large-cnn`
- Si pertenece a “Consultas de Banking”, se extrae automáticamente el CVU desde el cuerpo
- Se exporta un archivo `emails_output.csv` con las columnas requeridas:
  - ID del cliente, Asunto, Categoría, Resumen, CVU

---

### 2. Clasificación de intenciones

Entrené un modelo simple usando `TfidfVectorizer` y `LogisticRegression` sobre un dataset simulado de 100-200 ejemplos. Las intenciones clasificadas son:

- Retiros Crypto
- Retiros Fiat
- Denuncia de Tarjeta Perdida
- Desconocimiento de Transacciones
- Lemon Earn

Este modelo está expuesto mediante un endpoint REST `/clasificar_intencion`.

---

### 3. Sistema RAG

Implementé un sistema de recuperación de información donde:

- Se almacena un archivo `help_articles.txt` con artículos representativos del centro de ayuda
- Se utiliza `difflib` para encontrar el artículo más relevante para una pregunta
- Se genera una respuesta automática usando el modelo `gpt2` (HuggingFace `transformers`)
- Todo esto está disponible desde el endpoint `/consulta_help`

---

### 4. API REST

Construí una API con Flask que expone los siguientes endpoints:

- `POST /clasificar_email`: clasifica un correo, resume y extrae CVU
- `POST /clasificar_intencion`: devuelve el tipo de intención detectada
- `POST /consulta_help`: responde preguntas sobre el centro de ayuda utilizando RAG + generación de texto

---

## 🧠 Tecnologías utilizadas

- Python 3.12
- Flask
- Scikit-learn
- Transformers (Hugging Face)
- BeautifulSoup4
- Difflib
- Joblib


lemon_automation/
├── api/
│   └── main.py                        # API REST con Flask
├── data/
│   ├── emails.csv                     # Archivo de entrada con correos
│   ├── intents.csv                    # Dataset de entrenamiento de intenciones
│   └── emails_output.csv              # Resultado procesado
├── models/
│   ├── email_classifier.joblib        # Modelo entrenado para correos
│   └── intent_classifier.joblib       # Modelo entrenado para intenciones
├── rag/
│   ├── help_articles.txt              # Base de conocimiento usada en el RAG
│   ├── scraper.py                     # (opcional) Scraper para construir help_articles
│   └── build_vector_db.py             # Funciones de carga y búsqueda del contexto
├── src/
│   ├── train_email_classifier.py      # Entrenamiento del modelo de correos
│   └── train_intent_classifier.py     # Entrenamiento del modelo de intenciones
├── requirements.txt                   # Librerías necesarias
└── README.md  

