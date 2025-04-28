import requests
from bs4 import BeautifulSoup
import time
import os

BASE_URL = "https://help.lemon.me/es/"
OUTPUT_PATH = "rag/help_articles.txt"

def get_articles():
    r = requests.get(BASE_URL)
    soup = BeautifulSoup(r.text, "html.parser")
    
    links = []
    for a in soup.find_all("a", href=True):
        href = a['href']
        if "/es/articles/" in href and href not in links:
            links.append(href)

    print(f"🔗 Se encontraron {len(links)} artículos...")

    contents = []
    for link in links:
        full_url = "https://help.lemon.me" + link
        try:
            res = requests.get(full_url)
            art_soup = BeautifulSoup(res.text, "html.parser")
            texto = art_soup.get_text(separator="\n", strip=True)
            contents.append(texto)
            print(f"✅ Extraído: {full_url}")
            time.sleep(0.5)  # pausa corta para no saturar
        except Exception as e:
            print(f"❌ Error en {full_url}: {e}")
            continue

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n---\n".join(contents))

    print(f"\n📄 Guardado en {OUTPUT_PATH}")

if __name__ == "__main__":
    get_articles()
