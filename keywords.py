import streamlit as st
import PyPDF2
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------------------------------------
#  Sugeridor de Palabras Clave basado en el Tesauro UNESCO
#  ---------------------------------------------------------
#  • Ingrese un resumen en español (o cualquier texto breve).
#  • El sistema propone las 3 voces más próximas del Tesauro UNESCO.
#  • Por defecto carga el PDF "unesco-thesaurus-es.pdf" incluido en
#    el repositorio, pero permite sustituirlo por otro.
# ------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_thesaurus(pdf_file="unesco-thesaurus-es.pdf"):
    """Extrae un conjunto de términos (en minúsculas) del Tesauro PDF."""
    if isinstance(pdf_file, str):
        fp = open(pdf_file, "rb")
        close_needed = True
    else:  # UploadedFile de Streamlit
        fp = pdf_file
        close_needed = False

    reader = PyPDF2.PdfReader(fp)
    terms = set()

    for page in reader.pages:
        text = page.extract_text() or ""
        for raw in text.splitlines():
            line = raw.strip()
            # Filtra numeración u otras líneas vacías
            if not line or line.isdigit():
                continue
            # Suprime prefijos de relaciones (AR:, BT:, UF:, etc.)
            if re.match(r"^(AR|EN|FR|RU|UF|BT|NT|RT|SN|MT)\s*:", line):
                continue
            # El término acaba ante un ‘:’ si existe
            term = line.split(":", 1)[0]
            # Sólo letras y espacios
            term = re.sub(r"[^A-Za-zÁÉÍÓÚÜÑáéíóúüñ ]", "", term).strip().lower()
            if term:
                terms.add(term)

    if close_needed:
        fp.close()

    return sorted(terms)

@st.cache_data(show_spinner=False)
def prepare_vectorizer(terms):
    """Entrena un TF‑IDF simple sobre la lista de descriptores."""
    vect = TfidfVectorizer(analyzer="word")
    matrix = vect.fit_transform(terms)
    return vect, matrix


def suggest_keywords(summary: str, terms, vect, matrix, k: int = 3):
    """Devuelve las k voces del Tesauro más parecidas al resumen."""
    vec = vect.transform([summary])
    sims = cosine_similarity(vec, matrix).flatten()
    top = sims.argsort()[-k:][::-1]
    return [terms[i] for i in top if sims[i] > 0]


def main():
    st.set_page_config(page_title="Sugeridor Tesauro UNESCO", page_icon="📚")
    st.title("📚 Propuesta automática de palabras clave (Tesauro UNESCO)")
    st.write("Suba un resumen y obtenga las tres voces más afines del Tesauro UNESCO en español.")

    # Carga (opcional) de un PDF distinto
    pdf_up = st.file_uploader("Tesauro PDF (opcional)", type=["pdf"], help="Si no selecciona nada se usará el PDF del repositorio.")
    pdf_source = pdf_up if pdf_up is not None else "unesco-thesaurus-es.pdf"

    # Prepara el Tesauro y el vectorizador
    with st.spinner("Analizando tesauro ..."):
        terms = load_thesaurus(pdf_source)
        vect, matrix = prepare_vectorizer(terms)

    text = st.text_area("Pegue su resumen aquí:", height=220)
    if st.button("🔑 Proponer palabras clave") and text.strip():
        kw = suggest_keywords(text, terms, vect, matrix)
        if kw:
            st.success("Palabras clave sugeridas:")
            for t in kw:
                st.write(f"• {t.capitalize()}")
        else:
            st.info("No se hallaron coincidencias significativas. Intente reformular el texto.")

    st.caption("© UNESCO – Tesauro original. | App creada con Streamlit.")


if __name__ == "__main__":
    main()
