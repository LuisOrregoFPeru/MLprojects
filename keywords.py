import streamlit as st
import requests
import io
import re

# Intentamos importar PdfReader de pypdf, con fallback a PyPDF2 si no está instalado
try:
    from pypdf import PdfReader
except ImportError:
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        st.error("Ni 'pypdf' ni 'PyPDF2' están instalados. Añade al menos uno en requirements.txt.")
        raise

# Intentamos importar TfidfVectorizer y cosine_similarity de scikit-learn
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    st.error("scikit-learn no está instalado. Añade 'scikit-learn' en requirements.txt.")
    raise

# ------------------------------------------------------------
#  Sugeridor de Palabras Clave basado en el Tesauro UNESCO
# ------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_thesaurus(pdf_source="unesco-thesaurus-es.pdf"):
    """Extrae un conjunto de términos (en minúsculas) del Tesauro PDF."""
    fp = None
    close_needed = False

    if isinstance(pdf_source, str):
        if pdf_source.startswith("http://") or pdf_source.startswith("https://"):
            try:
                response = requests.get(pdf_source)
                response.raise_for_status()
                fp = io.BytesIO(response.content)
            except requests.RequestException as e:
                st.error(f"Error al descargar el PDF desde la URL: {e}")
                return []
        else:
            try:
                fp = open(pdf_source, "rb")
                close_needed = True
            except FileNotFoundError:
                st.error(f"Error: Archivo local no encontrado en '{pdf_source}'")
                return []
    elif hasattr(pdf_source, "read"):
        fp = pdf_source
    else:
        st.error("Tipo de fuente PDF no soportado.")
        return []

    reader = PdfReader(fp)
    terms = set()

    for page in reader.pages:
        text = page.extract_text() or ""
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.isdigit():
                continue
            if re.match(r"^(AR|EN|FR|RU|UF|BT|NT|RT|SN|MT)\s*:", line):
                continue
            term = line.split(":", 1)[0]
            term = re.sub(r"[^A-Za-zÁÉÍÓÚÜÑáéíóúüñ ]", "", term).strip().lower()
            if term:
                terms.add(term)

    if close_needed:
        fp.close()

    return sorted(terms)

@st.cache_data(show_spinner=False)
def prepare_vectorizer(terms):
    vect = TfidfVectorizer(analyzer="word")
    matrix = vect.fit_transform(terms)
    return vect, matrix


def suggest_keywords(summary: str, terms, vect, matrix, k: int = 3):
    vec = vect.transform([summary])
    sims = cosine_similarity(vec, matrix).flatten()
    top = sims.argsort()[-k:][::-1]
    return [terms[i] for i in top if sims[i] > 0]


def main():
    st.set_page_config(page_title="Sugeridor Tesauro UNESCO", page_icon="📚")
    st.title("📚 Propuesta automática de palabras clave (Tesauro UNESCO)")
    st.write("Suba un resumen y obtenga las tres voces más afines del Tesauro UNESCO en español.")

    pdf_up = st.file_uploader("Tesauro PDF (opcional)", type=["pdf"], help="Si no selecciona nada se usará el PDF del repositorio.")
    pdf_source = pdf_up if pdf_up is not None else "unesco-thesaurus-es.pdf"

    with st.spinner("Analizando tesauro ..."):
        terms = load_thesaurus(pdf_source)
        vect, matrix = prepare_vectorizer(terms)

    text = st.text_area("Pegue su resumen aquí:", height=220)
    if st.button("🔑 Proponer palabras clave") and text.strip():
        kw = suggest_keywords(text, terms, vect, matrix)
        if kw:
            st.success("Palabras clave sugeridas:")
            for t in kw:
                st.write(f"• {t.capitalize()}")
        else:
            st.info("No se hallaron coincidencias significativas. Intente reformular el texto.")

    st.caption("© UNESCO – Tesauro original. | App creada con Streamlit.")


if __name__ == "__main__":
    main()
