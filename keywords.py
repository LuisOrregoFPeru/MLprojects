import streamlit as st
import requests
import io
import re

# Primero intentamos usar pypdf, si no existe, intentamos PyPDF2
try:
    from pypdf import PdfReader
except ImportError:
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        raise ModuleNotFoundError(
            "Ni 'pypdf' ni 'PyPDF2' están instalados. Añade al menos uno en requirements.txt."
        )

# Intentamos importar TfidfVectorizer y cosine_similarity de scikit-learn
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    raise ModuleNotFoundError(
        "scikit-learn no está instalado. Añade 'scikit-learn' en requirements.txt."
    )

@st.cache_data(show_spinner=False)
def load_thesaurus(pdf_source: str = "unesco-thesaurus-es.pdf") -> list[str]:
    """Extrae y devuelve la lista ordenada de términos del Tesauro UNESCO."""
    # Preparar archivo PDF: ruta local, URL o UploadedFile
    if isinstance(pdf_source, str) and (pdf_source.startswith("http://") or pdf_source.startswith("https://")):
        response = requests.get(pdf_source)
        response.raise_for_status()
        fp = io.BytesIO(response.content)
        close_file = False
    elif hasattr(pdf_source, "read"):
        fp = pdf_source
        close_file = False
    else:
        fp = open(pdf_source, "rb")
        close_file = True

    reader = PdfReader(fp)
    terms = set()

    for page in reader.pages:
        text = page.extract_text() or ""
        for line in text.splitlines():
            line = line.strip()
            if not line or line.isdigit():
                continue
            if re.match(r"^(AR|EN|FR|RU|UF|BT|NT|RT|SN|MT)\s*:", line):
                continue
            term = re.sub(r"[^A-Za-zÁÉÍÓÚÜÑáéíóúüñ ]", "", line.split(":", 1)[0]).strip().lower()
            if term:
                terms.add(term)

    if close_file:
        fp.close()

    return sorted(terms)

@st.cache_data(show_spinner=False)
def prepare_vectorizer(terms: list[str]):
    """Entrena y devuelve el vectorizador TF-IDF y la matriz de términos."""
    vect = TfidfVectorizer()
    matrix = vect.fit_transform(terms)
    return vect, matrix


def suggest_keywords(summary: str, terms: list[str], vect, matrix, k: int = 3) -> list[str]:
    """Calcula y retorna las k voces más similares al resumen dado."""
    sims = cosine_similarity(vect.transform([summary]), matrix).flatten()
    return [terms[i] for i in sims.argsort()[-k:][::-1] if sims[i] > 0]


def main():
    st.set_page_config(page_title="Sugeridor Tesauro UNESCO")
    st.title("🔑 Generador de Palabras Clave con Tesauro UNESCO")

    pdf_up = st.file_uploader("Cargar Tesauro (opcional)", type="pdf")
    source = pdf_up if pdf_up else "unesco-thesaurus-es.pdf"

    with st.spinner("Cargando términos..."):
        terms = load_thesaurus(source)
        vect, matrix = prepare_vectorizer(terms)

    summary = st.text_area("Pega tu resumen aquí:")
    if st.button("Generar palabras clave") and summary.strip():
        kws = suggest_keywords(summary, terms, vect, matrix)
        if kws:
            st.markdown("**Sugerencias:**")
            for kw in kws:
                st.write(f"- {kw.capitalize()}")
        else:
            st.warning("No se encontraron coincidencias. Intenta reformular.")

if __name__ == "__main__":
    main()
