import streamlit as st
from thesaurus_terms import THESAURUS_TERMS as terms
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy

# Carga modelo de lenguaje para detección de frases nominales
nlp = spacy.load("es_core_news_sm")

# ------------------------------------------------------------
#  App Streamlit: Generador de palabras clave Mejorado
# ------------------------------------------------------------
# • Pega tu resumen en el área de texto.
# • TF-IDF con unigramas y bigramas, stop_words en español.
# • Extracción de frases nominales con spaCy para enriquecer candidatos.
# ------------------------------------------------------------

@st.cache_data(show_spinner=False)
def prepare_vectorizer_and_phrases(terms: list[str]):
    """Entrena vectorizador TF-IDF y extrae n-gramas candidatos del tesauro."""
    # TF-IDF con n-gramas 1 y 2 y stop words en español
    vect = TfidfVectorizer(ngram_range=(1,2), stop_words="spanish")
    matrix = vect.fit_transform(terms)
    return vect, matrix


def extract_phrases(text: str) -> list[str]:
    """Extrae frases nominales del texto con spaCy para mejorar coincidencias."""
    doc = nlp(text)
    phrases = set(chunk.text.lower().strip() for chunk in doc.noun_chunks if len(chunk.text.split()) <= 3)
    return list(phrases)


def suggest_keywords(summary: str, terms: list[str], vect, matrix, k: int = 3) -> list[str]:
    """Combina TF-IDF y frases nominales para sugerir k palabras clave."""
    # Similitud TF-IDF contra vocabulario
    sims = cosine_similarity(vect.transform([summary]), matrix).flatten()
    top_idx = sims.argsort()[-k:][::-1]
    tfidf_candidates = [terms[i] for i in top_idx if sims[i] > 0]

    # Frases nominales intersectadas con tesauro
    phrases = extract_phrases(summary)
    phrase_candidates = [p for p in phrases if p in terms]

    # Combinar y priorizar
    combined = []
    for cand in tfidf_candidates + phrase_candidates:
        if cand not in combined:
            combined.append(cand)
        if len(combined) >= k:
            break
    return combined


def main():
    st.set_page_config(page_title="Generador de Palabras Clave Mejorado")
    st.title("🔑 Sugeridor Avanzado de Palabras Clave")
    st.write(
        "Pega tu resumen y obtén sugerencias basadas en TF-IDF multigrama y extracción de frases nominales."
    )

    # Preparación inicial
    vect, matrix = prepare_vectorizer_and_phrases(terms)

    summary = st.text_area("Tu resumen aquí:", height=200)
    k = st.slider("Número de palabras clave", min_value=1, max_value=10, value=3)

    if st.button("Generar palabras clave"):
        if not summary.strip():
            st.warning("Por favor ingresa un resumen para generar las palabras clave.")
        else:
            kws = suggest_keywords(summary, terms, vect, matrix, k)
            if kws:
                st.markdown("**Palabras clave sugeridas:**")
                for kw in kws:
                    st.write(f"- {kw.capitalize()}")
            else:
                st.info("No se encontraron coincidencias. Intenta reformular.")

if __name__ == "__main__":
    main()


