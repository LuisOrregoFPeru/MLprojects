import streamlit as st
from thesaurus_terms import THESAURUS_TERMS as terms
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# ------------------------------------------------------------
#  App Streamlit: Generador de palabras clave Mejorado
# ------------------------------------------------------------
# • TF-IDF con unigramas y bigramas.
# • Extracción de n-gramas del resumen para enriquecer candidatos.
# ------------------------------------------------------------

@st.cache_data(show_spinner=False)
def prepare_vectorizer(terms):
    """Entrena y devuelve el vectorizador TF-IDF y la matriz de términos."""
    # Eliminado stop_words para compatibilidad con sklearn
    vect = TfidfVectorizer(ngram_range=(1,2))
    matrix = vect.fit_transform(terms)
    return vect, matrix


def extract_ngrams(text, max_n=2):
    """Extrae n-gramas (1 a max_n palabras) del texto de entrada."""
    tokens = [t.lower() for t in re.findall(r"\b\w+\b", text)]
    ngrams = set()
    length = len(tokens)
    for n in range(1, max_n+1):
        for i in range(length-n+1):
            gram = " ".join(tokens[i:i+n])
            ngrams.add(gram)
    return list(ngrams)


def suggest_keywords(summary, terms, vect, matrix, k=3):
    """Combina TF-IDF y n-gramas para sugerir k palabras clave."""
    sims = cosine_similarity(vect.transform([summary]), matrix).flatten()
    top_idx = sims.argsort()[-k:][::-1]
    tfidf_candidates = [terms[i] for i in top_idx if sims[i] > 0]

    ngrams = extract_ngrams(summary, max_n=2)
    phrase_candidates = [ng for ng in ngrams if ng in terms]

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
        "Pega tu resumen y obtén sugerencias basadas en TF-IDF multigrama y n-gramas."
    )

    vect, matrix = prepare_vectorizer(terms)
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
                st.info("No se encontraron coincidencias. Intenta reformular el texto.")

if __name__ == "__main__":
    main()

