import streamlit as st
from thesaurus_terms import THESAURUS_TERMS as terms
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# ------------------------------------------------------------
#  App Streamlit: Generador de palabras clave Priorizando Coincidencias Exactas
# ------------------------------------------------------------
# • TF-IDF multigrama para ranking botánico.
# • Extracción de n-gramas hasta 5 palabras para capturar términos largos.
# • Prioriza términos extraídos directamente del texto (n-gramas).
# ------------------------------------------------------------

@st.cache_data(show_spinner=False)
def prepare_vectorizer(terms):
    """Entrena y devuelve el vectorizador TF-IDF y la matriz de términos."""
    vect = TfidfVectorizer(ngram_range=(1,2))
    matrix = vect.fit_transform(terms)
    return vect, matrix


def extract_ngrams(text, max_n=5):
    """Extrae n-gramas (de 1 a max_n palabras) del texto de entrada."""
    tokens = [t.lower() for t in re.findall(r"\b\w+\b", text)]
    ngrams = set()
    length = len(tokens)
    for n in range(1, max_n+1):
        for i in range(length-n+1):
            gram = " ".join(tokens[i:i+n])
            ngrams.add(gram)
    return list(ngrams)


def suggest_keywords(summary, terms, vect, matrix, k=3):
    """Sugerir k palabras clave priorizando coincidencias exactas del tesauro."""
    # Extraer n-gramas y filtrar por vocabulario
    all_ngrams = extract_ngrams(summary, max_n=5)
    phrase_cands = [ng for ng in all_ngrams if ng in terms]

    # Ordenar por longitud descendente (prioriza frases más largas)
    phrase_cands = sorted(set(phrase_cands), key=lambda x: len(x.split()), reverse=True)

    # Si tenemos suficientes, retornamos las top k
    if len(phrase_cands) >= k:
        return phrase_cands[:k]

    # Sino, complementamos con TF-IDF
    sims = cosine_similarity(vect.transform([summary]), matrix).flatten()
    top_idx = sims.argsort()[-k:][::-1]
    tfidf_cands = [terms[i] for i in top_idx if sims[i] > 0]

    combined = phrase_cands.copy()
    for cand in tfidf_cands:
        if cand not in combined:
            combined.append(cand)
        if len(combined) >= k:
            break
    return combined


def main():
    st.set_page_config(page_title="Generador de Palabras Clave Priorizado")
    st.title("🔑 Sugeridor de Palabras Clave Priorizado")
    st.write(
        "Pega tu resumen y obtén sugerencias que priorizan términos exactos del tesauro y luego TF-IDF."
    )

    vect, matrix = prepare_vectorizer(terms)
    summary = st.text_area("Tu resumen aquí:", height=200)
    k = st.slider("Número de palabras clave", min_value=1, max_value=10, value=3)

    if st.button("Generar palabras clave"):
        if not summary.strip():
            st.warning("Por favor ingresa un resumen.")
        else:
            kws = suggest_keywords(summary, terms, vect, matrix, k)
            if kws:
                st.markdown("**Palabras clave sugeridas:**")
                for kw in kws:
                    st.write(f"- {kw.capitalize()}")
            else:
                st.info("No se hallaron coincidencias. Intenta reformular.")

if __name__ == "__main__":
    main()

