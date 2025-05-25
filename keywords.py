import streamlit as st
from thesaurus_terms import THESAURUS_TERMS as terms
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# ------------------------------------------------------------
#  App Streamlit: Generador de palabras clave con sesgo a salud
# ------------------------------------------------------------
# • Prioriza términos relacionados con salud (médico, dental, clínico).
# • N-gramas hasta 5 palabras para coincidencias exactas.
# • TF-IDF multigrama con refuerzo de puntuación para términos de salud.
# ------------------------------------------------------------

# Lista de prefijos comunes al vocabulario de salud para sesgo
HEALTH_KEYWORDS = [
    "salud", "dental", "odont", "clínica", "médico", "paciente", "enfermedad",
    "periodontal", "pulpar", "endo-periodontal", "oncológico", "radiológico",
    "maxilar", "quirúrgico", "farmac", "epidemiol",
]

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


def is_health_term(term):
    """Verifica si un término pertenece al dominio de salud mediante prefijos."""
    return any(h in term for h in HEALTH_KEYWORDS)


def suggest_keywords(summary, terms, vect, matrix, k=3):
    """Sugerir k palabras clave con prioridad a salud y coincidencias exactas."""
    # Exact matches de n-gramas
    all_ngrams = extract_ngrams(summary, max_n=5)
    phrase_cands = [ng for ng in all_ngrams if ng in terms]
    phrase_cands = sorted(set(phrase_cands), key=lambda x: len(x.split()), reverse=True)

    # Si suficientes exactas, devolvemos
    if len(phrase_cands) >= k:
        return phrase_cands[:k]

    # TF-IDF con refuerzo de salud
    sims = cosine_similarity(vect.transform([summary]), matrix).flatten()
    boosted = []
    for i, score in enumerate(sims):
        boost = 0.3 if is_health_term(terms[i]) else 0
        boosted.append((terms[i], score + boost))
    # Orden descendente por puntuación
    boosted.sort(key=lambda x: x[1], reverse=True)

    combined = phrase_cands.copy()
    for term, _ in boosted:
        if term not in combined:
            combined.append(term)
        if len(combined) >= k:
            break
    return combined


def main():
    st.set_page_config(page_title="Generador de Keywords de Salud")
    st.title("🔑 Sugeridor de Palabras Clave con Enfoque en Salud")
    st.write(
        "Pega tu resumen y obtén sugerencias priorizando términos de salud del Tesauro UNESCO."
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
