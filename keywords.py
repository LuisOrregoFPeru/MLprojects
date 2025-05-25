import streamlit as st
from thesaurus_terms_es import THESAURUS_TERMS as terms_es
from thesaurus_terms_en import THESAURUS_TERMS as terms_en
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# ------------------------------------------------------------
#  App Streamlit: Generador de Keywords Multilingüe con Sesgo a Salud
#  -> Sugerencias consistentes en ES y EN (alineadas por índice)
# ------------------------------------------------------------

HEALTH_KEYWORDS = {
    "es": [
        "salud", "dental", "odont", "clínica", "médico", "paciente", "enfermedad",
        "periodontal", "pulpar", "endo-periodontal", "oncológico", "radiológico",
        "maxilar", "quirúrgico", "farmac", "epidemiol",
    ],
    "en": [
        "health", "dent", "clinic", "medical", "patient", "disease",
        "periodontal", "pulpar", "endodont", "oncologic", "radiologic",
        "maxill", "surg", "pharmac", "epidemiol",
    ],
}

@st.cache_data(show_spinner=False)
def prepare_vectorizer(terms):
    """Entrena y devuelve el vectorizador TF-IDF multigrama (1-2)."""
    vect = TfidfVectorizer(ngram_range=(1,2))
    matrix = vect.fit_transform(terms)
    return vect, matrix


def extract_ngrams(text, max_n=5):
    """Extrae todos los n-gramas desde unigramas hasta max_n-gramas."""
    tokens = [t.lower() for t in re.findall(r"\b\w+\b", text)]
    ngrams = set()
    for n in range(1, max_n+1):
        for i in range(len(tokens)-n+1):
            ngrams.add(" ".join(tokens[i:i+n]))
    return ngrams


def suggest_indices(summary, terms, vect, matrix, health_keys, k=3):
    """Devuelve índices de k términos sugeridos con prioridad a salud y exact matches."""
    # exact n-gram matches
    ngrams = extract_ngrams(summary, max_n=5)
    exact = sorted(
        (i for i, term in enumerate(terms) if term in ngrams),
        key=lambda idx: len(terms[idx].split()),
        reverse=True
    )
    if len(exact) >= k:
        return exact[:k]

    # TF-IDF con boost de salud
    sims = cosine_similarity(vect.transform([summary]), matrix).flatten()
    scored = []
    for i, score in enumerate(sims):
        boost = 0.3 if any(h in terms[i] for h in health_keys) else 0
        scored.append((i, score + boost))
    scored.sort(key=lambda x: x[1], reverse=True)

    combined = exact.copy()
    for idx, _ in scored:
        if idx not in combined:
            combined.append(idx)
        if len(combined) >= k:
            break
    return combined


def main():
    st.set_page_config(page_title="Keywords de Salud Multilingüe")
    st.title("🔑 Sugeridor de Palabras Clave Multilingüe")
    st.write(
        "Selecciona idioma, pega tu resumen y obtén las mismas sugerencias en Español e Inglés."
    )

    lang = st.selectbox(
        "Idioma de las palabras clave", ["es", "en"],
        format_func=lambda x: "Español" if x=="es" else "Inglés"
    )
    terms = terms_es  # vectorizador siempre en ES
    health_keys = HEALTH_KEYWORDS[lang]

    vect, matrix = prepare_vectorizer(terms)
    summary = st.text_area("Tu resumen aquí:", height=200)
    k = st.slider("Número de palabras clave", 1, 10, 3)

    if st.button("Generar palabras clave"):
        if not summary.strip():
            st.warning("Por favor ingresa un resumen.")
        else:
            idxs = suggest_indices(summary, terms, vect, matrix, health_keys, k)
            # Mostrar términos alineados
            st.markdown("**Palabras clave sugeridas:**")
            for i in idxs:
                es = terms_es[i].capitalize()
                en = terms_en[i].capitalize()
                if lang == "es":
                    st.write(f"- {es}  ")
                else:
                    st.write(f"- {en}  ")

if __name__ == "__main__":
    main()


