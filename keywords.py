import streamlit as st
from thesaurus_terms_es import THESAURUS_TERMS as terms_es
from thesaurus_terms_en import THESAURUS_TERMS as terms_en
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# ------------------------------------------------------------
#  App Streamlit: Generador de Keywords Multilingüe con Sesgo a Salud
# ------------------------------------------------------------
# • Selecciona idioma (Español/Inglés).
# • Carga el tesauro correspondiente.
# • Prioriza términos de salud y coincidencias exactas de hasta 5-gramas.
# • Completa con TF-IDF multigrama.
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


def suggest_keywords(summary, terms, vect, matrix, health_keys, k=3):
    """Sugiere k palabras clave con prioridad a salud y coincidencias exactas."""
    # 1) Coincidencias exactas de n-gramas
    ngrams = extract_ngrams(summary, max_n=5)
    exact = sorted(
        (g for g in ngrams if g in terms),
        key=lambda x: len(x.split()),
        reverse=True
    )
    if len(exact) >= k:
        return exact[:k]

    # 2) TF-IDF con boost de salud
    sims = cosine_similarity(vect.transform([summary]), matrix).flatten()
    candidates = []
    for i, score in enumerate(sims):
        boost = 0.3 if any(h in terms[i] for h in health_keys) else 0
        candidates.append((terms[i], score + boost))
    candidates.sort(key=lambda x: x[1], reverse=True)

    # 3) Combina exactas + TF-IDF
    combined = exact.copy()
    for term, _ in candidates:
        if term not in combined:
            combined.append(term)
        if len(combined) >= k:
            break
    return combined


def main():
    st.set_page_config(page_title="Generador de Keywords de Salud Multilingüe")
    st.title("🔑 Sugeridor Multilingüe de Palabras Clave")
    st.write("Selecciona idioma, pega el resumen y obtén keywords con sesgo a salud.")

    lang = st.selectbox(
        "Idioma de las palabras clave", ["es", "en"],
        format_func=lambda x: "Español" if x == "es" else "Inglés"
    )
    terms = terms_es if lang == "es" else terms_en
    health_keys = HEALTH_KEYWORDS[lang]

    vect, matrix = prepare_vectorizer(terms)
    summary = st.text_area("Tu resumen aquí:", height=200)
    k = st.slider("Número de palabras clave", 1, 10, 3)

    if st.button("Generar palabras clave"):
        if not summary.strip():
            st.warning("Por favor ingresa un resumen.")
        else:
            kws = suggest_keywords(summary, terms, vect, matrix, health_keys, k)
            if kws:
                st.markdown("**Palabras clave sugeridas:**")
                for kw in kws:
                    st.write(f"- {kw.capitalize()}")
            else:
                st.info("No se encontraron coincidencias. Reformula tu resumen.")

if __name__ == "__main__":
    main()

