import streamlit as st
from thesaurus_terms_es import THESAURUS_TERMS as terms_es
from thesaurus_terms_en import THESAURUS_TERMS as terms_en
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# ------------------------------------------------------------
#  App Streamlit: Generador de palabras clave multilingüe con sesgo a salud
# ------------------------------------------------------------
# • El usuario selecciona idioma: Español o Inglés.
# • Carga vocabulario correspondiente desde módulos thesaurus_terms_es/en.
# • TF-IDF multigrama y extracción de n-gramas priorizados.
# • Sesgo a salud configurable por idioma.
# ------------------------------------------------------------

# Prefijos de salud por idioma
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
    """Entrena TF-IDF multigrama (1-2) sobre vocabulario dado."""
    vect = TfidfVectorizer(ngram_range=(1,2))
    matrix = vect.fit_transform(terms)
    return vect, matrix


def extract_ngrams(text, max_n=5):
    """Extrae n-gramas (1 a max_n palabras) del texto"""
    tokens = [t.lower() for t in re.findall(r"\b\w+\b", text)]
    ngrams = set()
    length = len(tokens)
    for n in range(1, max_n+1):
        for i in range(length-n+1):
            gram = " ".join(tokens[i:i+n])
            ngrams.add(gram)
    return list(ngrams)


def suggest_keywords(summary, terms, vect, matrix, health_keys, k=3):
    """Sugiere k keywords, priorizando salud y coincidencias exactas."""
    # Exact matches de n-gramas
    all_ngrams = extract_ngrams(summary, max_n=5)
    phrase_cands = [ng for ng in all_ngrams if ng in terms]
    phrase_cands = sorted(set(phrase_cands), key=lambda x: len(x.split()), reverse=True)
    if len(phrase_cands) >= k:
        return phrase_cands[:k]

    # TF-IDF con boost de salud
    sims = cosine_similarity(vect.transform([summary]), matrix).flatten()
    boosted = []
    for i, score in enumerate(sims):
        boost = 0.3 if any(h in terms[i] for h in health_keys) else 0
        boosted.append((terms[i], score + boost))
    boosted.sort(key=lambda x: x[1], reverse=True)

    combined = phrase_cands.copy()
    for term, _ in boosted:
        if term not in combined:
            combined.append(term)
        if len(combined) >= k:
            break
    return combined


def main():
    # Configuración y UI
    st.set_page_config(page_title="Sugeridor Multilingüe de Keywords")
    st.title("🔑 Generador de Palabras Clave (ES/EN)")
    st.write("Selecciona idioma, pega tu resumen y obtén keywords con sesgo a salud.")

    lang = st.selectbox("Idioma de las keywords", options=["es", "en"], format_func=lambda x: "Español" if x=="es" else "Inglés")
    terms = terms_es if lang == "es" else terms_en
    health_keys = HEALTH_KEYWORDS[lang]

    vect, matrix = prepare_vectorizer(terms)
    summary = st.text_area("Tu resumen aquí:", height=200)
    k = st.slider("Número de palabras clave", min_value=1, max_value=10, value=3)

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

