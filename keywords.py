import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# ------------------------------------------------------------
#  App Streamlit: Generador de Keywords Bilingüe Consistente
# ------------------------------------------------------------
# • Utiliza lista bilingüe de conceptos alineados por índice.
# • Esquema: CONCEPTS = [{'es':..., 'en':...}, ...]
# • Prioriza términos de salud y coincidencias exactas.
# ------------------------------------------------------------

# Importar lista alineada de conceptos (debes generar este módulo)
from thesaurus_terms_bilingual import CONCEPTS

# Construir vocabularios alineados
terms_es = [c['es'] for c in CONCEPTS]
terms_en = [c['en'] for c in CONCEPTS]

# Prefijos comunes de salud en ambos idiomas
HEALTH_KEYWORDS = [
    "salud", "dental", "odont", "clínica", "médico", "paciente", "enfermedad",
    "periodontal", "pulpar", "endo-periodontal", "oncológico", "radiológico",
    "maxilar", "quirúrgico", "farmac", "epidemiol",
    "health", "dent", "clinic", "medical", "patient", "disease",
    "periodontal", "pulpar", "endodont", "oncologic", "radiologic",
    "maxill", "surg", "pharmac", "epidemiol",
]

@st.cache_data(show_spinner=False)
def prepare_vectorizer(terms):
    vect = TfidfVectorizer(ngram_range=(1,2))
    matrix = vect.fit_transform(terms)
    return vect, matrix


def extract_ngrams(text, max_n=5):
    tokens = [t.lower() for t in re.findall(r"\b\w+\b", text)]
    ngrams = set()
    for n in range(1, max_n+1):
        for i in range(len(tokens)-n+1):
            ngrams.add(" ".join(tokens[i:i+n]))
    return ngrams


def suggest_indices(summary, terms, vect, matrix, k=3):
    # Coincidencias exactas de n-gramas
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
        boost = 0.3 if any(h in terms[i] for h in HEALTH_KEYWORDS) else 0
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
    st.set_page_config(page_title="Generador de Keywords Bilingüe")
    st.title("🔑 Sugeridor de Keywords Consistentes ES/EN")
    st.write(
        "Este generador usa un vocabulario alineado inmutable en ambos idiomas."
    )

    vect, matrix = prepare_vectorizer(terms_es)
    summary = st.text_area("Tu resumen aquí:", height=200)
    k = st.slider("Número de palabras clave", 1, 10, 3)

    if st.button("Generar palabras clave"):
        if not summary.strip():
            st.warning("Por favor ingresa un resumen.")
            return
        idxs = suggest_indices(summary, terms_es, vect, matrix, k)
        st.markdown("**Palabras clave sugeridas:**")
        for idx in idxs:
            es = terms_es[idx].capitalize()
            en = terms_en[idx].capitalize()
            st.write(f"- ES: {es}   |   EN: {en}")

if __name__ == "__main__":
    main()


