import streamlit as st
from thesaurus_terms_es import THESAURUS_TERMS as terms_es
from thesaurus_terms_en import THESAURUS_TERMS as terms_en
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# ------------------------------------------------------------
#  App Streamlit: Generador de Keywords Bilingüe Consistente
# ------------------------------------------------------------
# • Sugerencias idénticas para ES y EN.
# • Mostrar ambos idiomas juntos.
# • Sesgo a salud y coincidencias exactas.
# ------------------------------------------------------------

HEALTH_KEYWORDS = [
    # Prefijos comunes en ambas lenguas para mantener boost en índices alineados
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
    ngrams = extract_ngrams(summary, max_n=5)
    exact = sorted(
        (i for i, term in enumerate(terms) if term in ngrams),
        key=lambda idx: len(terms[idx].split()),
        reverse=True
    )
    if len(exact) >= k:
        return exact[:k]

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
    st.set_page_config(page_title="Sugeridor de Keywords Bilingüe")
    st.title("🔑 Generador Consistente de Keywords en ES y EN")
    st.write(
        "Pega tu resumen y obten sugerencias idénticas en Español e Inglés (mismos índices)."
    )

    # Prepara vectorizador una sola vez con vocabulario ES
    vect, matrix = prepare_vectorizer(terms_es)
    summary = st.text_area("Tu resumen aquí:", height=200)
    k = st.slider("Número de palabras clave", 1, 10, 3)

    if st.button("Generar palabras clave"):
        if not summary.strip():
            st.warning("Por favor ingresa un resumen.")
        else:
            idxs = suggest_indices(summary, terms_es, vect, matrix, k)
            st.markdown("**Palabras clave sugeridas:**")
            for idx in idxs:
                es = terms_es[idx].capitalize()
                en = terms_en[idx].capitalize()
                st.write(f"- ES: {es}   |   EN: {en}")

if __name__ == "__main__":
    main()


