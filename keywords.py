import streamlit as st
from thesaurus_terms import THESAURUS_TERMS as terms
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------------------------------------
#  App Streamlit: Generador de palabras clave desde Tesauro UNESCO Embebido
# ------------------------------------------------------------
# • El usuario puede pegar un resumen en el área de texto.
# • La app entrenará TF-IDF en los términos del tesauro y sugerirá las 3 voces más similares.
# ------------------------------------------------------------

@st.cache_data(show_spinner=False)
def prepare_vectorizer(terms: list[str]):
    """Entrena el vectorizador TF-IDF sobre la lista de términos y devuelve el modelo y la matriz."""
    vect = TfidfVectorizer()
    matrix = vect.fit_transform(terms)
    return vect, matrix


def suggest_keywords(summary: str, terms: list[str], vect, matrix, k: int = 3) -> list[str]:
    """Dada una cadena de texto, calcula y retorna las k voces más similares."""
    sims = cosine_similarity(vect.transform([summary]), matrix).flatten()
    top_indices = sims.argsort()[-k:][::-1]
    return [terms[i] for i in top_indices if sims[i] > 0]


def main():
    st.set_page_config(page_title="Generador de Palabras Clave")
    st.title("🔑 Sugeridor de Palabras Clave con Tesauro UNESCO")
    st.write(
        "Pega tu resumen en el siguiente cuadro de texto. "
        "La app propondrá las tres palabras clave más afines del Tesauro UNESCO."
    )

    # Preparamos el modelo una sola vez
    vect, matrix = prepare_vectorizer(terms)

    # Entrada del resumen
    summary = st.text_area("Tu resumen aquí:", height=200)

    # Botón para generar sugerencias
    if st.button("Generar palabras clave"):
        if not summary.strip():
            st.warning("Por favor ingresa un resumen para generar las palabras clave.")
        else:
            kws = suggest_keywords(summary, terms, vect, matrix)
            if kws:
                st.markdown("**Palabras clave sugeridas:**")
                for kw in kws:
                    st.write(f"- {kw.capitalize()}")
            else:
                st.info("No se encontraron coincidencias. Intenta reformular el texto.")

if __name__ == "__main__":
    main()

