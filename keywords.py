import streamlit as st
import requests
import io
import PyPDF2
import re

@st.cache_data(show_spinner=False)
def load_thesaurus(pdf_source="unesco-thesaurus-es.pdf"):
    """Extrae un conjunto de términos (en minúsculas) del Tesauro PDF.
       Acepta una ruta local, un objeto UploadedFile o una URL."""
    
    fp = None
    close_needed = False
    
    if isinstance(pdf_source, str):
        # Verificar si es una URL o una ruta local
        if pdf_source.startswith("http://") or pdf_source.startswith("https://"):
            try:
                response = requests.get(pdf_source)
                response.raise_for_status()  # Error si falla la descarga
                fp = io.BytesIO(response.content)
                close_needed = False
            except requests.RequestException as e:
                st.error(f"Error al descargar el PDF desde la URL: {e}")
                return []
        else:
            # Es una ruta de archivo local
            try:
                fp = open(pdf_source, "rb")
                close_needed = True
            except FileNotFoundError:
                st.error(f"Error: Archivo local no encontrado en '{pdf_source}'")
                return []
    elif hasattr(pdf_source, "read"):
        # UploadedFile u otro file-like
        fp = pdf_source
        close_needed = False
    else:
        st.error("Tipo de fuente PDF no soportado.")
        return []

    # Si no se pudo abrir ni descargar el archivo
    if fp is None:
        return []
        
    reader = PyPDF2.PdfReader(fp)
    terms = set()

    for page in reader.pages:
        text = page.extract_text() or ""
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.isdigit():
                continue
            if re.match(r"^(AR|EN|FR|RU|UF|BT|NT|RT|SN|MT)\s*:", line):
                continue
            term = line.split(":", 1)[0]
            term = re.sub(r"[^A-Za-zÁÉÍÓÚÜÑáéíóúüñ ]", "", term).strip().lower()
            if term:
                terms.add(term)

    if close_needed:
        fp.close()

    return sorted(terms)

