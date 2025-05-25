@st.cache_data(show_spinner=False)
def load_thesaurus(pdf_source="unesco-thesaurus-es.pdf"):
    """Extrae un conjunto de términos (en minúsculas) del Tesauro PDF.
       Acepta una ruta local, un objeto UploadedFile o una URL."""
    
    fp = None
    close_needed = False
    
    if isinstance(pdf_source, str):
        # Verificar si es una URL o una ruta local
        if pdf_source.startswith('http://') or pdf_source.startswith('https://'):
            try:
                response = requests.get(pdf_source)
                response.raise_for_status() # Lanza un error si la descarga falla
                # Usar BytesIO para tratar el contenido descargado como un archivo
                fp = io.BytesIO(response.content)
                close_needed = False # BytesIO no necesita close explícito en este contexto
            except requests.exceptions.RequestException as e:
                st.error(f"Error al descargar el PDF desde la URL: {e}")
                return [] # Retorna una lista vacía si falla la descarga
        else:
            # Es una ruta de archivo local
            try:
                fp = open(pdf_source, "rb")
                close_needed = True
            except FileNotFoundError:
                st.error(f"Error: Archivo local no encontrado en '{pdf_source}'")
                return [] # Retorna una lista vacía si el archivo local no existe
    elif hasattr(pdf_source, 'read'): # Verifica si es un objeto tipo archivo (como UploadedFile)
        fp = pdf_source
        close_needed = False
    else:
        st.error("Tipo de fuente PDF no soportado.")
        return [] # Retorna una lista vacía para tipos no soportados


    if fp is None: # Si no se pudo abrir ni descargar el archivo
        return []
        
    reader = PyPDF2.PdfReader(fp)
    terms = set()

    for page in reader.pages:
        text = page.extract_text() or ""
        for raw in text.splitlines():
            line = raw.strip()
            # Filtra numeración u otras líneas vacías
            if not line or line.isdigit():
                continue
            # Suprime prefijos de relaciones (AR:, BT:, UF:, etc.)
            if re.match(r"^(AR|EN|FR|RU|UF|BT|NT|RT|SN|MT)\s*:", line):
                continue
            # El término acaba ante un ‘:’ si existe
            term = line.split(":", 1)[0]
            # Sólo letras y espacios
            term = re.sub(r"[^A-Za-zÁÉÍÓÚÜÑáéíóúüñ ]", "", term).strip().lower()
            if term:
                terms.add(term)

    # Aunque BytesIO no necesita un .close() explícito en este caso,
    # si fp era un archivo local abierto, sí debemos cerrarlo.
    # La variable close_needed maneja esto.
    if close_needed:
        fp.close()

    return sorted(terms)
