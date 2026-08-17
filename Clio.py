import os
import io
import re
import ftfy
import unicodedata
import time
import pickle
import urllib.parse
import requests
import numpy as np
import faiss
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from google import genai
from google.genai import types
from google.genai.errors import APIError

# ------------------------------------------------------------------------------
# 1. Configuración General
# ------------------------------------------------------------------------------
ROOT_FOLDER_ID = "1EOtPbfr9tH0lhvB4KK9JRb3MA_QjJUzp"
MODEL_NAME = "gemini-3.5-flash-lite" 
INDEX_FILE = "faiss_index.bin"
CHUNKS_FILE = "chunks_data.pkl"

st.set_page_config(page_title="Clio - Asistente Virtual")
st.title("CLIO")
st.caption("Asistente virtual especializado en procesos y manuales operativos de la empresa.")

@st.cache_resource
def get_gemini_client():
    # Soporta tanto GEMINI_API_KEY como GEMINI_FREE_KEY o variables de entorno
    api_key = (
        st.secrets.get("GEMINI_API_KEY")
        or st.secrets.get("GEMINI_FREE_KEY")
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GEMINI_FREE_KEY")
    )
    if not api_key:
        st.error("⚠️ No se encontró la API Key en los secretos de Streamlit ni en las variables de entorno.")
        st.stop()
    return genai.Client(api_key=api_key)

client = get_gemini_client()

@st.cache_resource
def get_local_embedder():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

embedder = get_local_embedder()

# ------------------------------------------------------------------------------
# 2. Descarga de Drive y Extracción con Barra de Progreso
# ------------------------------------------------------------------------------
def escanear_carpetas_y_subcarpetas(folder_id, visitados=None):
    if visitados is None:
        visitados = set()
        
    if folder_id in visitados:
        return set()
        
    visitados.add(folder_id)
    file_ids = set()
    
    try:
        url_folder = f"https://drive.google.com/embeddedfolderview?id={folder_id}"
        response = requests.get(url_folder)
        
        if response.status_code == 200:
            encontrados_files = set(re.findall(r'/file/d/([a-zA-Z0-9_-]+)/view', response.text))
            file_ids.update(encontrados_files)
            
            encontradas_subcarpetas = set(re.findall(r'/drive/folders/([a-zA-Z0-9_-]+)', response.text))
            for sub_id in encontradas_subcarpetas:
                if sub_id not in visitados:
                    file_ids.update(escanear_carpetas_y_subcarpetas(sub_id, visitados))
    except Exception:
        pass
        
    return file_ids

def obtener_nombre_real_archivo(response, file_id):
    cd = response.headers.get("Content-Disposition", "")
    if "filename" in cd:
        match = re.search(r'filename\*?=(?:UTF-8\'\')?["\']?([^"\';\n]+)["\']?', cd)
        if match:
            nombre = urllib.parse.unquote(match.group(1))
            return nombre.strip('"\'')
    return f"Procedimiento_{file_id[:6]}.pdf"

def limpiar_texto_pdf(texto):
    texto = re.sub(r'Docusign Envelope ID: [A-Z0-9-]+', '', texto)
    texto = re.sub(r'Página \d+ de \d+', '', texto, flags=re.IGNORECASE)
    texto = re.sub(r'LOGRAND\s+ENTERTAINMENT GROUP', '', texto, flags=re.IGNORECASE)
    return texto.strip()

def extraer_paginas_pdf(root_id):
    file_ids = list(escanear_carpetas_y_subcarpetas(root_id))
    total_archivos = len(file_ids)
    documentos_paginas = []
    
    barra_progreso = st.progress(0)
    texto_estado = st.empty()
    
    for i, file_id in enumerate(file_ids, start=1):
        porcentaje = i / total_archivos
        barra_progreso.progress(porcentaje)
        texto_estado.caption(f"📂 Descargando y procesando archivo **{i} de {total_archivos}**...")
        
        download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        try:
            pdf_res = requests.get(download_url)
            if pdf_res.status_code == 200 and b"%PDF" in pdf_res.content[:10]:
                nombre_archivo = obtener_nombre_real_archivo(pdf_res, file_id)
                reader = PdfReader(io.BytesIO(pdf_res.content))
                
                for num_pagina, page in enumerate(reader.pages, start=1):
                    raw_text = page.extract_text()
                    if raw_text and raw_text.strip():
                        texto_limpio = limpiar_texto_pdf(raw_text)
                        lower_text = texto_limpio.lower()
                        
                        tiene_fin = "fin de procedimiento" in lower_text or "fin del procedimiento" in lower_text
                        es_matriz = any(kw in lower_text for kw in ["posición / rol", "id actividad", "descripción", "sistema / documento", "riesgos y controles"])
                        
                        documentos_paginas.append({
                            "file_id": file_id,
                            "nombre_archivo": nombre_archivo,
                            "pagina": num_pagina,
                            "texto": texto_limpio,
                            "es_matriz": es_matriz,
                            "tiene_fin": tiene_fin
                        })
        except Exception:
            continue
            
    barra_progreso.empty()
    texto_estado.empty()
    return documentos_paginas

# ------------------------------------------------------------------------------
# 3. Chunking Inteligente y Base Vectorial
# ------------------------------------------------------------------------------
def crear_chunks_inteligentes(paginas_doc):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = []
    chunk_id = 0
    
    for pag in paginas_doc:
        etiquetas = []
        if pag["tiene_fin"]:
            etiquetas.append("[TIPO: ACTIVIDAD FINAL - FIN DE PROCEDIMIENTO OFICIAL]")
        if pag["es_matriz"]:
            etiquetas.append("[TIPO: MATRIZ DE ACTIVIDADES OPERATIVAS DETALLADA]")
        else:
            etiquetas.append("[TIPO: GENERAL / CARÁTULA]")
            
        encabezado = "\n".join(etiquetas)
        sub_chunks = text_splitter.split_text(pag["texto"])
        
        for sub_texto in sub_chunks:
            texto_chunk = f"{encabezado}\n{sub_texto}"
            chunks.append({
                "chunk_id": chunk_id,
                "file_id": pag["file_id"],
                "nombre_archivo": pag["nombre_archivo"],
                "pagina": pag["pagina"],
                "texto": texto_chunk,
                "es_matriz": pag["es_matriz"]
            })
            chunk_id += 1
            
    return chunks

@st.cache_resource
def inicializar_base_vectorial(root_id):
    # Cargar los índices si ya fueron generados
    if os.path.exists(INDEX_FILE) and os.path.exists(CHUNKS_FILE):
        index = faiss.read_index(INDEX_FILE)
        with open(CHUNKS_FILE, "rb") as f:
            chunks = pickle.load(f)
        return index, chunks

    # Generación completa si no existen los binarios
    paginas = extraer_paginas_pdf(root_id)
    if not paginas:
        return None, []
        
    chunks = crear_chunks_inteligentes(paginas)
    textos_chunks = [c["texto"] for c in chunks]
    
    with st.spinner(f"🧠 Generando vectores para {len(chunks)} fragmentos de texto..."):
        vectors = embedder.encode(textos_chunks, show_progress_bar=True)
        vectors = np.array(vectors, dtype="float32")
        
        faiss.normalize_L2(vectors)
        dimension = vectors.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(vectors)
    
    faiss.write_index(index, INDEX_FILE)
    with open(CHUNKS_FILE, "wb") as f:
        pickle.dump(chunks, f)
    
    return index, chunks

# Carga inicial
vector_index, chunks_data = inicializar_base_vectorial(ROOT_FOLDER_ID)

# ------------------------------------------------------------------------------
# 4. Búsqueda RAG Híbrida (Opción B)
# ------------------------------------------------------------------------------
def normalizar_texto(texto):
    """Limpia la codificación UTF-8 y elimina tildes/acentos."""
    if not texto:
        return ""
    # Corregir errores de codificación (ej. DestrucciÃ³n -> Destrucción)
    try:
        texto = ftfy.fix_text(texto)
    except Exception:
        pass
    
    texto_nfd = unicodedata.normalize('NFD', texto.lower())
    return "".join(c for c in texto_nfd if unicodedata.category(c) != 'Mn')

def corregir_utf8_corrupto(texto):
    if not texto:
        return ""
    try:
        return texto.encode('latin-1').decode('utf-8')
    except (UnicodeEncodeError, UnicodeDecodeError):
        return texto

def normalizar_texto(texto):
    if not texto:
        return ""
    texto = corregir_utf8_corrupto(texto)
    texto_nfd = unicodedata.normalize('NFD', texto.lower())
    return "".join(c for c in texto_nfd if unicodedata.category(c) != 'Mn')

def corregir_utf8_corrupto(texto):
    if not texto:
        return ""
    try:
        return texto.encode('latin-1').decode('utf-8')
    except (UnicodeEncodeError, UnicodeDecodeError):
        return texto

def normalizar_texto(texto):
    if not texto:
        return ""
    texto = corregir_utf8_corrupto(texto)
    texto_nfd = unicodedata.normalize('NFD', texto.lower())
    return "".join(c for c in texto_nfd if unicodedata.category(c) != 'Mn')

def buscar_contexto_relevante(pregunta, doc_activo_actual=None):
    if not chunks_data or not vector_index:
        return "", [], None

    pregunta_norm = normalizar_texto(pregunta)
    doc_activo_norm = normalizar_texto(doc_activo_actual) if doc_activo_actual else None

    STOPWORDS_OPERATIVAS = {
        "procedimiento", "proceso", "evento", "inicia", "inicio", "final", 
        "conclusion", "conclusión", "marca", "este", "esta", "estos", "estas",
        "cual", "cuál", "como", "cómo", "para", "donde", "dónde", "pasos",
        "actividad", "actividades", "manual", "politica", "política", "empresa", "saber",
        "del", "las", "los", "que", "con", "por", "para", "una", "uno", "unos", "su", "sus",
        "existe", "algún", "paso", "entre", "confirma", "únicamente", "presente", "diagrama",
        "conector", "pasa", "directo", "otro", "flujo", "compuerta", "decisión", "desicion",
        "encargado", "ejecutar", "rol", "departamento", "tiene", "indícame", "indicame",
        "flecha", "regreso", "retrabajo", "ciclo", "regresa", "sale"
    }

    palabras_especificas = [
        w for w in re.findall(r'\w+', pregunta_norm) 
        if len(w) > 2 and w not in STOPWORDS_OPERATIVAS
    ]

    numeros_buscados = set(re.findall(r'\b\d+\b', pregunta))

    # 1. Detectar si la pregunta solicita EXPLÍCITAMENTE un NUEVO manual
    mejor_doc_coincidente = None
    max_matches_titulo = 0

    if palabras_especificas:
        for c in chunks_data:
            nombre_norm = normalizar_texto(c["nombre_archivo"])
            matches = sum(1 for kw in palabras_especificas if kw in nombre_norm)
            if matches > max_matches_titulo:
                max_matches_titulo = matches
                mejor_doc_coincidente = c["nombre_archivo"]

    es_cambio_de_tema = False
    if mejor_doc_coincidente:
        doc_coincidente_norm = normalizar_texto(mejor_doc_coincidente)
        # Solo cambia de tema si hay al menos 1 coincidencia en el título de UN MANUAL DISTINTO
        if doc_activo_norm is None or doc_coincidente_norm not in doc_activo_norm:
            if max_matches_titulo >= 1:
                es_cambio_de_tema = True

    chunk_scores = {c["chunk_id"]: 0.0 for c in chunks_data}

    # 2. Puntuación Vectorial
    q_vector = embedder.encode([pregunta])
    q_vector = np.array(q_vector, dtype="float32")
    faiss.normalize_L2(q_vector)
    
    distances, indices = vector_index.search(q_vector, k=25)
    for sim, idx in zip(distances[0], indices[0]):
        if idx < len(chunks_data):
            chunk_scores[idx] += float(sim) * 1.5

    # 3. Puntuación Jerárquica y Anclaje Estricto
    nuevo_doc_activo = doc_activo_actual

    for c in chunks_data:
        nombre_norm = normalizar_texto(c["nombre_archivo"])
        texto_norm = normalizar_texto(c["texto"])
        
        matches_nombre = sum(1 for kw in palabras_especificas if kw in nombre_norm) if palabras_especificas else 0
        matches_texto = sum(1 for kw in palabras_especificas if kw in texto_norm) if palabras_especificas else 0

        # CASO A: Solicitud explícita de cambio de documento
        if es_cambio_de_tema and mejor_doc_coincidente and normalizar_texto(mejor_doc_coincidente) in nombre_norm:
            chunk_scores[c["chunk_id"]] += 100.0
            nuevo_doc_activo = mejor_doc_coincidente

        # CASO B: Continuidad en el manual activo (Bloqueo prioritario +150.0)
        elif doc_activo_norm and doc_activo_norm in nombre_norm and not es_cambio_de_tema:
            chunk_scores[c["chunk_id"]] += 150.0

        if numeros_buscados:
            for num in numeros_buscados:
                if re.search(r'\b' + re.escape(num) + r'\b', texto_norm):
                    chunk_scores[c["chunk_id"]] += 30.0

        if matches_texto > 0 and palabras_especificas:
            chunk_scores[c["chunk_id"]] += (matches_texto / len(palabras_especificas)) * 2.0

    # 4. Filtrado de chunks
    chunks_ordenados_por_score = sorted(
        chunks_data, 
        key=lambda c: chunk_scores[c["chunk_id"]], 
        reverse=True
    )

    chunks_finales = []
    for c in chunks_ordenados_por_score:
        if chunk_scores[c["chunk_id"]] > 0.1 and len(chunks_finales) < 10:
            chunks_finales.append(c)

    # Definir el documento ganador para actualizar la memoria
    if chunks_finales and (nuevo_doc_activo is None or es_cambio_de_tema):
        nuevo_doc_activo = chunks_finales[0]["nombre_archivo"]

    chunks_ordenados = sorted(chunks_finales, key=lambda x: (x['nombre_archivo'], x['pagina'], x['chunk_id']))
    
    contexto_recuperado = ""
    fuentes_usadas = set()
    
    for chunk in chunks_ordenados:
        nombre_limpio = corregir_utf8_corrupto(chunk['nombre_archivo'])
        contexto_recuperado += f"\n--- [DOCUMENTO: {nombre_limpio} | PÁGINA {chunk['pagina']}] ---\n"
        contexto_recuperado += chunk['texto'] + "\n"
        fuentes_usadas.add(f"{nombre_limpio} (Pág. {chunk['pagina']})")
            
    return contexto_recuperado, sorted(list(fuentes_usadas)), nuevo_doc_activo
# ------------------------------------------------------------------------------
# 5. Renderizado e Interfaz de Usuario
# ------------------------------------------------------------------------------
def renderizar_mensaje_asistente(contenido):
    texto_principal = contenido
    fuente_texto = None

    if "---FUENTE---" in contenido:
        partes = contenido.split("---FUENTE---")
        texto_principal = partes[0].strip()
        fuente_texto = partes[1].strip()

    st.markdown(texto_principal)
    
    if fuente_texto:
        with st.expander("Ver documento fuente y páginas de referencia"):
            st.markdown(f"**Referencia:** {fuente_texto}")

if "messages" not in st.session_state:
    st.session_state.messages = []

for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            renderizar_mensaje_asistente(message["content"])
            
            feedback_key = f"feedback_{idx}"
            current_feedback = message.get("feedback", None)
            tiempo_seg = message.get("tiempo_respuesta", None)

            def on_feedback_change(index=idx, key=feedback_key):
                val = st.session_state[key]
                st.session_state.messages[index]["feedback"] = val
                calif = "👍 Positiva" if val == 1 else "👎 Negativa"
                st.toast(f"¡Gracias por calificar la respuesta! ({calif})")

            col1, col2 = st.columns([1, 4])
            with col1:
                st.feedback(
                    "thumbs", 
                    key=feedback_key, 
                    on_change=on_feedback_change,
                    disabled=(current_feedback is not None)
                )
            with col2:
                if tiempo_seg:
                    st.caption(f"⏱️ {tiempo_seg} seg tiempo real")
        else:
            st.markdown(message["content"])

# ------------------------------------------------------------------------------
# 6. Procesamiento de Preguntas y Medición de Tiempo
# ------------------------------------------------------------------------------
if prompt := st.chat_input("¿En qué te puedo ayudar hoy?"):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Buscando información..."):
            try:
                inicio_tiempo = time.time()

                contexto_filtrado, fuentes, nuevo_doc = buscar_contexto_relevante(
                    prompt,
                    doc_activo_actual=st.session_state.get("doc_activo", None)
                )
                
                # Actualizar el documento activo en la memoria de sesión
                if nuevo_doc:
                    st.session_state.doc_activo = nuevo_doc

                SYSTEM_PROMPT = f"""
Eres Clio, el asistente virtual oficial de la empresa. Tu objetivo es explicar procesos, políticas y manuales operativos con máximo detalle, exactitud profesional y rigor.

FRAGMENTOS RECUPERADOS DE LOS MANUALES OFICIALES:
\"\"\"
{contexto_filtrado if contexto_filtrado else "No se encontraron fragmentos relevantes."}
\"\"\"

Instrucción de extracción estricta: Responde ÚNICAMENTE con los datos explícitos del texto. Queda estrictamente prohibido suponer, deducir o inventar cargos, puestos o actividades que no aparezcan literalmente escritos en el documento.

REGLAS DE RESPUESTA ESTRICTAS:
1. Responde ÚNICAMENTE basándote en el texto explícito y literal del contexto proporcionado.
2. Si el usuario pregunta por algo que NO existe en el procedimiento (por ejemplo: flechas de regreso, ciclos de retrabajo, excepciones, roles o pasos no mencionados), DEBES responder categóricamente que dicho elemento NO EXISTE o NO ESTÁ ESPECIFICADO en el documento.
3. ESTÁ ESTRICTAMENTE PROHIBIDO:
   - Asumir o deducir consecuencias indirectas (ej. decir que "detener el proceso" equivale a "un ciclo de regreso").
   - Intentar adaptar o acomodar una respuesta si el documento no contiene el concepto exacto preguntado.
   - Usar frases como "o definir un estado restrictivo" para justificar la falta de información.
4. Si un proceso es completamente lineal, indícalo con claridad: "El procedimiento es completamente lineal y no cuenta con ciclos de retrabajo ni condiciones de retorno".
5. Mantén la respuesta concisa, directa y siempre citando el documento y página correspondiente.

REGLAS DE RESPUESTA UNIVERSALES:
1. EXPLICACIÓN COMPLETA DE PROCESOS:
   - Al explicar el inicio, detonador o conclusión de cualquier proceso, NO te limites a los resúmenes ejecutivos de carátula si la documentación contiene la matriz detallada de actividades.
   - Para definir el EVENTO FINAL o CONCLUSIÓN de cualquier proceso, busca en la matriz de actividades la actividad específica que marque el término del flujo (indicada por leyendas como 'Fin de Procedimiento', 'Conclusión', o la última actividad numerada de la secuencia). No confundir con revisiones rutinarias intermedias.
2. EXTRAER DATOS CLAVE DE LA MATRIZ:
   Siempre que la información esté disponible en los fragmentos, detalla:
   - ID / Número de Actividad.
   - Posición / Rol responsable de ejecutarla.
   - Descripción concreta de la acción y canales empleados (correos, sistemas, etc.).
   - Entregable, documento o resultado final.
3. EXHAUSTIVIDAD EN LISTAS Y FLUJOS:
   - Revisa todos los fragmentos recibidos de principio a fin para descartar imprecisiones.
4. ESTILO Y FORMATO:
   - Usa un tono corporativo, claro y estructurado con viñetas.
   - Utiliza negritas para resaltar roles, códigos de documentos y números de actividad.
5. FORMATO DE FUENTE OBLIGATORIO:
   - Al final de cualquier respuesta operativa, coloca la etiqueta `---FUENTE---` en una línea nueva y abajo enlista los documentos y páginas utilizados:
     Ejemplo:
     ---FUENTE---
     * Nombre_Documento.pdf (Pág. X)
6. Para saludos o preguntas generales de cortesía, responde brevemente sin usar `---FUENTE---`.
"""

                chat_history = []
                for msg in st.session_state.messages[:-1]:
                    role = "user" if msg["role"] == "user" else "model"
                    chat_history.append(
                        types.Content(
                            role=role, 
                            parts=[types.Part.from_text(text=msg["content"])]
                        )
                    )

                chat_history.append(
                    types.Content(
                        role="user", 
                        parts=[types.Part.from_text(text=prompt)]
                    )
                )

                gemini_config = types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.0,
                )

                # LÓGICA DE REINTENTOS AUTOMÁTICOS (MÁXIMO 3 INTENTOS)
                max_reintentos = 3
                response = None

                for intento in range(max_reintentos):
                    try:
                        response = client.models.generate_content(
                            model=MODEL_NAME,
                            contents=chat_history,
                            config=gemini_config
                        )
                        break # Si tuvo éxito, sale del bucle de reintentos
                    except APIError as e:
                        if ("503" in str(e) or "UNAVAILABLE" in str(e)) and intento < max_reintentos - 1:
                            tiempo_espera = (intento + 1) * 2  # Espera 2s en el 1er fallo, 4s en el 2do fallo
                            time.sleep(tiempo_espera)
                            continue
                        raise e # Si es otro error o ya superó los 3 reintentos, lanza la excepción

                tiempo_total = time.time() - inicio_tiempo

                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response.text,
                    "feedback": None,
                    "tiempo_respuesta": f"{tiempo_total:.1f}"
                })
                st.rerun()

            except APIError as e:
                if "503" in str(e) or "UNAVAILABLE" in str(e):
                    st.error("☁️ **Servidores de Google saturados (Error 503).** Los servidores de Gemini tienen alta demanda en este momento. Por favor, reintenta tu pregunta en unos segundos.")
                elif "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    st.error("⏳ **Límite de solicitudes alcanzado (Error 429).** Has superado la cuota permitida por minuto. Por favor espera un momento.")
                else:
                    st.error(f"⚠️ Ocurrió un error en la API de Gemini: {e}")
            except Exception as e:
                st.error(f"⚠️ Ocurrió un error al conectar con Clio: {e}")