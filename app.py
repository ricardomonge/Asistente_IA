import streamlit as st
import pandas as pd
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import tempfile
import os
import uuid
from datetime import datetime
from supabase import create_client, Client

# ==========================================
# 1. CONFIGURACIÓN Y SEGURIDAD
# ==========================================
st.set_page_config(page_title="Asistente IA - IMFE", layout="wide")

# Conexión con Supabase
try:
    url: str = st.secrets["SUPABASE_URL"]
    key: str = st.secrets["SUPABASE_KEY"]
    supabase: Client = create_client(url, key)
except KeyError:
    st.error("⚠️ Error: Credenciales de Supabase no configuradas.")
    st.stop()

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Estados de sesión
if "session_uuid" not in st.session_state:
    st.session_state.session_uuid = str(uuid.uuid4())[:8]
if "messages" not in st.session_state:
    st.session_state.messages = []
if "log_buffer" not in st.session_state:
    st.session_state.log_buffer = []
if "configurado" not in st.session_state:
    st.session_state.configurado = False
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

# ==========================================
# 2. PANTALLA DE CONFIGURACIÓN (DISEÑO PRO)
# ==========================================
if not st.session_state.configurado:
    col_logo, col_titulo = st.columns([1, 8])
    with col_logo:
        st.markdown("# 🔬")
    with col_titulo:
        st.title("Configuración del Entorno de Aprendizaje")
        st.markdown("_Asistente de IA Colaborativa | IMFE_")

    # Guía clara para el estudiante
    with st.expander("📖 Guía de Registro e Instrucciones", expanded=False):
        st.markdown("""
        1. **Identificación**: Ingrese el NRC y el ID de su grupo de trabajo.
        2. **Tema**: Defina el concepto a trabajar (Ej.: Distribución Normal).
        3. **Materiales**: Puede subir varios archivos PDF (Máximo 25 MB en total).
        4. **Integrantes**: Registre los nombres de su equipo, uno por línea.
        """)

    st.divider()

    with st.form("registro_investigacion"):
        st.subheader("🛠️ Panel de Control de Sesión")
        col_left, col_right = st.columns([1, 1], gap="large")
        
        with col_left:
            st.markdown("**Datos del curso**")
            nrc = st.text_input("Asignatura / Código NRC", placeholder="Ej: MAT101 / 2345")
            grupo = st.text_input("Identificador del Grupo", placeholder="Ej: Grupo A-1")
            tema = st.text_input("Tema a trabajar en esta sesión", placeholder="Ej.: Distribución Normal")
            
        with col_right:
            st.markdown("**Recursos y participantes**")
            archivos_pdf = st.file_uploader(
                "Subir materiales PDF (Opcional)", 
                type="pdf", 
                accept_multiple_files=True, 
                help="Límite máximo del lote completo: 25 MB."
            )
            integrantes = st.text_area(
                "Integrantes del grupo (uno por línea)", 
                placeholder="Ej.: Juan P.\nMaría G.\nPedro A. ...", 
                height=110
            )

        # === NUEVO: CONSENTIMIENTO ÉTICO ACADÉMICO ===
        st.divider()
        st.markdown("**Consentimiento para participantes en investigación educativa**")
        acepta_terminos = st.checkbox(
            "Consiento voluntariamente mi participación en esta sesión y autorizo el tratamiento automatizado de los datos "
            "derivados de mi interacción con este asistente. La información recolectada será procesada de forma estrictamente "
            "anónima y confidencial, con el propósito exclusivo de realizar análisis pedagógicos y contribuir a la "
            "investigación educativa desarrollada en el IMFE."
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Estilo CSS para el botón azul tenue
        st.markdown("""
            <style>
            div.stButton > button:first-child {
                background-color: #E1F5FE !important; color: #01579B !important;
                border: 1px solid #B3E5FC !important; border-radius: 8px;
                transition: all 0.3s ease; font-weight: bold;
            }
            div.stButton > button:first-child:hover {
                background-color: #B3E5FC !important; transform: scale(1.01);
            }
            </style>
        """, unsafe_allow_html=True)

        lanzar = st.form_submit_button("🚀 Inicializar Asistente Académico", width='stretch')
        
        if lanzar:
            # Validación: Se requiere que todos los campos y el consentimiento estén listos
            if nrc and grupo and tema and integrantes:
                if acepta_terminos:
                    if archivos_pdf:
                        total_size_mb = sum([f.size for f in archivos_pdf]) / (1024 * 1024)
                        if total_size_mb > 25:
                            st.error(f"❌ El total de archivos ({total_size_mb:.2f} MB) supera el límite de 25 MB.")
                        else:
                            with st.spinner("⏳ Procesando materiales pedagógicos..."):
                                todos_los_docs = []
                                for archivo in archivos_pdf:
                                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                                        tmp.write(archivo.getvalue())
                                        loader = PyPDFLoader(tmp.name)
                                        todos_los_docs.extend(loader.load_and_split())
                                    os.remove(tmp.name)
                                
                                embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])
                                st.session_state.vector_db = FAISS.from_documents(todos_los_docs, embeddings)
                                st.session_state.nombres_archivos = [a.name for a in archivos_pdf]
                                
                                st.session_state.nrc = nrc
                                st.session_state.grupo = grupo
                                st.session_state.tema = tema
                                st.session_state.estudiantes = [i.strip() for i in integrantes.split("\n") if i.strip()]
                                st.session_state.configurado = True
                                st.rerun()
                    else:
                        # Inicialización sin archivos (Conocimiento general)
                        st.session_state.nrc = nrc
                        st.session_state.grupo = grupo
                        st.session_state.tema = tema
                        st.session_state.estudiantes = [i.strip() for i in integrantes.split("\n") if i.strip()]
                        st.session_state.configurado = True
                        st.rerun()
                else:
                    st.warning("⚠️ Para continuar, debe leer y aceptar el consentimiento ético de tratamiento de datos.")
            else:
                st.error("❌ Por favor, complete todos los campos obligatorios.")

    st.stop()

# ==========================================
# 3. INTERFAZ DE CHAT Y CONTROL DE SESIÓN
# ==========================================

# Título y subtítulo profesional
st.title(f"🤖 Laboratorio IA: {st.session_state.tema}")
st.caption(f"ID Único: {st.session_state.session_uuid} | IMFE")

# --- SIDEBAR ACADÉMICA ---
with st.sidebar:
    st.image("https://raw.githubusercontent.com/ricardomonge/Asistente_IA/refs/heads/main/image/logo.png", width='stretch') 
    st.caption("⚠️ **Aviso importante:** Este asistente puede cometer errores; por lo tanto, revisa y verifica siempre sus respuestas.")    
    st.header("Control de sesión")
    
    # Resumen de datos de la sesión
    with st.container(border=True):
        st.markdown(f"**NRC:** {st.session_state.nrc}")
        st.markdown(f"**Grupo:** {st.session_state.grupo}")
        st.markdown(f"**Estudiantes:** {len(st.session_state.estudiantes)}")

# ============ INDICADOR DE MODO RAG ACTUALIZADO ===
    st.markdown("**Estado de la IA**")
    if st.session_state.get("vector_db"):
        st.success("Base de conocimiento activa", icon="✅")
        nombres = st.session_state.get('nombres_archivos', [])
        with st.expander(f"📚 Archivos en memoria ({len(nombres)})"):
            for nombre in nombres:
                st.caption(f"• {nombre}")
    else:
        st.warning("Modo conocimiento general", icon="🌐")
        st.caption("No hay archivos cargados.")
    # ==================================================
    
    # Selector de autor
    autor = st.selectbox("📝 Estudiante interactuando:", st.session_state.estudiantes)
    
    # st.divider()
    # Guía de Apoyo Académico (Fomenta CoT)
    # with st.expander("💡 Tips para aprender mejor"):
    #    st.info("""
    #    1. **Pregunta el 'Por qué'**: No solo pidas el resultado, pide el razonamiento paso a paso.
    #    2. **Valida con el PDF**: Si subiste material, pide a la IA que cite la página o sección.
    #    3. **Corrige a la IA**: Si detectas un error en una fórmula, explícaselo para ver cómo rectifica.
    #    """)
    
    # st.divider()

    # BOTÓN DE FINALIZACIÓN CON DOBLE VERIFICACIÓN
    if "finalizado" not in st.session_state:
        st.session_state.finalizado = False

    if not st.session_state.finalizado:
        if st.button("🔴 Finalizar sesión", width='stretch'):
            st.session_state.esperando_confirmacion = True
        
        if st.session_state.get("esperando_confirmacion"):
            st.warning("¿Está seguro? No podrá enviar más mensajes.")
            col_si, col_no = st.columns(2)
            with col_si:
                if st.button("Sí, cerrar", type="primary"):
                    st.session_state.finalizado = True
                    st.session_state.esperando_confirmacion = False
                    st.rerun()
            with col_no:
                if st.button("Cancelar"):
                    st.session_state.esperando_confirmacion = False
                    st.rerun()
    else:
        st.error("🔒 Sesión Concluida")

for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        if msg["role"] == "assistant" and "db_id" in msg:
            f_key = f"fb_{msg['db_id']}"
            feedback = st.feedback("thumbs", key=f_key)
            
            # --- MEJORA DE CONSISTENCIA ---
            # Creamos una llave en el estado de sesión para saber si ya procesamos este feedback
            processed_key = f"last_state_{f_key}"
            
            if feedback is not None and st.session_state.get(processed_key) != feedback:
                val = "up" if feedback == 1 else "down"
                try:
                    # Actualizamos Supabase solo si el valor cambió
                    supabase.table("interacciones_investigacion").update({"feedback": val}).eq("id", msg["db_id"]).execute()
                    st.session_state[processed_key] = feedback # Guardamos el estado actual como procesado
                    
                    # Mostramos el mensaje solo en el momento del click
                    if val == "up":
                        st.toast("¡Gracias! Feedback positivo registrado.", icon="👍")
                    else:
                        st.toast("Feedback negativo registrado.", icon="👎")
                except Exception as e:
                    pass

            # Lógica del cuadro de texto cualitativo
            if feedback == 0: 
                t_key = f"txt_{msg['db_id']}"
                comentario = st.text_input(
                    "¿Cómo podemos mejorar esta respuesta?", 
                    key=t_key,
                    placeholder="Ej: La respuesta es incorrecta o no es clara..."
                )
                
                # Verificamos si el comentario es nuevo para no repetir el agradecimiento
                comment_key = f"last_com_{msg['db_id']}"
                if comentario and st.session_state.get(comment_key) != comentario:
                    try:
                        supabase.table("interacciones_investigacion").update({"feedback_text": comentario}).eq("id", msg["db_id"]).execute()
                        st.session_state[comment_key] = comentario # Marcamos comentario como guardado
                        st.toast("Comentario guardado. ¡Gracias!", icon="📝")
                    except:
                        pass

# --- ENTRADA DE MENSAJES (Bloqueada si terminó la sesión) ---
if not st.session_state.finalizado:
    prompt = st.chat_input("Escribe...")
else:
    st.info("La sesión ha finalizado. Los datos han sido resguardados en el servidor. Gracias por participar.")
    prompt = None

if prompt:
    # 1. Registro visual del mensaje del usuario
    display_user = f"**{autor}:** {prompt}"
    st.session_state.messages.append({"role": "user", "content": display_user})
    with st.chat_message("user"):
        st.markdown(display_user)
    
    # 2. ESPERA ACTIVA: Todo lo que tarda tiempo va dentro del spinner
    with st.spinner("El asistente está analizando los materiales (si los subiste) y pensando su respuesta..."):
        
        # Lógica de búsqueda RAG
        contexto_txt = ""
        if st.session_state.vector_db:
            docs_rel = st.session_state.vector_db.similarity_search(prompt, k=3)
            contexto_txt = "\n\nCONTEXTO MATERIAL:\n" + "\n".join([d.page_content for d in docs_rel])

        # Configuración del rol de la IA
        sys_prompt = (
            f"Eres un asistente experto en {st.session_state.tema}. "
            "Tu tono es profesional, pedagógico y resolutivo. "
            "Ayuda a los estudiantes a entender el concepto y resolver problemas paso a paso. "
            "\n\nIMPORTANTE (FORMATO MATEMÁTICO): "
            "Usa SIEMPRE LaTeX para fórmulas. "
            "Usa un solo '$' para fórmulas en línea (ej: $z = \\frac{x - \\mu}{\\sigma}$) "
            "y doble '$$' para fórmulas destacadas en bloques. "
            r"PROHIBIDO usar delimitadores como \( \) o \[ \]."
    f"{contexto_txt}"
        )

        # Llamada a la API de OpenAI (aquí es donde ocurre la espera principal)
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": prompt}],
                temperature=0.7
            )
            ai_res = response.choices[0].message.content
            ai_res = ai_res.replace(r"\(", "$").replace(r"\)", "$").replace(r"\[", "$$").replace(r"\]", "$$")
        except Exception as e:
            ai_res = f"Error al generar respuesta: {e}"
            

    # 3. Guardar en Base de Datos para obtener el ID
    log_data = {
        "session_id": st.session_state.session_uuid,
        "nrc": st.session_state.nrc,
        "grupo": st.session_state.grupo,
        "tema": st.session_state.tema,
        "estudiante": autor,
        "mensaje_usuario": prompt,
        "respuesta_ia": ai_res,
        "usa_rag": bool(st.session_state.vector_db),
        "timestamp": datetime.now().isoformat()
    }
    
    try:
        # Insertamos en Supabase y recuperamos el ID generado automáticamente
        res_db = supabase.table("interacciones_investigacion").insert(log_data).execute()
        new_db_id = res_db.data[0]['id']
        
        # 4. Agregamos al historial de la sesión el ID de la DB
        st.session_state.messages.append({"role": "assistant", "content": ai_res, "db_id": new_db_id})
        st.session_state.log_buffer.append(log_data)
        
        # 5. RECARGA: Necesaria para que aparezcan los dedos inmediatamente
        st.rerun()
        
    except Exception as e:
        # En caso de error, guardamos el mensaje sin ID para no bloquear el chat
        st.session_state.messages.append({"role": "assistant", "content": ai_res})
        st.sidebar.error(f"Error de registro: {e}")
        
    st.session_state.log_buffer.append(log_data)
