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
# 2. PANTALLA DE CONFIGURACIÓN
# ==========================================
if not st.session_state.configurado:
    st.title("🔬 Configuración de la sesión")
    st.info("Ingresa los datos del grupo para habilitar el asistente.")
    
    with st.form("registro"):
        col1, col2 = st.columns(2)
        with col1:
            nrc = st.text_input("Asignatura/NRC")
            grupo = st.text_input("ID del grupo")
            tema = st.text_input("Tema a trabajar", placeholder="Ej: Distribución Normal")
        with col2:
            archivo_pdf = st.file_uploader("Subir materiales (opcional)", type="pdf")
            integrantes = st.text_area("Integrantes (uno por línea)")
        
        if st.form_submit_button("Lanzar asistente"):
            if nrc and grupo and tema and integrantes:
                if archivo_pdf:
                    with st.spinner("Indexando PDF..."):
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                            tmp.write(archivo_pdf.getvalue())
                            loader = PyPDFLoader(tmp.name)
                            docs = loader.load_and_split()
                            embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])
                            st.session_state.vector_db = FAISS.from_documents(docs, embeddings)
                        os.remove(tmp.name)
                
                st.session_state.nrc = nrc
                st.session_state.grupo = grupo
                st.session_state.tema = tema
                st.session_state.estudiantes = [i.strip() for i in integrantes.split("\n") if i.strip()]
                st.session_state.configurado = True
                st.rerun()
            else:
                st.warning("Completa todos los campos obligatorios.")
    st.stop()

# ==========================================
# 3. INTERFAZ DE CHAT Y LOGGING
# ==========================================
st.title(f"🤖 Asistente: {st.session_state.tema}")
st.sidebar.markdown(f"**ID de Sesión:** `{st.session_state.session_uuid}`")


for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Si el mensaje es de la IA y tiene un ID de base de datos
        if msg["role"] == "assistant" and "db_id" in msg:
            f_key = f"fb_{msg['db_id']}"
            # feedback: 0 es abajo (👎), 1 es arriba (👍)
            feedback = st.feedback("thumbs", key=f_key)
            
            if feedback is not None:
                val = "up" if feedback == 1 else "down"
                
                try:
                    # 1. Actualizamos el pulgar (up/down) en Supabase
                    supabase.table("interacciones_investigacion").update({"feedback": val}).eq("id", msg["db_id"]).execute()
                    
                    # 2. Si el feedback es negativo (0), mostramos el cuadro de texto
                    if feedback == 0: 
                        t_key = f"txt_{msg['db_id']}"
                        comentario = st.text_input(
                            "¿Cómo podemos mejorar esta respuesta?", 
                            key=t_key,
                            placeholder="Ej: La respuesta es incorrecta o no es clara..."
                        )
                        
                        # Si el estudiante escribe y presiona Enter
                        if comentario:
                            supabase.table("interacciones_investigacion").update({"feedback_text": comentario}).eq("id", msg["db_id"]).execute()
                            st.toast("Comentario guardado. ¡Gracias!", icon="📝")
                    
                    # 3. Mensajes de confirmación (toast)
                    elif feedback == 1:
                        st.toast("¡Gracias! Feedback positivo registrado.", icon="👍")
                        
                except Exception as e:
                    pass # Evita que errores de red bloqueen la interfaz


with st.sidebar:
    st.header("Asistente")
    autor = st.selectbox("¿Quién escribe ahora?", st.session_state.estudiantes)
    st.divider()
    if st.button("🔴 Descargar respaldo CSV"):
        df = pd.DataFrame(st.session_state.log_buffer)
        csv = df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
        st.download_button("Guardar archivo", data=csv, file_name=f"log_{st.session_state.session_uuid}.csv")

prompt = st.chat_input("Escribe tu duda o explicación...")

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
            "PROHIBIDO usar delimitadores como \( \) o \[ \]."
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