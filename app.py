import streamlit as st
from pathlib import Path
from sqlalchemy import create_engine

# IMPORTACIONES CORRECTAS Y ACTUALIZADAS (Sin módulos obsoletos)
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

# Conector de Google Gemini (Evita bloqueos regionales de Groq)
from langchain_google_genai import ChatGoogleGenerativeAI

# Configuración de página
st.set_page_config(page_title="Chat with SQL DB", layout="wide")
st.title("Chat with SQL DataBase")
st.subheader("LangChain Application (Powered by Gemini)")
st.markdown("<br>", unsafe_allow_html=True)

# Barra lateral para credenciales
st.sidebar.header("Configuración de Conexiones")
api_key = st.sidebar.text_input(label="Google AI Studio API Key", type="password")

st.sidebar.markdown("---")

options = ["localhost", "Custom"]
host_option = st.sidebar.selectbox("Choose MySQL Host", options)

if host_option == "localhost":
    mysql_host = "localhost" 
else:
    mysql_host = st.sidebar.text_input("Enter MySQL Host Address")
    
mysql_user = st.sidebar.text_input("MYSQL User")
mysql_password = st.sidebar.text_input("MYSQL password", type="password")
mysql_db = st.sidebar.text_input("MySQL database")

# Función de caché optimizada para la base de datos
@st.cache_resource(ttl="2h")
def configure_db(mysql_host, mysql_user, mysql_password, mysql_db):
    # Soporta contraseñas vacías perfectamente de forma local
    connection_string = f"mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_db}"
    engine = create_engine(connection_string)
    return SQLDatabase(engine)

# Validaciones críticas de la interfaz
if not api_key:
    st.info("Por favor, ingresa tu Google API Key en la barra lateral para comenzar.")
    st.stop()

# Validación corregida: ya NO obliga a rellenar el password
if not (mysql_host and mysql_user and mysql_db):
    st.warning("Por favor, completa los campos de Host, Usuario y Base de datos en la barra lateral.")
    st.stop()

# --- Inicialización Segura de Componentes ---

# 1. Inicializar el modelo LLM con Gemini
llm = ChatGoogleGenerativeAI(
    google_api_key=api_key, 
    model="gemini-3.5-flash", 
    temperature=0,
    streaming=True
)

# 2. Inicializar base de datos de manera segura
try:
    db = configure_db(mysql_host, mysql_user, mysql_password, mysql_db)
except Exception as e:
    st.error(f"Error al conectar a la base de datos: {e}")
    st.stop()

# --- Construir el toolkit y el agente estructurado avanzado ---
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

# INSTRUCCIONES ESTRICTAS PARA EVITAR EL BUCLE INFINITO
suffix_prompt = """
CRITICAL RULES FOR SQL GENERATION:
1. DO NOT wrap the SQL query in markdown code blocks like ```sql or ```. Return ONLY the plain text SQL string.
2. If you find the answer from the database tool, STOP immediately and formulate your final response to the user in Spanish.
3. Do not run the same tool multiple times with the exact same arguments if it gives you an error.
"""

agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type="openai-tools",      # Excelente compatibilidad con llamadas de funciones estructuradas
    handle_parsing_errors=True,
    max_iterations=8,               # Límite seguro para no quemar tokens de forma infinita
    suffix=suffix_prompt            # Inyección de las instrucciones de parada
)

# --- Gestión del Historial de Chat ---

if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [{"role": "assistant", "content": "¡Hola! Estoy conectado a tu base de datos local. ¿Qué deseas consultar?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_query = st.chat_input(placeholder="Ask anything from the database")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        streamlit_callback = StreamlitCallbackHandler(st.container())
        
        try:
            response = agent.run(user_query, callbacks=[streamlit_callback])
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write(response)
        except Exception as e:
            st.error(f"Hubo un error al procesar la consulta: {e}")
