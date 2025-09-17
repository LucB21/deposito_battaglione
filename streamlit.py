import streamlit as st
from streamlit_option_menu import option_menu
from openai import AzureOpenAI

################################################################################################
from tenacity import retry, stop_after_attempt, wait_exponential, before_log, after_log
import logging

# Configura logging
logging.basicConfig(level=logging.INFO)
################################################################################################

class Sidebar:
    @staticmethod
    def render():
        with st.sidebar:
            selected = option_menu(
                "Menu",
                ["Chat", "Impostazioni"],
                menu_icon="umbrella",
                icons=["house", "gear"]
            )
        return selected
 
class SettingsPage:
    @staticmethod
    def render():
        st.title("Inserisci le tue informazioni")
        st.divider()
        apikey = st.session_state["api_key"] = st.text_input(
            "Inserisci la tua key",
            value=st.session_state.get("api_key", "")
        )
        endpoint = st.session_state["endpoint"] = st.text_input(
            "Inserisci il tuo endpoint",
            value=st.session_state.get("endpoint", "")
        )
        apiversion = st.session_state["api_version"] = st.text_input(
            "Inserisci la tua versione API",
            value=st.session_state.get("api_version", "")
        )
 
 
import streamlit as st
from streamlit_option_menu import option_menu
from openai import AzureOpenAI
 
 
class Sidebar:
    @staticmethod
    def render():
        with st.sidebar:
            selected = option_menu(
                "Menu",
                ["Chat", "Impostazioni"],
                menu_icon="umbrella",
                icons=["house", "gear"]
            )
        return selected
 
 
class SettingsPage:
    @staticmethod
    def render():
        st.title("Inserisci le tue informazioni")
        st.divider()
        st.session_state["api_key"] = st.text_input(
            "Inserisci la tua key",
            value=st.session_state.get("api_key", "")
        )
        st.session_state["endpoint"] = st.text_input(
            "Inserisci il tuo endpoint",
            value=st.session_state.get("endpoint", "")
        )
        st.session_state["api_version"] = st.text_input(
            "Inserisci la tua versione API",
            value=st.session_state.get("api_version", "")
        )
 
 
class ChatPage:
    @staticmethod
    def render():
        st.title("ChatGPT-like clone (Azure Streaming)")
 
        # Recupera i dati salvati dall'utente
        api_key = st.session_state.get("api_key")
        endpoint = st.session_state.get("endpoint")
        api_version = st.session_state.get("api_version")
 
        # Se mancano credenziali -> avvisa l’utente
        if not api_key or not endpoint or not api_version:
            st.warning("⚠️ Inserisci le credenziali nella pagina 'Impostazioni' prima di usare la chat.")
            return
 
        # Inizializza client
        client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version
        )
 
        DEPLOYMENT_NAME = "gpt-4o"
 
        if "messages" not in st.session_state:
            st.session_state.messages = []
 
        # Mostra messaggi precedenti
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
#####################################################################################################
        # --- funzione con retry ---
        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            before=before_log(logging.getLogger(), logging.INFO),   # log prima del tentativo
            after=after_log(logging.getLogger(), logging.INFO)      # log dopo il tentativo
            )
        def do_something_unreliable():
            raise Exception("Errore simulato!")
        def get_stream_response(client, DEPLOYMENT_NAME, messages):
            st.info("Tentativo in corso...")
            return client.chat.completions.create(
                model=DEPLOYMENT_NAME,
                messages=messages,
        stream=True,
    )

######################################################################################################

        # Input utente
        if prompt := st.chat_input("Scrivi qualcosa..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
 
            # Stream della risposta
            with st.chat_message("assistant"):
                placeholder = st.empty()
                full_response = ""
 
                stream = client.chat.completions.create(
                    model=DEPLOYMENT_NAME,
                    messages=st.session_state.messages,
                    stream=True,
                )
 
                for chunk in stream:
                    if len(chunk.choices) > 0 and chunk.choices[0].delta.content is not None:
                        full_response += chunk.choices[0].delta.content
                        placeholder.markdown(full_response + "▌")  # cursore
 
                placeholder.markdown(full_response)  # rimuove il cursore finale
 
            st.session_state.messages.append({"role": "assistant", "content": full_response})
 
 
 
 
st.set_page_config(page_title="Menu", layout="wide", initial_sidebar_state="expanded")
 
# Render sidebar
selected_page = Sidebar.render()
 
# Render selected page
if selected_page == "Chat":
    ChatPage.render()
elif selected_page == "Impostazioni":
    SettingsPage.render()