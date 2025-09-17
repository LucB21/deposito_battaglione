from openai import AzureOpenAI
import streamlit as st

# Leggi i secrets
client = AzureOpenAI(
    api_key=st.secrets["AZURE_OPENAI_KEY"],
    azure_endpoint=st.secrets["AZURE_OPENAI_ENDPOINT"],
    api_version=st.secrets["AZURE_OPENAI_API_VERSION"]
)

# Deployment del modello embedding (lo stesso nome dato in fase di deploy su Azure)
embedding_model = st.secrets["AZURE_EMBEDDING_DEPLOYMENT"]

# Frasi di esempio
texts = [
    "Azure OpenAI è fantastico!",
    "Mi piace usare i modelli di embedding per NLP."
]

for txt in texts:
    response = client.embeddings.create(
        model=embedding_model,
        input=txt
    )
    vector = response.data[0].embedding
    print(f"Testo: {txt}")
    print(f"Dimensione embedding: {len(vector)}\n")
