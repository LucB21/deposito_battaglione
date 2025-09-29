import spacy
from spacy import displacy

# Carica il modello italiano
nlp = spacy.load("it_core_news_sm")

text = """
Il 15 marzo 2023, il presidente Sergio Mattarella ha incontrato a Roma il CEO di Microsoft, Satya Nadella, 
insieme al commissario europeo Thierry Breton. Durante la riunione, svoltasi a Palazzo Chigi, 
si è discusso di investimenti tecnologici per oltre 1 miliardo di euro, con particolare attenzione 
allo sviluppo dell'intelligenza artificiale in Italia e in Europa. Inoltre, è stato annunciato che 
nel 2025 aprirà un nuovo centro di ricerca a Milano.
"""

doc = nlp(text)

# Output analisi
for token in doc:
    print(f"Token: {token.text:12} | Lemma: {token.lemma_:12} | POS: {token.pos_:10} | Dipendenza: {token.dep_:10} | Head: {token.head.text}")


# === Visualizza dipendenze sintattiche ===
print("Apro la visualizzazione delle dipendenze sintattiche...")
displacy.serve(doc, style="ent",
port=8654, host="127.0.0.1")

print("aaaaaaaaaaaaaaa")