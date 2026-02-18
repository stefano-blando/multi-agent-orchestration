"""
Demo Streamlit — da completare DOPO che la pipeline funziona.
Priorità bassa: prima fai girare agent.py da CLI.

Avvio: streamlit run src/app.py
"""

import streamlit as st
from src.agent import run_pipeline

st.set_page_config(page_title="Hackapizza 2.0", page_icon="🍕", layout="centered")

st.title("🍕 Hackapizza 2.0 — Agent Edition")
st.caption("Multi-agent RAG system powered by datapizza-ai")

query = st.text_input("Inserisci la tua query:", placeholder="Es: Quali piatti sono adatti per...")

if st.button("Cerca", type="primary") and query:
    with st.spinner("Agenti al lavoro..."):
        try:
            result = run_pipeline(query)
            st.success("Risposta:")
            st.write(result)
        except Exception as e:
            st.error(f"Errore: {e}")
