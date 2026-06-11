 import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from openai import OpenAI
import os

# 🔐 OpenAI (Streamlit Secrets)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(page_title="Estatística+", layout="centered")

st.title("📊 Estatística+")
st.caption("Assistente de estatística com IA 🤖")

# ------------------------
# 📥 DADOS
# ------------------------
dados_input = st.text_area(
    "Insere os dados (separados por vírgula)",
    "12, 15, 18, 20, 22"
)

# ------------------------
# 🤖 FUNÇÃO IA
# ------------------------
def responder_ia(mensagem):
    resposta = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "És um assistente de estatística que explica tudo de forma simples em português de Angola/Portugal."
            },
            {"role": "user", "content": mensagem}
        ]
    )
    return resposta.choices[0].message.content

# ------------------------
# 📊 ANÁLISE ESTATÍSTICA
# ------------------------
if st.button("Analisar dados"):

    dados = np.array([float(x.strip()) for x in dados_input.split(",")])

    media = np.mean(dados)
    mediana = np.median(dados)
    moda = stats.mode(dados, keepdims=True).mode[0]
    variancia = np.var(dados)
    desvio = np.std(dados)

    st.subheader("📊 Resultados")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Média", f"{media:.2f}")
        st.metric("Mediana", f"{mediana:.2f}")

    with col2:
        st.metric("Desvio padrão", f"{desvio:.2f}")
        st.metric("Variância", f"{variancia:.2f}")

    st.metric("Moda", moda)

    # 📈 GRÁFICO
    st.subheader("📈 Gráfico de Distribuição")

    fig, ax = plt.subplots()
    ax.hist(dados, bins=5)
    st.pyplot(fig)

# ------------------------
# 💬 CHAT IA
# ------------------------
st.divider()
st.subheader("💬 Chat com IA")

if "messages" not in st.session_state:
    st.session_state.messages = []

# histórico
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# input
user_input = st.chat_input("Fala com a IA sobre estatística...")

if user_input:

    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.write(user_input)

    resposta = responder_ia(user_input)

    st.session_state.messages.append({"role": "assistant", "content": resposta})

    with st.chat_message("assistant"):
        st.write(resposta)
