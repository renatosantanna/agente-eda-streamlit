# app.py (versão final com seletor de modelos)
import streamlit as st
import pandas as pd
import os
import time
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from langchain_core.agents import AgentAction

# --- Configuração da Página ---
st.set_page_config(page_title="Agente de Análise de CSV", layout="wide")
st.title("🤖 Agente Autônomo para Análise de Dados (E.D.A.)")
st.write("Faça o upload de um arquivo CSV e escolha um modelo de IA na barra lateral para começar.")

# --- BARRA LATERAL COM OPÇÕES ---
with st.sidebar:
    st.header("Configurações do Agente")
    
    # Lista de modelos selecionados
    model_options = [
        "models/gemini-2.5-pro",
        "models/gemini-2.5-flash",
        "models/gemini-pro-latest",
        "models/gemini-flash-latest",
        "models/gemma-3-27b-it",
    ]
    
    # Seletor de modelo
    selected_model = st.selectbox("Escolha o Modelo de IA:", model_options)
    st.info(f"Modelo selecionado: `{selected_model}`")

uploaded_file = st.file_uploader("Selecione seu arquivo CSV", type="csv")

if uploaded_file is not None:
    try:
        dataframe = pd.read_csv(uploaded_file)
        st.subheader("Amostra dos Dados Carregados")
        st.dataframe(dataframe.head())
        st.divider()

        def get_api_key():
            try:
                return st.secrets["GOOGLE_API_KEY"]
            except (KeyError, FileNotFoundError):
                st.error("Chave de API do Google não encontrada. Configure o segredo 'GOOGLE_API_KEY'.")
                return None

        # A função agora recebe o nome do modelo como argumento
        def criar_agente(df, api_key, model_name):
            if not api_key: return None
            try:
                llm = ChatGoogleGenerativeAI(
                    model=model_name, # Usa o modelo selecionado
                    temperature=0,
                    google_api_key=api_key,
                    safety_settings={
                        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    },
                )
                
                agent = create_pandas_dataframe_agent(
                    llm,
                    df,
                    verbose=True,
                    agent_executor_kwargs={"handle_parsing_errors": True},
                    allow_dangerous_code=True,
                    return_intermediate_steps=True
                )
                return agent
            except Exception as e:
                st.error(f"Erro ao inicializar o agente de IA: {e}")
                return None

        google_api_key = get_api_key()
        if google_api_key:
            # Passa o modelo selecionado da barra lateral para a função que cria o agente
            agente = criar_agente(dataframe, google_api_key, selected_model)

            if "messages" not in st.session_state:
                st.session_state[uploaded_file.name] = []
            
            messages = st.session_state[uploaded_file.name]

            for message in messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            if prompt := st.chat_input("Faça sua pergunta sobre o arquivo..."):
                messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    with st.status("O agente está pensando e executando...", expanded=True) as status:
                        resposta_agente = ""
                        try:
                            historico_formatado = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
                            contexto_prompt = f"Considerando o histórico da conversa:\n{historico_formatado}\n\nResponda à nova pergunta: {prompt}"
                            
                            resposta = agente.invoke({"input": contexto_prompt})
                            
                            resposta_agente = resposta.get("output", "Não consegui processar a resposta.")
                            intermediate_steps = resposta.get("intermediate_steps", [])

                            status.update(label="Análise Concluída!", state="complete", expanded=False)
                            
                            st.markdown(resposta_agente)
                            
                            if intermediate_steps:
                                with st.expander("Ver Processamento do Backend (Log Detalhado)"):
                                    log_string = ""
                                    for action, observation in intermediate_steps:
                                        log_string += f"**Pensamento:**\n{action.log}\n\n"
                                        log_string += f"**Ação:** `{action.tool}`\n"
                                        log_string += f"**Input da Ação:**\n```python\n{action.tool_input}\n```\n\n"
                                        log_string += f"**Observação:**\n```text\n{observation}\n```\n---\n"
                                    st.markdown(log_string)

                        except Exception as e:
                            st.error(f"Ocorreu um erro ao chamar o agente: {e}")
                            resposta_agente = f"Erro: {e}"
                            status.update(label="Erro!", state="error")
                
                messages.append({"role": "assistant", "content": resposta_agente})

    except Exception as e:
        st.error(f"Ocorreu um erro ao processar o arquivo CSV. Verifique o formato do arquivo. Detalhes: {e}")