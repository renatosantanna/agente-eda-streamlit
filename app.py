# app.py (versão corrigida e robusta)
import streamlit as st
import pandas as pd
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI

# --- Configuração da Página ---
# Esta parte é segura e pode ser executada no início.
st.set_page_config(page_title="Agente de Análise de CSV", layout="wide")
st.title("🤖 Agente Autônomo para Análise de Dados (E.D.A.)")
st.write("Faça o upload de um arquivo CSV abaixo para começar a interagir com seus dados.")

# --- Lógica Principal da Aplicação ---

# 1. Inicia a interface de forma segura, pedindo o arquivo ao usuário.
uploaded_file = st.file_uploader("Selecione seu arquivo CSV", type="csv")

# 2. TODA a lógica crítica SÓ ACONTECE DEPOIS que um arquivo for carregado.
if uploaded_file is not None:
    try:
        # Carrega o dataframe a partir do arquivo enviado
        dataframe = pd.read_csv(uploaded_file)
        st.subheader("Amostra dos Dados Carregados")
        st.dataframe(dataframe.head())
        st.divider()

        # Funções aninhadas para organização, serão chamadas apenas se o arquivo for válido.
        def get_api_key():
            try:
                return st.secrets["GOOGLE_API_KEY"]
            except (KeyError, FileNotFoundError):
                st.error("Chave de API do Google não encontrada. Configure o segredo 'GOOGLE_API_KEY' nas configurações da aplicação.")
                return None

        def criar_agente(df, api_key):
            if not api_key: return None
            try:
                llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0, google_api_key=api_key)
                agent = create_pandas_dataframe_agent(llm, df, verbose=True, agent_executor_kwargs={"handle_parsing_errors": True}, allow_dangerous_code=True)
                return agent
            except Exception as e:
                st.error(f"Erro ao inicializar o agente de IA: {e}")
                return None

        # 3. Pega a chave da API e cria o agente somente agora.
        google_api_key = get_api_key()
        if google_api_key:
            agente = criar_agente(dataframe, google_api_key)

            # Se o agente foi criado com sucesso, exibe a interface de chat.
            if agente:
                if "messages" not in st.session_state:
                    st.session_state.messages = []

                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

                if prompt := st.chat_input("Faça sua pergunta sobre o arquivo..."):
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.markdown(prompt)

                    with st.chat_message("assistant"):
                        with st.spinner("O agente está pensando... 🧠"):
                            # A lógica de processamento da pergunta permanece a mesma
                            historico_formatado = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages])
                            contexto_prompt = f"Considerando o histórico da conversa:\n{historico_formatado}\n\nResponda à nova pergunta: {prompt}"
                            resposta = agente.invoke({"input": contexto_prompt})
                            resposta_agente = resposta.get("output", "Não consegui processar a resposta.")
                            st.write(resposta_agente)
                            st.pyplot(st.session_state.get("last_fig", None))
                    
                    st.session_state.messages.append({"role": "assistant", "content": resposta_agente})

    except Exception as e:
        st.error(f"Ocorreu um erro ao processar o arquivo CSV. Verifique o formato do arquivo. Detalhes: {e}")