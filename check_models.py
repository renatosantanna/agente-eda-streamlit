import google.generativeai as genai
import os

# --- CONFIGURAÇÃO ---
# Cole sua chave de API aqui, entre as aspas.
GOOGLE_API_KEY = "AIzaSyCFuHbg9tsMpLNM7Xw6HqoiZV0KXT6rDF0"
# --------------------

try:
    genai.configure(api_key=GOOGLE_API_KEY)

    print("Buscando modelos disponíveis para sua chave de API...")
    print("-" * 30)

    model_found = False
    for model in genai.list_models():
        # 'generateContent' é o método que o LangChain usa, então filtramos por ele.
        if 'generateContent' in model.supported_generation_methods:
            print(f"Nome do Modelo: {model.name}")
            model_found = True

    if not model_found:
        print("Nenhum modelo compatível com 'generateContent' foi encontrado para esta chave de API.")

    print("-" * 30)

except Exception as e:
    print(f"\nOcorreu um erro: {e}")
    print("\nVerifique se sua chave de API é válida e se você tem permissão para usar a API Generative Language.")