# Copilot Instructions for agente-eda-streamlit

## Project Overview
This is a Streamlit-based data analysis agent that integrates Google Generative AI (Gemini) via the `langchain-google-genai` and `langchain_experimental` packages. The main entry point is `app.py`, which loads secrets from `.streamlit/secrets.toml` and expects a valid `GOOGLE_API_KEY` for AI features.

## Architecture & Data Flow
- **Main UI:** Implemented in `app.py` using Streamlit widgets for user interaction and data upload.
- **AI Agent:** Created via `create_pandas_dataframe_agent` from `langchain_experimental.agents.agent_toolkits`, using a Gemini LLM (`ChatGoogleGenerativeAI`).
- **Secrets Management:** API keys are loaded from `.streamlit/secrets.toml` using `st.secrets`.
- **Data:** User uploads a CSV, which is loaded into a Pandas DataFrame and passed to the agent for analysis and Q&A.

## Developer Workflows
- **Install dependencies:**
  ```powershell
  pip install -r requirements.txt
  ```
- **Run app locally:**
  ```powershell
  streamlit run app.py
  ```
  If `streamlit` is not recognized, use:
  ```powershell
  python -m streamlit run app.py
  ```
- **Update requirements:** Edit `requirements.txt` and commit changes. If Git push is rejected, run `git pull` first to sync.

## Error Handling & Patterns
- Missing API key: The app checks for `GOOGLE_API_KEY` and shows a Streamlit error if not found.
- Missing dependencies: Errors like `ModuleNotFoundError` are resolved by installing the required package (see `requirements.txt`).
- Model errors: If Gemini model is not found, check the model name and API version. Use `gemini-pro-vision` or list available models via API.

## External Integrations
- **Google Generative AI:** Requires a valid API key in `.streamlit/secrets.toml`.
- **LangChain:** Used for agent creation and LLM integration.

## Model discovery and configuration

- The app attempts to initialize the Gemini model name from (in order):
  1. `st.secrets["GOOGLE_MODEL"]` (recommended for deployments)
  2. environment variable `GOOGLE_MODEL`
  3. a list of sensible fallbacks hard-coded in `app.py` (e.g. `gemini-1.5-pro-latest`, `gemini-pro-vision`, `gemini-pro`).

- If you see errors like: `404 models/XXX is not found for API version v1beta`, it means your API key does not support that model name or the model name is invalid. To resolve:
  - Open the Google Cloud Console (AI / Generative AI Studio) and look for the Models page to see which models your project/key supports.
  - Or set a known-supported model name in `.streamlit/secrets.toml`:

```toml
# .streamlit/secrets.toml (do NOT commit secrets with real keys to public repos)
GOOGLE_API_KEY = "your-google-api-key"
GOOGLE_MODEL = "gemini-pro-vision"
```

- As an alternative for local testing you can set the env var in PowerShell before running Streamlit:

```powershell
$env:GOOGLE_MODEL = "gemini-pro-vision"
python -m streamlit run app.py
```

- The app will show an informative error and the list of model names it tried if initialization fails. Use that output to pick a supported model and add it to secrets or env.

## Conventions & Patterns
- All secrets are managed via `.streamlit/secrets.toml`.
- Data is always loaded as a Pandas DataFrame before passing to the agent.
- Error messages are surfaced to the user via `st.error`.

## Notes for agents working on this repo

- `app.py` contains the main logic and demonstrates the pattern for:
  - reading secrets via `st.secrets`,
  - allowing env overrides,
  - trying multiple model names and surfacing detailed errors to the UI.
- When changing model behavior, update both `app.py` and this instruction file with the supported model names and any new required permissions.

## Key Files
- `app.py`: Main application logic and UI.
- `.streamlit/secrets.toml`: Stores API keys/secrets.
- `requirements.txt`: Python dependencies.

## Example: Minimal secrets.toml
```toml
GOOGLE_API_KEY = "your-google-api-key"
```

---
For unclear workflows or missing conventions, ask the user for clarification and update this file accordingly.
