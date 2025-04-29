# Vancomycin TDM App

This is a Streamlit app combining vancomycin PK calculations with RAG-guided LLM interpretation.

## Deploy
1. Fork or clone this repo
2. On Streamlit Cloud (or any host), add:
   - `OPENAI_API_KEY` as a secret
3. Point Streamlit to `vancomycin_tdm_app.py`
4. Ensure your `requirements.txt` is installed automatically

## Local run
```bash
pip install -r requirements.txt
streamlit run vancomycin_tdm_app.py
