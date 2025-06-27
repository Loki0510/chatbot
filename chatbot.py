import os
import streamlit as st
import boto3
import json
import speech_recognition as sr
import threading
import pyttsx3
from deep_translator import GoogleTranslator
from textblob import TextBlob
import plotly.express as px
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter

# === Load from environment variables (EC2 recommended) ===
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID", "")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY", "")
os.environ["AWS_DEFAULT_REGION"] = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")

st.set_page_config(page_title="🧠 Voice + RAG + Bedrock", layout="wide")
st.title("🧠 Voice + RAG + Bedrock Model Comparator")
st.caption("Supports Voice, Website RAG, Bedrock Model Comparison")

# === TTS ===
def speak(text):
    def _speak():
        try:
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
        except RuntimeError:
            pass
    threading.Thread(target=_speak).start()


def correct_spelling(text):
    return str(TextBlob(text).correct())

def scrape_website_recursive(base_url, depth=1, visited=None):
    if visited is None:
        visited = set()
    if depth == 0 or base_url in visited:
        return ""
    try:
        visited.add(base_url)
        response = requests.get(base_url, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = ' '.join(soup.stripped_strings)
        links = [urljoin(base_url, a.get('href')) for a in soup.find_all('a', href=True)]
        links = [link for link in links if urlparse(link).netloc == urlparse(base_url).netloc]
        for link in links[:5]:
            text += "\n" + scrape_website_recursive(link, depth - 1, visited)
        return text
    except Exception as e:
        st.error(f"Error scraping {base_url}: {e}")
        return ""

@st.cache_resource
def create_retriever(text_data):
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = splitter.split_text(text_data)
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_texts(texts, embedding=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 4})
    return retriever, embeddings, texts

def get_chatgpt_response(query):
    llm = ChatOpenAI(model="gpt-4o-mini")
    return llm.invoke(query)

def plot_vector_embeddings_interactive(texts, embeddings):
    vectors = embeddings.embed_documents(texts[:50])
    vectors = np.array(vectors)
    min_dim = min(vectors.shape[0], vectors.shape[1])
    if min_dim < 2:
        st.warning("❗ Not enough data for PCA visualization.")
        return
    pca = PCA(n_components=min(3, min_dim))
    reduced = pca.fit_transform(vectors)
    df = pd.DataFrame(reduced, columns=[f"PCA {i+1}" for i in range(reduced.shape[1])])
    df["Question"] = [t.strip().replace("\n", " ")[:100] + "..." for t in texts[:50]]
    if reduced.shape[1] == 3:
        fig = px.scatter_3d(df, x="PCA 1", y="PCA 2", z="PCA 3", hover_data=["Question"])
    else:
        fig = px.scatter(df, x="PCA 1", y="PCA 2", hover_data=["Question"])
    fig.update_traces(marker=dict(size=5, color='blue'))
    st.plotly_chart(fig, use_container_width=True)

def format_prompt(model_id, prompt):
    if model_id.startswith("anthropic."):
        return json.dumps({
            "messages": [{"role": "user", "content": prompt}],
            "anthropic_version": "bedrock-2023-05-31",
        })
    elif model_id.startswith("meta."):
        return json.dumps({
            "prompt": prompt,
            "temperature": 0.7,
            "top_p": 0.9,
            "max_gen_len": 512
        })
    else:
        return json.dumps({
            "prompt": prompt,
            "temperature": 0.7,
            "max_tokens": 512
        })

client = boto3.client("bedrock-runtime", region_name="us-east-1")
bedrock_models = {
    "Mistral 7B": "mistral.mistral-7b-instruct-v0:2",
    "LLaMA3-8B": "meta.llama3-8b-instruct-v1:0",
    "LLaMA3-70B": "meta.llama3-70b-instruct-v1:0"
}
selected_models = st.multiselect("🧠 Select Bedrock models", list(bedrock_models.keys()), default=["Mistral 7B", "LLaMA3-8B"])

# === Sidebar Website RAG ===
st.sidebar.markdown("## 🌐 Web RAG Source")
web_url = st.sidebar.text_input("Enter website URL to use as RAG source")
retriever = None
if web_url:
    scraped_data = scrape_website_recursive(web_url, depth=2)
    if scraped_data:
        retriever, embeddings, embedded_texts = create_retriever(scraped_data)
        qa_chain = RetrievalQA.from_chain_type(llm=ChatOpenAI(model="gpt-4o-mini"), retriever=retriever)
        st.sidebar.success("✅ Website content loaded.")
        with st.sidebar.expander("📄 Website Preview"):
            st.text_area("Scraped Content", scraped_data[:2000], height=300)
    if st.sidebar.button("📊 Show FAISS Vectors in 3D"):
        plot_vector_embeddings_interactive(embedded_texts, embeddings)

# === User Input ===
col1, col2 = st.columns([6, 1])
with col1:
    user_input = st.text_input("Type your message or click mic...")
with col2:
    use_voice = st.button("🎙️ Voice")

enable_tts = st.checkbox("🔊 Enable Voice Response", value=False)
#if use_voice:
    #user_input = get_voice_input()

if user_input:
    translated = GoogleTranslator(source='auto', target='en').translate(user_input)
    corrected = correct_spelling(translated) if len(translated.split()) <= 2 else translated
    corrected = corrected[:1600]

    if retriever:
        rag_response = qa_chain.invoke({"query": corrected})
        rag_text = rag_response.get("result", "") if isinstance(rag_response, dict) else rag_response
        fallback = any(p in rag_text.lower() for p in ["no info", "not found", "don't know"])
        final_response = rag_text if rag_text.strip() and not fallback else get_chatgpt_response(corrected).content
    else:
        final_response = get_chatgpt_response(corrected).content

    st.markdown("### 🧠 Base Response (OpenAI):")
    st.info(final_response)
    if enable_tts:
        speak(final_response)

    st.markdown("## 🔍 Bedrock Model Responses")
    for model in selected_models:
        try:
            model_id = bedrock_models[model]
            bedrock_prompt = f"Context: {rag_text}\n\nUser Query: {corrected}" if retriever else corrected
            body = format_prompt(model_id, bedrock_prompt)

            response = client.invoke_model(
                modelId=model_id,
                body=body.encode('utf-8'),
                accept="application/json",
                contentType="application/json"
            )
            result = json.loads(response['body'].read())

            with st.expander(f"🧠 {model} Response"):
                final_text = ""

                if isinstance(result, dict):
                    if "outputs" in result and isinstance(result["outputs"], list):
                        final_text = result["outputs"][0].get("text", "").strip()
                    elif "generation" in result:
                        final_text = result["generation"].strip()
                    elif "output" in result:
                        final_text = result["output"].strip()
                    else:
                        final_text = json.dumps(result, indent=2)
                else:
                    final_text = str(result)

                final_text = final_text.strip().rstrip(".1234567890")
                st.info(final_text)

        except Exception as e:
            st.error(f"{model} failed: {e}")
