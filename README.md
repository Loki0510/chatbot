
### 📘 `README.md`

````markdown
# 🧠 Voice + RAG + Bedrock Model Comparator

This Streamlit app allows users to interact with OpenAI and Amazon Bedrock language models using:

- 🔊 Voice input (via microphone or uploaded audio file)
- 🌐 Website scraping with FAISS-powered Retrieval-Augmented Generation (RAG)
- 🤖 Multi-model Bedrock inference (Mistral, LLaMA3, Claude, etc.)
- 🧠 Multilingual translation, spelling correction, and PCA visualization of text embeddings

---

## 🚀 Features

- **Voice Input:** Supports microphone input or uploading `.wav`, `.mp3`, or `.m4a` files
- **RAG with Website Scraping:** Extracts content from websites, indexes using FAISS, and performs question-answering
- **Bedrock Model Comparison:** Sends prompts to multiple Amazon Bedrock models and compares outputs
- **TTS Output:** Speaks out the final response using `pyttsx3`
- **Multilingual Support:** Translates voice or text queries to English
- **Vector Visualization:** Projects top 50 text chunks into 2D/3D using PCA + Plotly

---

## 🛠️ Requirements

```bash
pip install -r requirements.txt
````

### Required Python Packages:

* streamlit
* boto3
* speechrecognition
* pyttsx3
* deep-translator
* textblob
* pydub
* openai / langchain / faiss-cpu
* scikit-learn
* plotly
* beautifulsoup4
* requests
* numpy
* pandas

### System Requirements

* **FFmpeg:** Required by `pydub` to convert `.mp3` or `.m4a` to `.wav`

  * Ubuntu: `sudo apt install ffmpeg`
  * Mac (Homebrew): `brew install ffmpeg`
  * Windows: Add FFmpeg to your PATH

---

## 🌐 Deployment

### ✅ Local Machine (Recommended)

```bash
streamlit run app.py
```

### ⚠️ Streamlit Cloud

* Only `.wav` upload is supported (no FFmpeg support)
* Voice mic input will **not** work

### ✅ EC2 / Replit / VPS

* Full support for `.mp3`, `.m4a`, `.wav`
* Supports microphone and TTS if configured with audio permissions

---

## 🔐 Environment Variables

Add the following to your `.streamlit/secrets.toml` or set as environment variables:

```toml
AWS_ACCESS_KEY_ID = "your_aws_key"
AWS_SECRET_ACCESS_KEY = "your_aws_secret"
AWS_DEFAULT_REGION = "us-east-1"
OPENAI_API_KEY = "your_openai_key"
```

---

## 📁 File Upload Notes

* `.wav` preferred
* `.mp3` and `.m4a` require FFmpeg
* Audio is converted to PCM WAV (mono, 16-bit) for compatibility with `speech_recognition`

---

## 📊 Model Comparison

Supported Bedrock models:

* `Mistral 7B`
* `LLaMA3-8B`
* `LLaMA3-70B`

---

## 🧪 Example Use Cases

* Ask a question using your voice or text
* Scrape a university website and ask questions about departments
* Upload a podcast clip and summarize it with Bedrock models
* Compare OpenAI GPT-4o-mini vs. Bedrock model outputs

---

## 📎 Known Issues

* Mic input does not work on Streamlit Cloud
* `.mp3/.m4a` upload will fail if FFmpeg is missing
* Large websites may slow down FAISS indexing

---

## 📞 Contact

Built by \[Lokesh Vinnakota]
For inquiries, reach out via GitHub 

```

