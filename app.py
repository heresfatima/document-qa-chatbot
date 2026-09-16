import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import tempfile

load_dotenv()

st.title("📄 Document Q&A Chatbot")
st.write("Upload your PDF and ask relevant questions!")

# File upload karne ka box
uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file is not None:
    # Uploaded file ko temporarily save karein taake loader use kr sake
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # Sirf ek dafa process karein (baar baar nahi, jab tak file change na ho)
    if "processed_file" not in st.session_state or st.session_state.processed_file != uploaded_file.name:
        with st.spinner("Processing..."):
            loader = PyMuPDFLoader(tmp_path)
            pages = loader.load()

            splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
            chunks = splitter.split_documents(pages)

            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)

            st.session_state.vectorstore = vectorstore
            st.session_state.processed_file = uploaded_file.name

        st.success(f"✅ PDF is ready! {len(chunks)} chunks made.")

    # Sawal poochne ka box
    question = st.text_input("Ask anything from pdf:")

    if question:
        with st.spinner("Searching answer..."):
            llm = ChatGroq(model="openai/gpt-oss-120b", api_key=os.getenv("GROQ_API_KEY"))

            relevant_chunks = st.session_state.vectorstore.similarity_search(question, k=10)
            context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])

            prompt = f"""Neeche diye gaye context ka istemal kr k sawal ka jawab dein.

Zaroori Instructions:
- Jawab SIRF isi context se aana chahiye, bahar ki knowledge use na karein
- Agar sawal ka jawab dene k liye multiple sections/topics ki info combine karni pare, to bilkul combine karein — ye acha hai jab genuinely related ho
- Lekin: har fact ko sirf uske ASAL topic k sath hi use karein. Kisi aik cheez (jaise Narrow AI k examples) ko doosri cheez (jaise General AI) ka example mat banayein, sirf is liye k wo pass mn likhi thi
- Agar context clearly kehta hai koi cheez "theoretical hai", "exist nahi karti", ya "abhi nahi di gayi", to yehi cheez honestly bata dein us specific part k liye — khud se examples mat banayein sirf jawab complete dikhane k liye
- Jawab ko apne alfaz mn clearly explain karein, information ko samjha k complete aur meaningful tareeqe se present karein
- Agar sawal k kisi hisse ka jawab context mn nahi milta, saaf bata dein k wo specific info document mn nahi di gayi — baqi sawal ka jawab phir bhi dein
- Jawab hamesha ENGLISH mn dein

Context:
{context}

Question: {question}

Answer (complete accurate):"""

            response = llm.invoke(prompt)

        st.write("### Answer:")
        st.write(response.content)