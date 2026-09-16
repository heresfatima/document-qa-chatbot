import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import tempfile
import uuid

load_dotenv()

st.title("📄 Document Q&A Chatbot")
st.write("Upload your PDF and ask relevant questions!")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    if "processed_file" not in st.session_state or st.session_state.processed_file != uploaded_file.name:
        with st.spinner("Processing..."):
            loader = PyMuPDFLoader(tmp_path)
            pages = loader.load()

            splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
            chunks = splitter.split_documents(pages)

            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            if "processed_file" not in st.session_state or st.session_state.processed_file != uploaded_file.name:
                with st.spinner("Processing..."):
                    loader = PyMuPDFLoader(tmp_path)
                    pages = loader.load()

                splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
                chunks = splitter.split_documents(pages)

                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

                # Har upload k lie naya, unique collection name
                collection_name = f"doc_{uuid.uuid4().hex[:8]}"
                vectorstore = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    collection_name=collection_name
                )

                st.session_state.vectorstore = vectorstore
                st.session_state.processed_file = uploaded_file.name

            st.session_state.vectorstore = vectorstore
            st.session_state.processed_file = uploaded_file.name

        st.success(f"✅ PDF is ready!")

    question = st.text_input("Ask anything from pdf:")

    answer_length = st.radio("Answer style:", ["Short", "Long"], horizontal=True)

    if question:
        with st.spinner("Searching answer..."):
            llm = ChatGroq(model="openai/gpt-oss-120b", api_key=os.getenv("GROQ_API_KEY"))

            relevant_chunks = st.session_state.vectorstore.similarity_search(question, k=10)
            context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])

            if answer_length == "Short":
                length_instruction = """Jawab CHOTA rakhein (2-4 sentences ka mazmoon), lekin isay ASAN aur QUICKLY SCANNABLE banayein:
            - Zaroori keywords ya terms ko **bold** karein (double asterisks k sath)
            - Agar jawab mn multiple points/items hon (jese types, examples, steps), to unhein bullet list ki tarah likhein, har point new line pr "- " se shuru kr k
            - Har bullet chota aur to-the-point ho, lambi sentence na ho
            - Total content chota hi rahe — sirf presentation clear honi chahiye
            - Agar sawal mn do ya zyada cheezon ka COMPARISON ya DIFFERENCE poocha gaya ho, to jawab ek Markdown TABLE ki soorat mn dein (| Column | Column | wali formatting), taake dono cheezein side-by-side saaf nazar aayen"""
            else:
                length_instruction = """Jawab DETAILED dein 
                — poori tarha explain karein, examples aur context k sath, taake reader ko gehri samajh aa jaye." \
                - Zaroori keywords ya terms ko **bold** karein (double asterisks k sath)
                - Agar sawal mn COMPARISON ya DIFFERENCE poocha gaya ho, or agr document mn koi sawal already Markdown TABLE mn ho to usko bhi jawab mn show krte hue Markdown TABLE use karein (| Column | Column | format mn) taake points clearly side-by-side dikhein"""

            prompt = f"""Neeche diye gaye context ka istemal kr k sawal ka jawab dein.

Zaroori Instructions:
- Jawab SIRF isi context se aana chahiye, bahar ki knowledge use na karein
- Agar sawal ka jawab dene k liye multiple sections/topics ki info combine karni pare, to bilkul combine karein — ye acha hai jab genuinely related ho
- Lekin: har fact ko sirf uske ASAL topic k sath hi use karein. Kisi aik cheez ko doosri cheez ka example mat banayein sirf is liye k wo pass mn likhi thi
- Agar context clearly kehta hai koi cheez "theoretical hai" ya "exist nahi karti", to yehi honestly bata dein
- {length_instruction}
- Sirf **bold** aur "- " bullet points allowed hain (jese upar length instruction mn bataya gaya) — koi HTML tags (jese <br>, <ul>) bilkul use na karein
- Agar sawal k kisi hisse ka jawab context mn nahi milta, saaf bata dein
- Jawab hamesha ENGLISH mn dein

Context:
{context}

Question: {question}

Answer:"""

            response = llm.invoke(prompt)

        st.write("### Answer:")
        st.write(response.content)