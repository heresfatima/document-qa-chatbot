from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

# Step 1: Purana database load karein (dobara banane ki zaroorat nahi)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# Step 2: LLM setup karein
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.getenv("GROQ_API_KEY")
)

# Step 3: User se sawal lein
question = input("Ask anything from pdf: ")

# Step 4: Relevant chunks dhoondein (top 3 sabse related)
relevant_chunks = vectorstore.similarity_search(question, k=3)

# Step 5: Un chunks ko ek context string mn jorein
context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])

prompt = f"""Neeche diye gaye context ka istemal kr k sawal ka jawab dein.
Jawab hamesha ENGLISH mn dein.
Agar context mn jawab na mile, to bata dein "Ye jawab document mn nahi mila".

Context:
{context}

Sawal: {question}

Jawab (English mn):"""

response = llm.invoke(prompt)
print("\n--- Jawab ---")
print(response.content)