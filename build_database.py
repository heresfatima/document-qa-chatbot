from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Step 1: PDF load karein
loader = PyMuPDFLoader("computer_science.pdf")
pages = loader.load()
print(f"✅ {len(pages)} pages load hue")

# Step 2: Chunks banayein
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(pages)
print(f"✅ {len(chunks)} chunks bane")

# Step 3: Embedding model load karein (ye local chalta hai, free hai)
print("Embedding model load ho raha hai... (pehli dafa thora time lagega)")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Step 4: Chunks ko embeddings mn convert kr k ChromaDB mn store karein
print("Embeddings ban rahe hain aur database mn store ho rahe hain...")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"  # yahan database save hoga
)

print("✅ Database ban gaya! 'chroma_db' folder mn save ho gaya.")