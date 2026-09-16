from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_pdf(filepath):
    """PDF load karta hai, agar fail ho to clear error deta hai"""
    try:
        loader = PyMuPDFLoader(filepath)
        pages = loader.load()
        
        # Check karein k actually text nikla hai ya khali hai
        total_text = sum(len(page.page_content.strip()) for page in pages)
        if total_text == 0:
            print(f"⚠️ Warning: '{filepath}' se koi text nahi nikla. Ye scanned/image PDF ho sakti hai.")
            return None
        
        return pages
    except Exception as e:
        print(f"❌ Error: '{filepath}' ko parh nahi paya. Reason: {e}")
        return None

# Use karein
pages = load_pdf("computer_science.pdf")
if pages:
    print(f"✅ {len(pages)} pages successfully load hue")
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(pages)
    print(f"✅ {len(chunks)} chunks bane")
else:
    print("Is file k sath aage nahi badh sakte.")