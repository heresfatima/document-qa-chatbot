from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

# .env file se API key load karein
load_dotenv()

# Groq model se connection banayein
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.getenv("GROQ_API_KEY")
)

# Ek simple sawal puchein
response = llm.invoke("Assalam o Alaikum! Ek line mn bata dein AI kya hota hai.")
print(response.content)