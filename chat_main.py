# Requirements:
# pip install fastapi uvicorn langchain openai python-dotenv pydantic pydantic-core tiktoken regex

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import os

from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationChain

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set in environment")

# Initialize the LLM with LangChain
llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    openai_api_key=openai_api_key,
    temperature=0.7
)

# System prompt with static guidelines
system_prompt = """
You are an AI chatbot who serves as the user’s conversational companion and friend. Always speak politely (using honorifics), maintain a warm, friendly tone, and respond exclusively in Korean.
- Show empathy and genuine interest in what the user says.
- Ask follow‑up questions to guide the conversation naturally.
- Use short, clear, yet gentle expressions.
- Never mention that you are an AI.
- Avoid unnecessary technical jargon; respond in a comfortable, easy‑going manner.
"""

# Build the prompt template
chat_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_prompt),
    SystemMessagePromptTemplate.from_template("대화 요약: {chat_history}"),
    HumanMessagePromptTemplate.from_template("{input}")
])

# Set up summary buffer memory
memory = ConversationSummaryBufferMemory(
    llm=llm,
    memory_key="chat_history",
    input_key="input",
    output_key="response"
)

# Create the conversational chain
conversation = ConversationChain(
    llm=llm,
    prompt=chat_prompt,
    memory=memory,
    verbose=False
)

# FastAPI app setup
app = FastAPI(title="유음 대화 연습 챗봇 API")

class ChatRequest(BaseModel):
    input: str

class ChatResponse(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    """
    POST /chat
    Request JSON: { "input": "사용자 입력 문자열" }
    Returns assistant response using LangChain conversation chain.
    """
    try:
        reply = conversation.predict(input=req.input)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return ChatResponse(response=reply)

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "ok"}
