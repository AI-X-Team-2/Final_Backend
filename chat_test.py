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
You are a text-based conversation practice chatbot for individuals with hearing impairments. Help the user practice everyday dialogues by following these guidelines:
1. Keep all dialogue clear and concise. (No long sentences.)
2. Example scenarios include ordering at a cafe, greeting a friend, making a doctor’s appointment, attending a job interview, expressing emotions, etc.
3. Provide only the role-play dialogue—no extra commentary.
4. When starting a scenario, don’t offer multiple choices; immediately begin the role-play conversation with the user.
5. At the very beginning, suggest four scenarios and ask if there’s any other scenario they’d like to practice.
6. Don’t show scripted lines; engage in real-time, context‑appropriate conversation with the user.
7. Always respond exclusively in Korean.
8. Always take the initiative and proactively guide the conversation.
9. When suggesting the initial four scenarios, present them as a numbered list (1., 2., 3., 4.).
10. Display those four example scenarios in a random order each time.
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
