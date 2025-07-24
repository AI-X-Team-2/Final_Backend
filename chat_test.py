from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

prompt_msg = """
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

message = [{'role':'system', 'content': prompt_msg}]

# GPT가 먼저 말을 꺼내도록 API 호출
response = client.chat.completions.create(
    model='gpt-4o-mini',
    messages=message  # 빈 메시지 → GPT가 첫 메시지 생성
    
)

chat_response = response.choices[0].message.content
print(f'ChatGPT : {chat_response}')
message.append({'role': 'assistant', 'content': chat_response})

while True:
    content = input('사용자 : ')
    if content == '종료':
        break

    message.append({'role': 'user', 'content': content})

    response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=message
    )

    chat_response = response.choices[0].message.content
    print(f'ChatGPT : {chat_response}')
    message.append({'role': 'assistant', 'content': chat_response})