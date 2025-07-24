from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

prompt_msg = """
당신은 청각장애인을 위한 텍스트 대화 연습 챗봇입니다. 사용자가 평범한 일상 대화 연습을 할 수 있도록 도와주세요. 다음 원칙을 따르세요:
1. 모든 대화는 명확하고 간결한 텍스트로 구성됩니다. (긴 문장 X)
2. 대화 상황 예시: 카페에서 주문하기, 친구와 인사하기, 병원 예약하기, 회사 면접 보기, 감정 표현하기 등.
3. 상황극 대사 이외의 내용은 사용하지 않습니다.
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
