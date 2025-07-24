import torch
import os
import shutil
import numpy as np
import difflib
import noisereduce as nr
import datetime
import random
import json
from openai import OpenAI
from transformers import pipeline
import soundfile as sf
from pydub import AudioSegment
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- FastAPI 앱 초기화 및 설정 ---
app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 폴더 설정 ---
TEMP_DIR = "temp_files"
os.makedirs(TEMP_DIR, exist_ok=True)

# --- 응답 모델 정의 --
class ConversationProcessResponse(BaseModel):
    raw_transcript: str

class PracticeSentenceResponse(BaseModel):
    sentence: str

class PronunciationCheckResponse(BaseModel):
    target_sentence: str
    user_transcript: str
    diff_details: list
    score: int
    positive_points: str
    areas_for_improvement: str
    overall_feedback: str


# --- 1. 핵심 로직 함수 (기존과 동일) ---
def reduce_noise(audio_data, sample_rate):
    """오디오 데이터에서 배경 소음을 제거합니다."""
    print("... 오디오 노이즈 제거를 시작합니다 ...")
    reduced_noise_audio = nr.reduce_noise(y=audio_data, sr=sample_rate, stationary=False)
    print("노이즈 제거 완료!")
    return reduced_noise_audio

def speech_to_text(audio_file_path):
    """Whisper 모델을 사용해 음성 파일을 텍스트로 변환합니다."""
    print(f"\n STT 시작: '{audio_file_path}'")
    transcriber = pipeline("automatic-speech-recognition", model="RecCode/whisper_final", torch_dtype=torch.float16, device_map="auto")
    result = transcriber(audio_file_path, generate_kwargs={"language": "korean"})
    print(f" STT 결과: {result['text'].strip()}")
    return result["text"].strip()

def evaluate_pronunciation_with_llm(target_sentence, user_transcript):
    """LLM을 사용하여 발음을 평가하고 점수 및 피드백을 생성합니다."""
    print(" LLM으로 발음 평가를 시작합니다...")
    try:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        system_prompt = """
        당신은 한국어 발음 교정을 전문으로 하는 언어 치료사입니다. '목표 발음'과 '사용자 발음'을 비교하여 평가를 수행합니다.
        특히 받침(종성)의 정확성, 이중 모음의 명확성, 거센소리(ㅋ,ㅌ,ㅍ,ㅊ)와 예사소리(ㄱ,ㄷ,ㅂ,ㅅ,ㅈ)의 구분 등을 중점적으로 분석해주세요.
        평가 결과는 반드시 다음의 JSON 형식으로만 반환해주세요. 다른 설명은 절대 추가하지 마세요.
        {
          "score": "0에서 100 사이의 정수 점수",
          "positive_points": "발음에서 잘한 점에 대한 긍정적인 피드백 (간단한 한 문장)",
          "areas_for_improvement": "개선이 필요한 부분에 대한 구체적인 피드백 (간단한 한 문장)",
          "overall_feedback": "종합적인 격려의 메시지 (한 문장)"
        }
        """
        user_prompt = f"정답 단어: \"{target_sentence}\"\n사용자 발음: \"{user_transcript}\""
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        
        evaluation_result = json.loads(response.choices[0].message.content)
        evaluation_result['score'] = int(evaluation_result.get('score', 0))
        print(" LLM 평가 완료!")
        return evaluation_result

    except Exception as e:
        print(f" LLM 평가 중 심각한 오류 발생: {e}")
        return {
            "score": 0,
            "positive_points": "AI 평가 중 오류가 발생했습니다.",
            "areas_for_improvement": "서버 로그를 확인해주세요.",
            "overall_feedback": "잠시 후 다시 시도해주세요."
        }

def analyze_difference_details(user_transcript, target_sentence):
    """difflib를 사용해 두 텍스트의 차이점을 상세히 분석합니다."""
    matcher = difflib.SequenceMatcher(None, user_transcript, target_sentence)
    details = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        user_text = user_transcript[i1:i2]
        target_text = target_sentence[j1:j2]
        if tag == 'equal':
            details.append({"tag": "equal", "text": user_text})
        elif tag == 'replace':
            details.append({"tag": "replace", "user_text": user_text, "target_text": target_text})
        elif tag == 'delete':
            details.append({"tag": "added", "text": user_text})
        elif tag == 'insert':
            details.append({"tag": "omitted", "text": target_text})
    return details


# --- 2. API 엔드포인트 ---
@app.get("/")
async def root():
    return {"message": "의사소통 보조 AI 유음"}

# @app.post("/analyze", response_model=ConversationProcessResponse)
# async def process_conversation_audio(audio_file: UploadFile = File(...)):
#     """대화 음성을 받아 처리하고 AI 응답 텍스트를 반환합니다."""
#     try:
#         timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#         temp_input_path = os.path.join(TEMP_DIR, f"conv_input_{timestamp}")
#         with open(temp_input_path, "wb") as buffer:
#             shutil.copyfileobj(audio_file.file, buffer)
        
#         wav_path = os.path.join(TEMP_DIR, f"conv_converted_{timestamp}.wav")
#         AudioSegment.from_file(temp_input_path).export(wav_path, format="wav")
        
#         audio_data, sample_rate = sf.read(wav_path)
#         clean_audio_data = reduce_noise(audio_data, sample_rate)
#         clean_audio_path = os.path.join(TEMP_DIR, f"conv_cleaned_{timestamp}.wav")
#         sf.write(clean_audio_path, clean_audio_data, sample_rate)

#         raw_transcript = speech_to_text(clean_audio_path)
#         if not raw_transcript: 
#             raise HTTPException(status_code=400, detail="음성을 인식할 수 없습니다.")

#         return JSONResponse(content={
#             "raw_transcript": raw_transcript,
#         })

#     except Exception as e:
#         print(f" 처리 중 오류 발생: {e}")
#         raise HTTPException(status_code=500, detail=str(e))

PRACTICE_SENTENCES = ["바나나", "토끼", "사과", "피자", "하마", "자동차", "우유", "학교", "라디오", "감자"]

# [엔드포인트]
@app.get("/analyze", response_model=PracticeSentenceResponse)
async def get_practice_sentence():
    """발음 교정 학습을 위한 단어를 랜덤으로 반환합니다."""
    chosen_sentence = random.choice(PRACTICE_SENTENCES)
    return {"sentence": chosen_sentence}


# [엔드포인트]
@app.post("/analyze", response_model=PronunciationCheckResponse)
async def check_pronunciation(target_sentence: str = Form(...), audio_file: UploadFile = File(...)):
    """발음 연습 음성을 받아 LLM과 difflib으로 평가하고 결과를 JSON으로 반환합니다."""
    try:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_input_path = os.path.join(TEMP_DIR, f"practice_input_{timestamp}")
        with open(temp_input_path, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)

        wav_path = os.path.join(TEMP_DIR, f"practice_converted_{timestamp}.wav")
        AudioSegment.from_file(temp_input_path).set_frame_rate(16000).set_channels(1).export(wav_path, format="wav")

        audio_data, sample_rate = sf.read(wav_path)
        clean_audio_data = reduce_noise(audio_data, sample_rate)
        clean_audio_path = os.path.join(TEMP_DIR, f"practice_cleaned_{timestamp}.wav")
        sf.write(clean_audio_path, clean_audio_data, sample_rate)

        user_transcript = speech_to_text(clean_audio_path)
        if not user_transcript: user_transcript = "..."

        diff_details = analyze_difference_details(user_transcript, target_sentence)
        evaluation = evaluate_pronunciation_with_llm(target_sentence, user_transcript)

        #최종 응답에 evaluation 결과를 합쳐서 반환
        response_data = {
            "target_sentence": target_sentence,
            "user_transcript": user_transcript,
            "diff_details": diff_details,
            **evaluation 
        }
        
        return JSONResponse(content=response_data)
        
    except Exception as e:
        print(f" 발음 분석 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=str(e))

