import torch
import os
import shutil
import numpy as np
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
from typing import List
from dotenv import load_dotenv
from fastapi.staticfiles import StaticFiles

# .env 파일에서 환경 변수를 로드합니다.
load_dotenv()

# FastAPI 앱 초기화 및 설정
app = FastAPI()

# CORS 설정을 먼저 적용합니다.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 이미지 파일 경로(StaticFiles)를 나중에 설정합니다.
app.mount("/static/images", StaticFiles(directory="static/images"), name="static_images")


# --- 폴더 설정 ---
TEMP_DIR = "temp_files"
os.makedirs(TEMP_DIR, exist_ok=True)


# --- 혀 위치 이미지 데이터 ---
PRONUNCIATION_GUIDES = [
    {"chars": ["ㄱ", "ㄲ", "ㅋ", "ㅇ"], "imageFile": "c-g.png"},
    {"chars": ["ㄴ", "ㄷ", "ㄸ", "ㅌ"], "imageFile": "c-n.png"},
    {"chars": ["ㄹ"], "imageFile": "c-r.png"},
    {"chars": ["ㅁ", "ㅂ", "ㅃ", "ㅍ"], "imageFile": "c-m.png"},
    {"chars": ["ㅅ", "ㅆ"], "imageFile": "c-s.png"},
    {"chars": ["변이음_ㅅ", "변이음_ㅆ"], "imageFile": "c-s-alt.png"},
    {"chars": ["ㅏ"], "imageFile": "v-a.png"},
    {"chars": ["ㅔ", "ㅐ"], "imageFile": "v-e.png"},
    {"chars": ["ㅓ", "ㅗ"], "imageFile": "v-eo.png"},
    {"chars": ["ㅣ", "ㅑ", "ㅒ", "ㅕ", "ㅖ", "ㅛ", "ㅠ"], "imageFile": "v-i.png"},
    {"chars": ["ㅡ", "ㅜ", "ㅘ", "ㅙ", "ㅚ", "ㅝ", "ㅞ", "ㅟ", "ㅢ"], "imageFile": "v-u.png"}
]

# 빠른 조회를 위한 이미지 경로 맵 생성
IMAGE_GUIDE_MAP = {}
for guide in PRONUNCIATION_GUIDES:
    for char in guide["chars"]:
        IMAGE_GUIDE_MAP[char] = guide["imageFile"]


# --- 새 응답 모델 정의 ---
class IncorrectPoint(BaseModel):
    expected: str
    actual: str
    img: str
    mouth_shape: str
    tongue_shape: str
    breathing: str

class PronunciationAnalysisResponse(BaseModel):
    score: str
    transcription: str
    incorrect_points: List[IncorrectPoint]


# --- 핵심 로직 함수 ---

def decompose_hangul(char):
    """한글 음절을 초성, 중성, 종성으로 분해합니다."""
    if not '가' <= char <= '힣':
        return (char, None, None)
    
    char_code = ord(char) - ord('가')
    
    chosung_index = char_code // (21 * 28)
    jungsung_index = (char_code % (21 * 28)) // 28
    jongsung_index = char_code % 28
    
    CHOSUNG = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
    JUNGSUNG = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
    JONGSUNG = ['', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
    
    chosung = CHOSUNG[chosung_index]
    jongsung = JONGSUNG[jongsung_index] if jongsung_index > 0 else None
    
    return (chosung, JUNGSUNG[jungsung_index], jongsung)


def reduce_noise(audio_data, sample_rate):
    """오디오 데이터에서 배경 소음을 제거합니다."""
    print("... 오디오 노이즈 제거를 시작합니다 ...")
    reduced_noise_audio = nr.reduce_noise(y=audio_data, sr=sample_rate, stationary=False, prop_decrease=0.8)
    print("노이즈 제거 완료!")
    return reduced_noise_audio

def speech_to_text(audio_file_path):
    """Whisper 모델을 사용해 음성 파일을 텍스트로 변환합니다."""
    print(f"\nSTT 시작: '{audio_file_path}'")
    transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3", torch_dtype=torch.float16, device_map="auto")
    result = transcriber(audio_file_path, generate_kwargs={"language": "korean", "task": "transcribe"})
    transcription = result['text'].strip()
    print(f"STT 결과: {transcription}")
    return transcription

def evaluate_pronunciation_with_llm(target_sentence, user_transcript):
    """LLM을 사용하여 발음을 글자 단위로 평가하고 구체적인 피드백을 생성합니다."""
    print("LLM으로 발음 평가를 시작합니다...")
    if not user_transcript or user_transcript == "...":
        return { "score": "0", "incorrect_points": [] }
        
    try:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        system_prompt = """
        당신은 한국어 발음 교정을 전문으로 하는 언어 치료사입니다. '목표 발음'과 '사용자 발음'을 글자 단위로 비교하여, 틀린 각 글자에 대한 구체적인 피드백을 생성합니다.

        # 분석 과정:
        1. '목표 발음'과 '사용자 발음'에서 다른 글자를 모두 찾습니다.
        2. 각각의 틀린 글자 쌍('expected', 'actual')에 대해, 'expected' 글자를 올바르게 발음하기 위한 '입 모양', '혀 위치', '호흡법'에 대한 피드백을 쉽고 구체적으로 생성합니다.
        3. 전체적인 발음의 정확도를 0에서 100 사이의 점수로 평가합니다.

        # 출력 형식:
        반드시 다음의 JSON 형식으로만 응답해야 합니다. 다른 설명은 절대 추가하지 마세요. 틀린 글자가 없다면 'incorrect_points'를 빈 배열(`[]`)로 반환하세요.
        "피자"를 "키차"로 발음했다면, 'incorrect_points' 배열에는 "피"->"키", "자"->"차" 두 개의 객체가 포함되어야 합니다.

        {
          "score": "0에서 100 사이의 정수 점수 (문자열 형태)",
          "incorrect_points": [
            {
              "expected": "목표 글자",
              "actual": "사용자가 발음한 글자",
              "mouth_shape": "해당 글자에 대한 입 모양 피드백",
              "tongue_shape": "해당 글자에 대한 혀 위치 피드백",
              "breathing": "해당 글자에 대한 호흡 및 공기 흐름 피드백"
            }
          ]
        }
        """
        user_prompt = f"목표 발음: \"{target_sentence}\"\n사용자 발음: \"{user_transcript}\""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        evaluation_result = json.loads(response.choices[0].message.content)
        print("LLM 평가 완료!")
        return evaluation_result

    except Exception as e:
        print(f"LLM 평가 중 심각한 오류 발생: {e}")
        return {
            "score": "0",
            "incorrect_points": []
        }


# --- API 엔드포인트 ---
@app.get("/")
async def root():
    return {"message": "의사소통 보조 AI 유음"}


@app.post("/analyze")
async def analyze_pronunciation(target_sentence: str = Form(...), audio_file: UploadFile = File(...)):
    """사용자의 발음을 분석하고, 틀린 글자별 상세 피드백을 반환합니다."""
    try:
        # 1. 오디오 파일 저장 및 변환
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_input_path = os.path.join(TEMP_DIR, f"practice_input_{timestamp}{os.path.splitext(audio_file.filename)[1]}")
        with open(temp_input_path, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)

        wav_path = os.path.join(TEMP_DIR, f"practice_converted_{timestamp}.wav")
        audio = AudioSegment.from_file(temp_input_path)
        audio.set_frame_rate(16000).set_channels(1).export(wav_path, format="wav")

        # 2. 노이즈 제거
        audio_data, sample_rate = sf.read(wav_path)
        if np.any(audio_data):
            clean_audio_data = reduce_noise(audio_data, sample_rate)
        else:
            clean_audio_data = audio_data # 오디오 데이터가 비어있으면 노이즈 제거 생략
        clean_audio_path = os.path.join(TEMP_DIR, f"practice_cleaned_{timestamp}.wav")
        sf.write(clean_audio_path, clean_audio_data, sample_rate)

        # 3. 음성을 텍스트로 변환 (STT)
        user_transcript = speech_to_text(clean_audio_path)
        if not user_transcript:
            user_transcript = "..."

        # 4. LLM을 통해 발음 평가
        llm_analysis = evaluate_pronunciation_with_llm(target_sentence, user_transcript)

        # 5. incorrect_points에 이미지 파일명 추가
        processed_incorrect_points = []
        if llm_analysis and "incorrect_points" in llm_analysis:
            # 'ㅈ', 'ㅉ', 'ㅊ'에 대한 이미지 매핑 추가 (혀 위치가 유사한 'ㅅ' 이미지 사용)
            IMAGE_GUIDE_MAP['ㅈ'] = 'c-s.png'
            IMAGE_GUIDE_MAP['ㅉ'] = 'c-s.png'
            IMAGE_GUIDE_MAP['ㅊ'] = 'c-s.png'

            for point in llm_analysis["incorrect_points"]:
                expected_char = point.get("expected")
                img_filename = "default.png" # 기본값 설정

                # 글자가 한글 음절일 경우 분해해서 초성을 가져옴
                if expected_char and '가' <= expected_char <= '힣':
                    chosung, _, _ = decompose_hangul(expected_char)
                    img_filename = IMAGE_GUIDE_MAP.get(chosung, "default.png")
                # 단일 자음 또는 모음일 경우 직접 찾아봄
                elif expected_char in IMAGE_GUIDE_MAP:
                    img_filename = IMAGE_GUIDE_MAP.get(expected_char, "default.png")
                
                point_with_img = {**point, "img": img_filename}
                processed_incorrect_points.append(point_with_img)

        # 6. 최종 응답 데이터 구성
        response_data = {
            "score": llm_analysis.get("score", "0"),
            "transcription": user_transcript,
            "incorrect_points": processed_incorrect_points
        }

        # 임시 파일 정리
        for f in [temp_input_path, wav_path, clean_audio_path]:
            if os.path.exists(f):
                os.remove(f)

        return JSONResponse(content=response_data)

    except Exception as e:
        print(f"발음 분석 중 심각한 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=f"서버에서 오디오 파일을 처리하는 중 오류가 발생했습니다: {str(e)}")

