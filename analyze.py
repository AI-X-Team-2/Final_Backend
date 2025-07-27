import torch
import os
import shutil
import numpy as np
import noisereduce as nr
import datetime
import random
import json
from openai import OpenAI
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


#혀 위치 이미지 데이터 (기존과 동일)
PRONUNCIATION_GUIDES = [
    {"chars": ["ㄱ", "ㄲ", "ㅋ", "ㅇ"], "imageFile": "c-g.png"},
    {"chars": ["ㄴ", "ㄷ", "ㄸ", "ㅌ"], "imageFile": "c-n.png"},
    {"chars": ["ㄹ"], "imageFile": "c-r.png"},
    {"chars": ["ㅁ", "ㅂ", "ㅃ", "ㅍ"], "imageFile": "c-m.png"},
    {"chars": ["ㅅ", "ㅆ"], "imageFile": "c-s.png"},
    {"chars": ["변이음_ㅅ", "변이음_ㅆ"], "imageFile": "c-s-alt.png"}, # 이 변이음은 LLM이 생성하지 않을 가능성이 높음
    {"chars": ["ㅏ"], "imageFile": "v-a.png"},
    {"chars": ["ㅔ", "ㅐ"], "imageFile": "v-e.png"},
    {"chars": ["ㅓ", "ㅗ"], "imageFile": "v-eo.png"},
    {"chars": ["ㅣ", "ㅑ", 'ㅒ', "ㅕ", "ㅖ", "ㅛ", "ㅠ"], "imageFile": "v-i.png"},
    {"chars": ["ㅡ", "ㅜ", "ㅘ", "ㅙ", "ㅚ", "ㅝ", "ㅞ", "ㅟ", "ㅢ"], "imageFile": "v-u.png"}
]

# 빠른 조회를 위한 이미지 경로 맵 생성 (명확화)
IMAGE_GUIDE_MAP = {}
for guide in PRONUNCIATION_GUIDES:
    for char in guide["chars"]:
        IMAGE_GUIDE_MAP[char] = guide["imageFile"]

# 추가: LLM이 단일 자모음을 반환할 경우를 대비하여 직접 매핑 추가
# ㅈ, ㅉ, ㅊ은 ㅅ과 혀 위치가 유사하여 'c-s.png' 사용
IMAGE_GUIDE_MAP['ㅈ'] = 'c-s.png'
IMAGE_GUIDE_MAP['ㅉ'] = 'c-s.png'
IMAGE_GUIDE_MAP['ㅊ'] = 'c-s.png'

# 모든 초성, 중성, 종성에 대해 기본 이미지 파일 설정 
CHOSUNG = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
JUNGSUNG = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
JONGSUNG = ['', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
for char in CHOSUNG + JUNGSUNG + JONGSUNG:
    if char and char not in IMAGE_GUIDE_MAP:
        # 기본적으로 해당 글자에 맞는 이미지를 찾되, 없으면 'default.png'를 사용
        # 이 부분은 실제 이미지 파일명 규칙에 따라 조절 필요
        IMAGE_GUIDE_MAP[char] = f"c-{char}.png" if os.path.exists(os.path.join("static/images", f"c-{char}.png")) else "default.png"


# --- 핵심 로직 함수 ---

def reduce_noise(audio_data, sample_rate):
    """오디오 데이터에서 배경 소음을 제거합니다."""
    print("... 오디오 노이즈 제거를 시작합니다 ...")
    reduced_noise_audio = nr.reduce_noise(y=audio_data, sr=sample_rate, stationary=False, prop_decrease=0.8)
    print("노이즈 제거 완료!")
    return reduced_noise_audio

def speech_to_text(audio_file_path):
    """OpenAI Whisper API를 사용해 음성 파일을 텍스트로 변환합니다."""
    print(f"\nSTT 시작 (Whisper API): '{audio_file_path}'")
    try:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        with open(audio_file_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text"
            )
        transcription = transcript.strip()
        print(f"STT 결과: {transcription}")
        return transcription
    except Exception as e:
        print(f"Whisper API 호출 중 오류 발생: {e}")
        return "..." # 오류 발생 시 기본값 반환


def decompose_hangul(char):
    """한글 음절을 초성, 중성, 종성으로 분해합니다."""
    if not ('가' <= char <= '힣'):
        return (char, '', '') # 음절이 아니면 초성만 있는 것으로 간주

    char_code = ord(char) - ord('가')
    chosung_index = char_code // (21 * 28)
    jungsung_index = (char_code % (21 * 28)) // 28
    jongsung_index = char_code % 28
    
    CHOSUNG = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
    JUNGSUNG = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
    JONGSUNG = ['', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
    
    chosung = CHOSUNG[chosung_index]
    jungsung = JUNGSUNG[jungsung_index]
    jongsung = JONGSUNG[jongsung_index] if jongsung_index > 0 else ''
    
    return (chosung, jungsung, jongsung)


def evaluate_pronunciation_with_llm(target_sentence, user_transcript):
    """LLM을 사용하여 발음을 평가하고, 글자 수 오류는 별도 처리 후 구체적인 피드백을 생성합니다."""
    print("LLM으로 발음 평가를 시작합니다...")
    normalized_target = target_sentence.strip()
    normalized_user = user_transcript.strip()

    if not normalized_user or normalized_user == "...":
        return {"score": "0", "incorrect_points": []}

    if normalized_target == normalized_user:
        return {"score": "100", "incorrect_points": []}
    
    # 오류 유형을 분리하기 위한 리스트
    pronunciation_pairs = [] # 발음이 다른 경우
    final_incorrect_points = [] # 최종 결과를 담을 리스트

    # 1. 글자 수가 다른 경우를 먼저 처리
    min_len = min(len(normalized_target), len(normalized_user))
    for i in range(min_len):
        if normalized_target[i] != normalized_user[i]:
            pronunciation_pairs.append({
                "expected": normalized_target[i],
                "actual": normalized_user[i]
            })
            
    # 누락된 단어 처리
    if len(normalized_target) > len(normalized_user):
        for i in range(min_len, len(normalized_target)):
            final_incorrect_points.append({
                "expected": normalized_target[i],
                "actual": "",
                "diff_detail": "누락된 단어", # 프론트엔드에서 구분하기 위한 키
                "mouth_shape": "", "tongue_shape": "", "breathing": "", "img": ""
            })
    # 추가된 단어 처리
    elif len(normalized_user) > len(normalized_target):
        for i in range(min_len, len(normalized_user)):
            final_incorrect_points.append({
                "expected": "",
                "actual": normalized_user[i],
                "diff_detail": "추가된 단어", # 프론트엔드에서 구분하기 위한 키
                "mouth_shape": "", "tongue_shape": "", "breathing": "", "img": ""
            })
            
    # 2. 발음이 다른 경우에만 LLM 호출
    if not pronunciation_pairs:
        # 점수 계산 (누락/추가만 있는 경우)
        correct_count = len(normalized_target) - len(final_incorrect_points)
        score = max(0, int((correct_count / len(normalized_target)) * 100)) if len(normalized_target) > 0 else 0
        return {"score": str(score), "incorrect_points": final_incorrect_points}

    try:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
        system_prompt = f"""
        당신은 한국어 발음 교정을 전문으로 하는 언어 치료사입니다. '목표 발음'과 '사용자 발음'을 비교하여, 틀린 각 글자에 대한 구체적인 피드백을 생성합니다.

        # 분석 대상:
        목표 발음: "{target_sentence}"
        사용자 발음: "{user_transcript}"

        # 분석 과정:
        1. 아래 '틀린 글자 목록'에 있는 각 쌍('expected', 'actual')에 대해서만 분석을 수행합니다.
        2. 각 쌍에 대해 'expected' 글자를 올바르게 발음하기 위한 '입 모양', '혀 위치', '호흡법' 피드백을 쉽고 구체적으로 생성합니다.
        3. 'expected'와 'actual' 글자를 초성, 중성, 종성으로 분해하여 어느 부분이 다른지 분석하고, 그 결과를 'diff_detail'에 "[올바른 발음] → [내 발음]" 형식으로 요약합니다. 예를 들어, 'expected'의 초성이 'ㅅ'이고 'actual'의 초성이 'ㅂ'이라면 "초성: ㅅ → ㅂ"과 같이 요약합니다.
        4. 전체적인 발음 정확도를 0에서 100 사이의 점수로 평가합니다. (목표: {len(normalized_target)} 글자 중 {len(normalized_target) - (len(pronunciation_pairs) + len(final_incorrect_points))} 글자 맞음)

        # 출력 형식:
        반드시 다음의 JSON 형식으로만 응답해야 합니다. 다른 설명은 절대 추가하지 마세요.
        {{
          "score": "0에서 100 사이의 정수 점수 (문자열 형태)",
          "incorrect_points": [
            {{
              "expected": "목표 글자",
              "actual": "사용자가 발음한 글자",
              "diff_detail": "틀린 부분 상세 분석 (예: '초성: ㅅ → ㅂ')",
              "mouth_shape": "해당 글자에 대한 입 모양 피드백",
              "tongue_shape": "해당 글자에 대한 혀 위치 피드백",
              "breathing": "해당 글자에 대한 호흡 및 공기 흐름 피드백"
            }}
          ]
        }}
        """

        user_prompt = f"다음은 사용자가 틀리게 발음한 글자 목록입니다. 이 목록을 바탕으로 위 지시에 따라 JSON을 생성해주세요: {json.dumps(pronunciation_pairs, ensure_ascii=False)}"

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        evaluation_result = json.loads(response.choices[0].message.content)
        # LLM 결과와 수동 처리 결과를 합침
        evaluation_result["incorrect_points"].extend(final_incorrect_points)
        print("LLM 평가 완료! 응답 데이터:", evaluation_result)
        return evaluation_result

    except Exception as e:
        print(f"LLM 평가 중 심각한 오류 발생: {e}")
        return {"score": "0", "incorrect_points": final_incorrect_points}


class IncorrectPoint(BaseModel):
    expected: str              #올바른 글자
    actual: str                #사용자가 발음한 글자
    img: str                   #관련 이미지 파일명
    diff_detail: str
    mouth_shape: str           #각 항목에 대한 피드백
    tongue_shape: str
    breathing: str

class PronunciationAnalysisResponse(BaseModel):
    score: str                 #종합 점수
    transcription: str         #사용자의 전체 발음 텍스트
    incorrect_points: List[IncorrectPoint]  #모델의 리스트(틀린 발음이 여러개 일 수 있으므로)


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

        # 3. 음성을 텍스트로 변환 (STT) - Whisper API 사용
        user_transcript = speech_to_text(clean_audio_path)
        if not user_transcript:
            user_transcript = "..."

        # 4. LLM을 통해 발음 평가
        llm_analysis = evaluate_pronunciation_with_llm(target_sentence, user_transcript)

        # 5. incorrect_points에 이미지 파일명 추가
        processed_incorrect_points = []
        if llm_analysis and "incorrect_points" in llm_analysis:
            # 기존에 ㅈ, ㅉ, ㅊ에 대한 매핑은 IMAGE_GUIDE_MAP 선언부에 명확히 추가되었음
            
            for point in llm_analysis["incorrect_points"]:
                expected_char = point.get("expected")
                img_filename = "default.png" # 기본값 설정

                # 혀 위치 이미지를 결정하는 로직 개선
                # 1. 'expected' 글자 자체가 IMAGE_GUIDE_MAP에 직접 매핑되어 있는지 확인 (단일 자음/모음)
                if expected_char in IMAGE_GUIDE_MAP:
                    img_filename = IMAGE_GUIDE_MAP[expected_char]
                    print(f"DEBUG: 직접 매핑된 '{expected_char}' -> {img_filename}") # 디버그 로그
                # 2. 'expected' 글자가 한글 음절일 경우 초성을 분해하여 찾아봄
                elif expected_char and '가' <= expected_char <= '힣':
                    chosung, jungsung, jongsung = decompose_hangul(expected_char)
                    
                    # 초성 이미지를 우선적으로 찾음
                    if chosung in IMAGE_GUIDE_MAP:
                        img_filename = IMAGE_GUIDE_MAP[chosung]
                        print(f"DEBUG: 음절 '{expected_char}'에서 초성 '{chosung}' 매핑 -> {img_filename}") # 디버그 로그
                    # 초성 이미지가 없으면 중성 이미지를 찾아봄 (모음)
                    elif jungsung in IMAGE_GUIDE_MAP:
                        img_filename = IMAGE_GUIDE_MAP[jungsung]
                        print(f"DEBUG: 음절 '{expected_char}'에서 중성 '{jungsung}' 매핑 -> {img_filename}") # 디버그 로그
                    # 그 외의 경우 (종성 관련 이미지는 따로 없다고 가정하거나 default)
                    else:
                        print(f"DEBUG: 음절 '{expected_char}'에 대한 초성/중성 이미지 매핑 없음. default.png 사용.") # 디버그 로그
                # 3. 그 외의 경우 (한글 음절도, 단일 자모음도 아닌 경우)
                else:
                    print(f"DEBUG: '{expected_char}'는 한글 음절/자모음 아님. default.png 사용.") # 디버그 로그
                
                # 최종적으로 이미지 파일명이 결정되었는지 확인
                print(f"DEBUG: 최종 img_filename for '{expected_char}': {img_filename}")

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
        # 오류가 발생했을 때 HTTP 500 응답을 반환하면서 클라이언트에게도 메시지 전달
        raise HTTPException(status_code=500, detail=f"서버에서 오디오 파일을 처리하는 중 오류가 발생했습니다: {str(e)}")

