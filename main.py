import torch
import os
import shutil
import numpy as np
import difflib # 상세 분석을 위해 difflib를 사용합니다.
import noisereduce as nr
import datetime
import random
import json
from openai import OpenAI
from transformers import pipeline # VitsModel, AutoTokenizer는 TTS 제거로 인해 필요 없어집니다.
import soundfile as sf # TTS 제거로 인해 필요 없어집니다.
from pydub import AudioSegment
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse # FileResponse 대신 JSONResponse를 사용합니다.
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel # 응답 모델 정의를 위해 추가합니다.


# FastAPI 앱 초기화 및 설정
app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 임시 파일 저장 폴더 생성
TEMP_DIR = "temp_files"
os.makedirs(TEMP_DIR, exist_ok=True)


# 응답 모델 정의 (FastAPI의 response_model에 사용)
class ConversationProcessResponse(BaseModel):
    raw_transcript: str
    processed_text: str

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


# 1. 핵심 로직 함수
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

def correct_and_process_text(raw_text):
    """GPT 모델을 사용해 텍스트를 교정하고, 상황에 맞는 응답을 생성합니다."""
    print(" GPT-4o-mini로 텍스트 교정 및 응답 생성을 시작합니다...")
    # OpenAI API 키를 환경 변수에서 가져옵니다.
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    system_prompt = """당신은 언어장애인의 의사소통을 돕는 매우 친절하고 상냥한 AI 언어 치료사입니다. 
    사용자의 어눌한 발화를 문맥에 맞게 문장을 교정한 뒤, 
    정확한 응답을 생성해주세요. 최종 결과는 음성으로 출력되므로, 
    상대방과 대화하는 듯한 친근한 말투로 답변해야 합니다."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": raw_text}],
        temperature=0.7
    )
    processed_text = response.choices[0].message.content
    print(" GPT-4o-mini 처리 완료!")
    return processed_text


def evaluate_pronunciation_with_llm(target_sentence, user_transcript):
    """LLM을 사용하여 발음을 평가하고 점수 및 피드백을 생성합니다."""
    print(" LLM으로 발음 평가를 시작합니다...")
    try:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        system_prompt = """
        당신은 언어 치료사 입니다. '정답 문장'과 사용자가 발음한 '사용자 발음' 텍스트를 비교하여 평가를 수행합니다.
        평가 결과를 반드시 다음의 JSON 형식으로만 반환해주세요. 다른 설명은 절대 추가하지 마세요.
        {
          "score": "0에서 100 사이의 정수 점수",
          "positive_points": "발음에서 잘한 점에 대한 긍정적인 피드백 (간단한 한 문장)",
          "areas_for_improvement": "개선이 필요한 부분에 대한 구체적인 피드백 (간단한 한 문장)",
          "overall_feedback": "종합적인 격려의 메시지 (한 문장)"
        }
        """
        user_prompt = f"정답 문장: \"{target_sentence}\"\n사용자 발음: \"{user_transcript}\""
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"}, # JSON 형식으로 응답을 받도록 설정합니다.
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        
        evaluation_result = json.loads(response.choices[0].message.content)
        # 점수를 정수형으로 변환하고, 없을 경우 기본값 0을 설정합니다.
        evaluation_result['score'] = int(evaluation_result.get('score', 0))
        print(" LLM 평가 완료!")
        return evaluation_result

    except Exception as e:
        print(f" LLM 평가 중 심각한 오류 발생: {e}")
        # 오류 발생 시 기본 평가 결과를 반환합니다.
        return {
            "score": 0,
            "positive_points": "AI 평가 중 오류가 발생했습니다.",
            "areas_for_improvement": "서버 로그를 확인해주세요.",
            "overall_feedback": "잠시 후 다시 시도해주세요."
        }

def analyze_difference_details(user_transcript, target_sentence):
    """difflib를 사용해 두 문장의 차이점을 상세히 분석합니다."""
    # SequenceMatcher를 사용하여 두 문자열의 차이점을 비교합니다.
    matcher = difflib.SequenceMatcher(None, user_transcript, target_sentence)
    details = []
    # get_opcodes()를 사용하여 두 문자열 간의 변경 사항을 나타내는 태그와 인덱스를 가져옵니다.
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        user_text = user_transcript[i1:i2]
        target_text = target_sentence[j1:j2]
        if tag == 'equal': # 두 문자열이 동일한 부분
            details.append({"tag": "equal", "text": user_text})
        elif tag == 'replace': # 한 문자열이 다른 문자열로 대체된 부분
            details.append({"tag": "replace", "user_text": user_text, "target_text": target_text})
        elif tag == 'delete':  # 사용자가 추가로 발음한 부분 (정답 문장에는 없고 사용자 발음에만 있는 부분)
            details.append({"tag": "added", "text": user_text})
        elif tag == 'insert':  # 사용자가 누락한 부분 (사용자 발음에는 없고 정답 문장에만 있는 부분)
            details.append({"tag": "omitted", "text": target_text})
    return details


# 2. API 엔드포인트
@app.get("/")
async def root():
    return {"message": "AI 의사소통 보조"}

@app.post("/api/v1/conversation/process", response_model=ConversationProcessResponse)
async def process_conversation_audio(audio_file: UploadFile = File(...)):
    """대화 음성을 받아 처리하고 AI 응답 텍스트를 반환합니다."""
    try:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_input_path = os.path.join(TEMP_DIR, f"conv_input_{timestamp}")
        # 업로드된 오디오 파일을 임시 파일로 저장합니다.
        with open(temp_input_path, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)
        
        wav_path = os.path.join(TEMP_DIR, f"conv_converted_{timestamp}.wav")
        # 모든 오디오 형식을 WAV로 변환하여 처리 일관성을 유지합니다.
        AudioSegment.from_file(temp_input_path).export(wav_path, format="wav")
        
        # WAV 파일에서 오디오 데이터와 샘플링 레이트를 읽어옵니다.
        audio_data, sample_rate = sf.read(wav_path)
        # 노이즈 제거 함수를 호출합니다.
        clean_audio_data = reduce_noise(audio_data, sample_rate)
        clean_audio_path = os.path.join(TEMP_DIR, f"conv_cleaned_{timestamp}.wav")
        # 노이즈 제거된 오디오 데이터를 WAV 파일로 저장합니다.
        sf.write(clean_audio_path, clean_audio_data, sample_rate)

        # 음성 파일을 텍스트로 변환합니다.
        raw_transcript = speech_to_text(clean_audio_path)
        if not raw_transcript: 
            raise HTTPException(status_code=400, detail="음성을 인식할 수 없습니다.")

        # 텍스트를 교정하고 응답을 생성합니다.
        final_response_text = correct_and_process_text(raw_transcript)
        if not final_response_text: 
            raise HTTPException(status_code=500, detail="응답 생성에 실패했습니다.")

        # 이제 텍스트 응답을 JSON으로 반환합니다.
        return JSONResponse(content={
            "raw_transcript": raw_transcript,
            "processed_text": final_response_text
        })

    except Exception as e:
        print(f" 처리 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=str(e))

PRACTICE_SENTENCES = ["오늘은 날씨가 정말 좋네요.", "학교 다녀오겠습니다.", "이 사과는 얼마인가요?", "만나서 반갑습니다.", "즐거운 하루 보내세요."]
@app.get("/api/v1/practice/sentence", response_model=PracticeSentenceResponse)
async def get_practice_sentence():
    """발음 교정 학습을 위한 문장을 랜덤으로 반환합니다."""
    # PRACTICE_SENTENCES 리스트에서 무작위로 문장을 선택하여 반환합니다.
    return {"sentence": random.choice(PRACTICE_SENTENCES)}

@app.post("/api/v1/practice/check", response_model=PronunciationCheckResponse)
async def check_pronunciation(target_sentence: str = Form(...), audio_file: UploadFile = File(...)):
    """발음 연습 음성을 받아 LLM과 difflib으로 평가하고 결과를 JSON으로 반환합니다."""
    try:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_input_path = os.path.join(TEMP_DIR, f"practice_input_{timestamp}")
        # 업로드된 오디오 파일을 임시 파일로 저장합니다.
        with open(temp_input_path, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)

        wav_path = os.path.join(TEMP_DIR, f"practice_converted_{timestamp}.wav")
        # 모든 오디오 형식을 WAV로 변환하여 처리 일관성을 유지합니다.
        AudioSegment.from_file(temp_input_path).export(wav_path, format="wav")

        # WAV 파일에서 오디오 데이터와 샘플링 레이트를 읽어옵니다.
        audio_data, sample_rate = sf.read(wav_path)
        # 노이즈 제거 함수를 호출합니다.
        clean_audio_data = reduce_noise(audio_data, sample_rate)
        clean_audio_path = os.path.join(TEMP_DIR, f"practice_cleaned_{timestamp}.wav")
        # 노이즈 제거된 오디오 데이터를 WAV 파일로 저장합니다.
        sf.write(clean_audio_path, clean_audio_data, sample_rate)

        # 음성 파일을 텍스트로 변환합니다.
        user_transcript = speech_to_text(clean_audio_path)
        # 음성 인식이 실패할 경우 기본값을 설정합니다.
        if not user_transcript: user_transcript = "..."

        # 상세 분석과 LLM 평가를 모두 호출하고 결과를 합칩니다.
        diff_details = analyze_difference_details(user_transcript, target_sentence)
        evaluation = evaluate_pronunciation_with_llm(target_sentence, user_transcript)
        
        # 모든 결과를 하나의 JSON 응답으로 반환합니다.
        return JSONResponse(content={
            "target_sentence": target_sentence,
            "user_transcript": user_transcript,
            "diff_details": diff_details,  # 상세 분석 결과 추가
            **evaluation # LLM 평가 결과 (score, positive_points 등)를 병합합니다.
        })
    except Exception as e:
        print(f" 발음 분석 중 오류 발생: {e}")
        # 예외 발생 시 HTTP 500 오류를 반환합니다.
        raise HTTPException(status_code=500, detail=str(e))

