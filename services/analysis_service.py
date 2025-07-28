# services/analysis_service.py
#사용자로부터 받은 음성 파일을 분석하고 최종 피드백을 생성하기까지의 모든 복잡한 실제 작업이 이 파일 안에서 순서대로 이루어짐
import os
import numpy as np
import noisereduce as nr
import datetime
import json
import io
from openai import OpenAI
import soundfile as sf
from pydub import AudioSegment
from fastapi import UploadFile

from utils.hangul import decompose_hangul
from core.config import IMAGE_GUIDE_MAP

TEMP_DIR = "temp_files"
os.makedirs(TEMP_DIR, exist_ok=True)

#오디오 처리
def _reduce_noise(audio_data, sample_rate):
    print("... 오디오 노이즈 제거를 시작합니다 ...")
    reduced_noise_audio = nr.reduce_noise(y=audio_data, sr=sample_rate, stationary=False, prop_decrease=0.8)
    print("노이즈 제거 완료!")
    return reduced_noise_audio

#음성 인식 (STT) -> 텍스트 변환
def _speech_to_text(audio_file_path):
    print(f"\nSTT 시작 (Whisper API): '{audio_file_path}'")
    try:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        # 겹받침 등 발음이 복잡하여 AI가 헷갈릴 만한 단어들을 추가합니다.
        hints = "앉다, 안다, 값, 갑, 삶, 삼, 닭, 읊다, 옳다"
        with open(audio_file_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text",
                prompt=hints  
            )
        transcription = transcript.strip() if transcript else ""
        print(f"STT 결과: {transcription}")
        return transcription
    except Exception as e:
        print(f"Whisper API 호출 중 오류 발생: {e}")
        return ""

# 정확한 오류 분석 (Code-based)
#_create_diff_detail 함수를 호출하여, 코드가 직접 초성/중성/종성을 분석하고 '추가', 
#'제외' 등을 포함한 100% 정확한 '교정 포인트' 문자열을 생성합니다.
def _create_diff_detail(expected_char, actual_char):
    if not actual_char: return "누락된 단어"
    if not expected_char: return "추가된 단어"
    e_cho, e_jung, e_jong = decompose_hangul(expected_char)
    a_cho, a_jung, a_jong = decompose_hangul(actual_char)
    diffs = []
    if a_cho != e_cho: diffs.append(f"초성: {a_cho} → {e_cho}")
    if a_jung != e_jung: diffs.append(f"중성: {a_jung} → {e_jung}")
    if a_jong != e_jong:
        if not a_jong: diffs.append(f"종성: {e_jong} 추가")
        elif not e_jong: diffs.append(f"종성: {a_jong} 제외")
        else: diffs.append(f"종성: {a_jong} → {e_jong}")
    return ", ".join(diffs)

#AI 피드백 생성 (LLM-based)
#앞 단계에서 코드가 정확하게 계산한 '교정 포인트'를 AI에게 전달하면서, 
#"이 교정 포인트를 바탕으로 입 모양, 혀 위치, 호흡법에 대한 설명만 자연스럽게 만들어줘" 라고 요청합니다.
def _evaluate_pronunciation_with_llm(target_sentence, user_transcript, pronunciation_pairs, llm_input_pairs):
    if not llm_input_pairs:
        final_points = []
        for p in pronunciation_pairs:
            if p.get("is_missing"):
                final_points.append({"expected": p["expected"], "actual": "", "diff_detail": "누락된 단어", "mouth_shape": "", "tongue_shape": "", "breathing": ""})
            elif p.get("is_added"):
                final_points.append({"expected": "", "actual": p["actual"], "diff_detail": "추가된 단어", "mouth_shape": "", "tongue_shape": "", "breathing": ""})
        correct_count = len(target_sentence) - len(final_points)
        score = max(0, int((correct_count / len(target_sentence)) * 100)) if len(target_sentence) > 0 else 0
        return {"score": str(score), "incorrect_points": final_points}
    
    try:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        system_prompt = f"""
        당신은 한국어 발음 교정을 전문으로 하는 언어 치료사입니다.
        '목표 발음'과 '사용자 발음', 그리고 **정확하게 분석된 '교정 포인트'**를 바탕으로, 'expected' 글자를 올바르게 발음하기 위한 '입 모양', '혀 위치', '호흡법' 피드백을 쉽고 구체적으로 생성합니다.
        '교정 포인트'는 이미 정확하므로, 당신이 이를 다시 계산하거나 수정해서는 안 됩니다.
        # 분석 대상 정보
        - 전체 목표 발음: "{target_sentence}"
        - 전체 사용자 발음: "{user_transcript}"
        # 출력 지침
        1. 아래 '분석 요청 목록'에 있는 각 JSON 객체에 대해 피드백을 생성합니다.
        2. 각 쌍에 대해 'mouth_shape', 'tongue_shape', 'breathing' 피드백만 생성합니다.
        3. 전체적인 발음 정확도를 0에서 100 사이의 점수로 평가합니다.
        4. 반드시 아래의 JSON 형식으로만 응답해야 합니다.
        {{
          "score": "0에서 100 사이의 정수 점수 (문자열 형태)",
          "incorrect_points": [
            {{ "expected": "목표 글자", "actual": "사용자가 발음한 글자", "diff_detail": "주어진 교정 포인트 그대로", "mouth_shape": "생성된 입 모양 피드백", "tongue_shape": "생성된 혀 위치 피드백", "breathing": "생성된 호흡법 피드백" }}
          ]
        }}
        """
        user_prompt = f"다음은 분석 요청 목록입니다. 이 목록을 바탕으로 위 지시에 따라 JSON을 생성해주세요: {json.dumps(llm_input_pairs, ensure_ascii=False)}"
        response = client.chat.completions.create(model="gpt-4o-mini", response_format={"type": "json_object"}, messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}])
        evaluation_result = json.loads(response.choices[0].message.content)

        for p in pronunciation_pairs:
            if p.get("is_missing"):
                evaluation_result["incorrect_points"].append({"expected": p["expected"], "actual": "", "diff_detail": "누락된 단어", "mouth_shape": "", "tongue_shape": "", "breathing": ""})
            elif p.get("is_added"):
                evaluation_result["incorrect_points"].append({"expected": "", "actual": p["actual"], "diff_detail": "추가된 단어", "mouth_shape": "", "tongue_shape": "", "breathing": ""})
        
        print("LLM 평가 완료! 응답 데이터:", evaluation_result)
        return evaluation_result
    except Exception as e:
        print(f"LLM 평가 중 심각한 오류 발생: {e}")
        return {"score": "0", "incorrect_points": []}

# 최종 데이터 종합 및 정리
#AI로부터 받은 결과에 틀린 글자별 이미지 파일 경로(img)를 추가함.
#최종적으로 점수, 텍스트, 피드백 목록을 하나의 response_data로 묶음.
#마지막으로 사용했던 임시 오디오 파일을 삭제하여 서버를 깨끗하게 유지함.
async def analyze_user_pronunciation(target_sentence: str, audio_file: UploadFile):
    #오디오 전처리
    audio_content = await audio_file.read()
    audio_stream = io.BytesIO(audio_content)
    audio = AudioSegment.from_file(audio_stream)
    wav_stream = io.BytesIO()
    audio.set_frame_rate(16000).set_channels(1).export(wav_stream, format="wav")
    wav_stream.seek(0)
    audio_data, sample_rate = sf.read(wav_stream)
    clean_audio_data = _reduce_noise(audio_data, sample_rate) if np.any(audio_data) else audio_data
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_audio_path = os.path.join(TEMP_DIR, f"practice_cleaned_{timestamp}.wav")
    sf.write(clean_audio_path, clean_audio_data, sample_rate)

    #텍스트 변환 (음성 인식)
    user_transcript = _speech_to_text(clean_audio_path)
    if not user_transcript: user_transcript = ""

    #1차 검수 (글자 비교)
    normalized_target = target_sentence.strip()
    normalized_user = user_transcript.strip()
    
    if normalized_target == normalized_user:
        return {"score": "100", "transcription": user_transcript, "incorrect_points": []}

    pronunciation_pairs = []
    min_len = min(len(normalized_target), len(normalized_user))
    for i in range(min_len):
        if normalized_target[i] != normalized_user[i]:
            pronunciation_pairs.append({"expected": normalized_target[i], "actual": normalized_user[i]})
    if len(normalized_target) > len(normalized_user):
        for i in range(min_len, len(normalized_target)):
            pronunciation_pairs.append({"expected": normalized_target[i], "actual": "", "is_missing": True})
    elif len(normalized_user) > len(normalized_target):
        for i in range(min_len, len(normalized_user)):
            pronunciation_pairs.append({"expected": "", "actual": normalized_user[i], "is_added": True})

    #2차 정밀 분석 (코드 기반 교정 포인트 생성)
    for pair in pronunciation_pairs:
        if not pair.get("is_missing") and not pair.get("is_added"):
            pair["diff_detail"] = _create_diff_detail(pair["expected"], pair["actual"])

    llm_input_pairs = [p for p in pronunciation_pairs if not p.get("is_missing") and not p.get("is_added")]

    #3차 심층 분석 (AI 기반 피드백 생성)
    llm_analysis = _evaluate_pronunciation_with_llm(normalized_target, normalized_user, pronunciation_pairs, llm_input_pairs)

    processed_incorrect_points = []
    if "incorrect_points" in llm_analysis:
        for point in llm_analysis["incorrect_points"]:
            expected_char = point.get("expected")
            img_filename = "default.png"
            if expected_char:
                if expected_char in IMAGE_GUIDE_MAP:
                    img_filename = IMAGE_GUIDE_MAP[expected_char]
                elif '가' <= expected_char <= '힣':
                    chosung, jungsung, _ = decompose_hangul(expected_char)
                    img_filename = IMAGE_GUIDE_MAP.get(chosung, IMAGE_GUIDE_MAP.get(jungsung, "default.png"))
            point_with_img = {**point, "img": img_filename}
            processed_incorrect_points.append(point_with_img)
            
    #최종 포장 및 출하 (결과 종합)
    response_data = {
        "score": llm_analysis.get("score", "0"),
        "transcription": user_transcript,
        "incorrect_points": processed_incorrect_points
    }
    #임시 오디오 파일 삭제
    if os.path.exists(clean_audio_path):
        os.remove(clean_audio_path)

    return response_data