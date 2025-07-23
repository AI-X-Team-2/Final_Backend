import torch
import os
import queue
import numpy as np
import difflib  # 문장 비교를 위해 추가
import noisereduce as nr # 노이즈 제거를 위해 추가
from openai import OpenAI
from transformers import pipeline, VitsModel, AutoTokenizer
import sounddevice as sd
import soundfile as sf
import random # 연습 문장을 랜덤으로 선택하기 위해 추가


# 0단계: 오디오 노이즈 제거 (Noise Reduction)
def reduce_noise(audio_data, sample_rate):
    """
    오디오 데이터에서 배경 소음을 제거합니다.
    스테레오 오디오는 모노로 자동 변환합니다.
    """
    print("... 오디오 노이즈 제거를 시작합니다 ...")

    # [수정됨] 스테레오 데이터(2D 배열, 2채널)를 모노(1D 배열)로 변환
    if audio_data.ndim > 1 and audio_data.shape[1] > 1:
        print("... 스테레오 오디오를 모노로 변환합니다 ...")
        audio_data = audio_data.mean(axis=1) # 두 채널을 평균내어 하나로 합침

    # squeeze는 크기가 1인 불필요한 차원을 제거 (예: (n, 1) -> (n,))
    if audio_data.ndim > 1:
        audio_data = audio_data.squeeze()

    reduced_noise_audio = nr.reduce_noise(y=audio_data, sr=sample_rate)
    print("👍 노이즈 제거 완료!")
    return reduced_noise_audio


# 1단계: 음성 -> 텍스트 (Speech-to-Text)
def speech_to_text(audio_file_path):
    """
    구음장애 특화 Whisper 모델을 사용해 음성 파일을 텍스트로 변환합니다.
    """
    print(f"\n✅ 1단계: '{audio_file_path}' 파일의 음성 인식을 시작합니다...")
    transcriber = pipeline(
        "automatic-speech-recognition",
        model="RecCode/whisper_final",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    result = transcriber(
        audio_file_path,
        generate_kwargs={"language": "korean"},
        return_timestamps=True
    )
    print("👍 음성 인식 완료!")
    return result["text"].strip()


# 2단계: 텍스트 교정 및 처리 (Text Correction & Processing)
def correct_and_process_text(raw_text):
    """
    GPT-4o-mini 모델을 사용해 텍스트를 교정하고, 상황에 맞는 응답을 생성합니다.
    """
    print("✅ 2단계: GPT-4o-mini로 텍스트 교정 및 응답 생성을 시작합니다...")
    try:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    except TypeError:
        print("🚨 오류: OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        return None

    system_prompt = """
    당신은 언어장애인의 의사소통을 돕는 매우 친절하고 상냥한 AI 비서입니다.
    사용자의 발화를 텍스트로 변환한 결과가 주어집니다. 이 텍스트는 다소 어눌하거나 문법에 맞지 않을 수 있습니다.
    주어진 텍스트를 바탕으로 사용자의 본래 의도에 맞게 문장을 교정한 뒤, 그에 대한 정확한 응답을 생성해주세요.
    최종 결과는 상대방에게 음성으로 출력되므로, 사용자가 직접 말하는 듯한 친근한 말투로 답변해야 합니다.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": raw_text}
        ],
        temperature=0.7
    )
    processed_text = response.choices[0].message.content
    print("👍 GPT-4o-mini 처리 완료!")
    return processed_text


# 3단계: 텍스트 -> 음성 (Text-to-Speech)
def text_to_speech_generate(text_to_speak):
    """
    MMS-TTS 모델을 사용해 텍스트를 음성 데이터로 변환합니다. (재생은 하지 않음)
    """
    print("✅ 3단계: MMS-TTS로 음성 합성을 시작합니다...")
    model = VitsModel.from_pretrained("facebook/mms-tts-kor")
    tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-kor")
    inputs = tokenizer(text_to_speak, return_tensors="pt")
    inputs['input_ids'] = inputs['input_ids'].long()

    with torch.no_grad():
        output = model(**inputs).waveform

    sampling_rate = model.config.sampling_rate
    audio_data = output.squeeze().cpu().numpy()
    print("👍 음성 데이터 생성 완료!")
    return audio_data, sampling_rate

def play_audio(audio_data, sampling_rate):
    """
    생성된 음성 데이터를 스피커로 재생합니다.
    """
    print("🔊 음성을 재생합니다...")
    sd.play(audio_data, sampling_rate)
    sd.wait()
    print("👍 음성 재생 완료!")


# 기능 A: 실시간 대화 보조 모드
def start_realtime_mode():
    """
    사용자가 원할 때 녹음을 시작하고 중지한 뒤, 변환된 음성을 재생합니다.
    """
    samplerate = 16000
    filename = "temp_recording.wav"
    q = queue.Queue()

    def audio_callback(indata, frames, time, status):
        if status: print(status)
        q.put(indata.copy())

    print("\n🎤 실시간 대화 보조 모드를 시작합니다. Ctrl+C를 눌러 메뉴로 돌아갑니다.")
    while True:
        try:
            with sd.InputStream(samplerate=samplerate, channels=1, callback=audio_callback):
                print("\n... 녹음 시작. 중지하시려면 Enter 키를 누르세요 ...")
                input()
                print("... 녹음 중지 ...")

            recording = []
            while not q.empty():
                recording.append(q.get())

            if not recording:
                print("...녹음된 내용이 없습니다.")
                continue

            recording = np.concatenate(recording, axis=0)

            # [수정됨] 노이즈 제거 적용
            clean_recording = reduce_noise(recording, samplerate)
            sf.write(filename, clean_recording, samplerate)

            raw_transcript = speech_to_text(filename)
            print("\n--- [Whisper STT 원본 결과] ---\n", raw_transcript)

            if raw_transcript and len(raw_transcript.strip()) > 0:
                final_response = correct_and_process_text(raw_transcript)
                print("\n--- [GPT-4o 최종 답변] ---\n", final_response)

                if final_response:
                    audio_to_play, rate = text_to_speech_generate(final_response)
                    input("\n▶️  준비 완료! 재생하시려면 Enter 키를 누르세요...")
                    play_audio(audio_to_play, rate)
            else:
                print("...음성이 인식되지 않았습니다.")

        except KeyboardInterrupt:
            print("\n실시간 대화 보조 모드를 종료하고 메뉴로 돌아갑니다.")
            break
        except Exception as e:
            print(f"🚨 실시간 처리 중 오류가 발생했습니다: {e}")
            break

# 기능 B: 녹음 파일 변환 모드
def start_file_upload_mode():
    """
    사용자로부터 파일 경로를 입력받아 파이프라인을 한 번 실행합니다.
    """
    print("\n📁 녹음 파일 변환 모드입니다.")
    try:
        input_audio_path = input("처리할 음성 파일의 경로를 입력하세요 (예: my_voice.wav): ")
        if not os.path.exists(input_audio_path):
            print(f"🚨 오류: 입력 파일 '{input_audio_path}'을(를) 찾을 수 없습니다.")
            return

        # 원본 파일 로드
        audio_data, sample_rate = sf.read(input_audio_path)

        # [수정됨] 노이즈 제거 적용
        clean_audio_data = reduce_noise(audio_data, sample_rate)

        # 노이즈 제거된 파일을 임시로 저장
        clean_audio_path = "temp_clean_file.wav"
        sf.write(clean_audio_path, clean_audio_data, sample_rate)

        raw_transcript = speech_to_text(clean_audio_path)
        print("\n--- [Whisper STT 원본 결과] ---\n", raw_transcript)

        if raw_transcript and len(raw_transcript.strip()) > 0:
            final_response = correct_and_process_text(raw_transcript)
            print("\n--- [GPT-4o 최종 답변] ---\n", final_response)
            if final_response:
                audio_data, sampling_rate = text_to_speech_generate(final_response)
                play_audio(audio_data, sampling_rate)
        else:
            print("...음성이 인식되지 않았습니다.")

    except Exception as e:
        print(f"🚨 파일 처리 중 오류가 발생했습니다: {e}")


# 기능 C: 발음 교정 학습 모드
def start_pronunciation_practice_mode():
    """
    제시된 문장을 따라 읽고 발음의 정확도를 피드백 받습니다.
    """
    practice_sentences = [
        "오늘은 날씨가 정말 좋네요.",
        "학교 다녀오겠습니다.",
        "이 사과는 얼마인가요?",
        "만나서 반갑습니다.",
        "즐거운 하루 보내세요."
    ]
    samplerate = 16000
    filename = "practice_recording.wav"

    print("\n🎓 발음 교정 학습 모드를 시작합니다. Ctrl+C를 눌러 메뉴로 돌아갑니다.")

    while True:
        try:
            target_sentence = random.choice(practice_sentences)
            print("\n---------------------------------------------")
            print(f"🎯 연습 문장: {target_sentence}")
            print("---------------------------------------------")

            input("준비가 되면 Enter 키를 눌러 녹음을 시작하세요...")

            q = queue.Queue()
            def audio_callback(indata, frames, time, status):
                if status: print(status)
                q.put(indata.copy())

            with sd.InputStream(samplerate=samplerate, channels=1, callback=audio_callback):
                print("... 녹음 시작. 다 읽고 Enter 키를 누르세요 ...")
                input()
                print("... 녹음 완료. 발음을 분석합니다 ...")

            recording = []
            while not q.empty():
                recording.append(q.get())

            if not recording:
                print("...녹음된 내용이 없습니다.")
                continue

            recording = np.concatenate(recording, axis=0)

            # [수정됨] 노이즈 제거 적용
            clean_recording = reduce_noise(recording, samplerate)
            sf.write(filename, clean_recording, samplerate)

            user_transcript = speech_to_text(filename)

            print("\n--- [📈 발음 분석 결과] ---")
            print(f"정답 문장: {target_sentence}")
            print(f"나의 발음: {user_transcript}")
            print("---------------------------")

            # 문장 비교 및 결과 출력
            matcher = difflib.SequenceMatcher(None, target_sentence, user_transcript)
            correct_count = 0
            total_count = 0

            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                target_part = target_sentence[i1:i2]
                user_part = user_transcript[j1:j2]
                total_count += len(target_part.replace(" ", ""))

                if tag == 'equal':
                    print(f"  [ O ] 일치: '{target_part}'")
                    correct_count += len(target_part.replace(" ", ""))
                elif tag == 'replace':
                    print(f"  [ X ] 교체: '{target_part}' → '{user_part}'")
                elif tag == 'delete':
                    print(f"  [ - ] 누락: '{target_part}'")
                elif tag == 'insert':
                    print(f"  [ + ] 추가: '{user_part}'")

            if total_count > 0:
                accuracy = (correct_count / total_count) * 100
                print(f"\n✨ 정확도: {accuracy:.2f}%")
            else:
                print("\n... 분석할 발음이 없습니다.")

            print("---------------------------------------------")
            another = input("다른 문장으로 더 연습하시겠어요? (y/n): ")
            if another.lower() != 'y':
                break

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"🚨 학습 진행 중 오류가 발생했습니다: {e}")
            break

    print("\n발음 교정 학습을 종료하고 메뉴로 돌아갑니다.")


# 메인 메뉴 실행
if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("🚨 오류: OPENAI_API_KEY 환경 변수가 설정되지 않았습니다. 프로그램을 시작할 수 없습니다.")
    else:
        while True:
            print("\n=============================================")
            print("  AI 의사소통 보조 솔루션 (프로토타입)")
            print("=============================================")
            print("1. 실시간 대화 보조 시작 (마이크 사용)")
            print("2. 녹음 파일 변환 (파일 경로 입력)")
            print("3. 발음 교정 학습")
            print("4. 프로그램 종료")
            print("---------------------------------------------")

            choice = input("원하는 기능의 번호를 입력하세요: ")

            if choice == '1':
                start_realtime_mode()
            elif choice == '2':
                start_file_upload_mode()
            elif choice == '3':
                start_pronunciation_practice_mode()
            elif choice == '4':
                print("프로그램을 종료합니다.")
                break
            else:
                print("🚨 잘못된 번호입니다. 1, 2, 3, 4 중에서 다시 선택해주세요.")