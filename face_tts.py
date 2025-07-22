from transformers import VitsModel, AutoProcessor
import torch
#facebook tts 모델 사용했습니다ㅏ


# AutoProcessor를 사용하여 토크나이저와 다른 전처리기를 함께 로드
# VitsModel은 텍스트를 처리할 때 단순히 토크나이징만 하는 것이 아니라
# 음성 합성에 필요한 다양한 전처리 단계를 거칩니다.
processor = AutoProcessor.from_pretrained("facebook/mms-tts-kor")
model = VitsModel.from_pretrained("facebook/mms-tts-kor")

# 한국어 텍스트 입력
text = "오늘은 냉면을 먹었습니다. 오는 길에 비가 왔습니다."

# 텍스트를 모델이 요구하는 형식으로 전처리
# return_tensors="pt"는 PyTorch 텐서로 반환하라는 의미입니다.
# is_split_into_words=True는 필요에 따라 추가될 수 있습니다. (일반 텍스트에는 필요 없음)
inputs = processor(text=text, return_tensors="pt")

# GPU가 있다면 GPU로 모델과 입력을 이동
if torch.cuda.is_available():
    model = model.to("cuda")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

# 추론 (음성 합성)
with torch.no_grad():
    output = model(**inputs).waveform

# 생성된 음성 파형 출력 (예: 재생하거나 파일로 저장)
# print(output)
# print(output.shape) # 예: torch.Size([1, 48000])


import soundfile as sf
sampling_rate = model.config.sampling_rate
sf.write("output.wav", output.cpu().numpy().squeeze(), sampling_rate)
print(f"음성 파일이 output.wav로 저장되었습니다. 샘플링 레이트: {sampling_rate} Hz")