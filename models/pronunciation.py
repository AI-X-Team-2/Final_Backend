# models/pronunciation.py
# Pydantic 라이브러리를 사용해 API가 주고받을 데이터의 형식(Schema)을 정의하는 부분

#틀린 글자 한 개'에 대한 상세 피드백 정보의 설계도
from pydantic import BaseModel
from typing import List

class IncorrectPoint(BaseModel):
    expected: str
    actual: str
    img: str
    diff_detail: str
    mouth_shape: str
    tongue_shape: str
    breathing: str

#/analyze API가 프론트엔드에 보내는 최종 응답 전체의 설계도
class PronunciationAnalysisResponse(BaseModel):
    score: str
    transcription: str
    incorrect_points: List[IncorrectPoint]