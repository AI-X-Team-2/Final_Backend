# routes/pronunciation.py
from fastapi import APIRouter, Form, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from services.analysis_service import analyze_user_pronunciation

router = APIRouter()

@router.get("/")
async def root():
    return {"message": "의사소통 보조 AI 유음"}

#경로(Endpoint) 정의
@router.post("/analyze", tags=["Pronunciation Analysis"])
async def analyze_pronunciation_endpoint(target_sentence: str = Form(...), audio_file: UploadFile = File(...)):
    """사용자의 발음을 분석하고, 틀린 글자별 상세 피드백을 반환합니다."""
    #최종 응답 및 예외 처리
    try:
        response_data = await analyze_user_pronunciation(target_sentence, audio_file)  #핵심 기능 위임 (Delegation)
        return JSONResponse(content=response_data)
    except Exception as e:
        print(f"발음 분석 중 심각한 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=f"서버에서 오디오 파일을 처리하는 중 오류가 발생했습니다: {str(e)}")