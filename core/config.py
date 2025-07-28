# core/config.py
#한글 자음/모음과 그에 해당하는 혀 위치/입 모양 이미지 파일을 연결(매핑)하는 설정 부분

#데이터의 원본 목록
PRONUNCIATION_GUIDES = [       
    {"chars": ["ㄱ", "ㄲ", "ㅋ", "ㅇ"], "imageFile": "c-g.png"},
    {"chars": ["ㄴ", "ㄷ", "ㄸ", "ㅌ"], "imageFile": "c-n.png"},
    {"chars": ["ㄹ"], "imageFile": "c-r.png"},
    {"chars": ["ㅁ", "ㅂ", "ㅃ", "ㅍ"], "imageFile": "c-m.png"},
    {"chars": ["ㅅ", "ㅆ", "ㅈ", "ㅉ", "ㅊ"], "imageFile": "c-s.png"},
    {"chars": ["변이음_ㅅ", "변이음_ㅆ"], "imageFile": "c-s-alt.png"},
    {"chars": ["ㅏ"], "imageFile": "v-a.png"},
    {"chars": ["ㅔ", "ㅐ"], "imageFile": "v-e.png"},
    {"chars": ["ㅓ", "ㅗ"], "imageFile": "v-eo.png"},
    {"chars": ["ㅣ", "ㅑ", 'ㅒ', "ㅕ", "ㅖ", "ㅛ", "ㅠ"], "imageFile": "v-i.png"},
    {"chars": ["ㅡ", "ㅜ", "ㅘ", "ㅙ", "ㅚ", "ㅝ", "ㅞ", "ㅟ", "ㅢ"], "imageFile": "v-u.png"}
]

#PRONUNCIATION_GUIDES 목록을 빠른 조회를 위한 사전(Dictionary) 형태로 가공한 것
IMAGE_GUIDE_MAP = {char: guide["imageFile"] for guide in PRONUNCIATION_GUIDES for char in guide["chars"]}


