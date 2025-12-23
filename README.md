# 룸메이트 매칭 & 시설 고장 신고 서비스

AI 기반의 룸메이트 매칭 및 시설물 고장 자동 분석 백엔드 서비스입니다.

## 주요 기능

### 1. 룸메이트 매칭 (`/api/matching`)
- **하이브리드 점수 시스템 (100점 만점)**:
  - **태그 점수 (40점)**: 나이, 기상/취침 시간, 생활 습관(청소/음주 등)의 유사도 분석.
  - **선호 조건 (30점)**: 비흡연, 벌레 잡기 가능 여부 등 사용자의 선호 조건 만족 시 가산점.
  - **텍스트 유사도 (30점)**: FAISS 및 임베딩을 활용한 자기소개/룸메이트상 의미 기반 매칭.
- **필터링**: 성별(필수), 나이 범위 등.

### 2. 시설 고장 신고 (`/api/repair`)
- **AI Vision 분석**: 고장난 시설물 사진을 업로드하면 AI가 이미지를 분석.
- **자동 정보 추출**: 고장 카테고리(배관, 전기 등), 항목, 증상, 심각도 등을 자동으로 JSON 데이터로 반환.

## 기술 스택
- **프레임워크**: FastAPI
- **AI/ML**: 
  - `FAISS` (벡터 검색)
  - `Upstage Solar` (임베딩 및 Vision/Chat API 연동 가능)
  - `OpenAI` (Vision Model 호환 구조)
- **Data Validation**: Pydantic

## 설치 및 실행 방법

```bash
# 1. 의존성 패키지 설치
pip install -r requirements.txt

# 2. API 키 설정
# .env 파일을 생성하고 아래 내용을 추가하세요:
# UPSTAGE_API_KEY=your_key_here
```

## 사용 방법

```bash
# 서버 실행
uvicorn app.main:app --reload
```

## API 문서
서버가 실행 중일 때 `http://localhost:8000/docs` 로 접속하면 Swagger UI를 통해 API를 직접 테스트해볼 수 있습니다.

### 1. 룸메이트 매칭 API
*   **URL**: `/api/matching/match`
*   **Method**: `POST`
*   **설명**: 내 프로필과 후보자 리스트를 받아 점수 순으로 정렬된 매칭 결과를 반환합니다.

#### Request Body 예시
```json
{
  "myProfile": {
    "id": 99,
    "gender": "MALE",
    "birthYear": 2002,
    "name": "내 이름",
    "smoker": false,
    "sleepTime": 14,
    "wakeTime": 7,
    "cleaningCycle": "DAILY",
    "drinkingStyle": "RARELY",
    "snoring": false,
    "bugKiller": false,
    "roommateCriteriaEmbedding": [0.1, 0.2, ...] // 내가 원하는 룸메이트 상 벡터
  },
  "preferences": {
    "targetGender": "MALE",
    "targetAgeRange": [20, 25],
    "preferNonSmoker": true
  },
  "candidates": [
    {
      "id": 1,
      "name": "후보자1",
      "gender": "MALE",
      "birthYear": 2002,
      "selfIntroductionEmbedding": [0.1, 0.2, ...] // 후보자의 자기소개 벡터
      // ... 기타 프로필 필드
    }
  ]
}
```

#### Response 예시
```json
[
  {
    "userId": 1,
    "name": "후보자1",
    "totalScore": 95.5,
    "rank": 1,
    "matchDetails": {
      "tagScore": 40.0,
      "prefScore": 30.0,
      "textScore": 25.5,
      "age": 23
    }
  }
]
```

### 2. 시설 고장 신고 API
*   **URL**: `/api/repair/analyze`
*   **Method**: `POST`
*   **Content-Type**: `multipart/form-data`
*   **설명**: 고장난 시설물 이미지를 전송하면 AI가 분석 결과를 반환합니다.

#### Request
*   `file`: 이미지 파일 (binary)

#### Response 예시
```json
{
  "category": "plumbing",
  "item": "faucet",
  "issue": "leakage",
  "severity": "medium",
  "description": "수도꼭지에서 물이 새고 있습니다."
}
```
