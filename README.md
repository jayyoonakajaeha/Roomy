# 룸메이트 매칭 & 시설 고장 신고 서비스

AI 기반의 룸메이트 매칭 및 시설물 고장 자동 분석 백엔드 서비스입니다.

## 주요 기능

### 1. 룸메이트 매칭 (`/api/matching`)
- **하이브리드 점수 시스템 (100점 만점)**:
  - **태그 점수 (40점)**:
    - **나이 (5점)**: 사용자(Seeker)와 후보자의 나이 차이 기반 점수 (차이가 클수록 감점).
    - **생활 시간 (20점)**: 기상 시간(10점) + 취침 시간(10점) 차이 분석.
    - **생활 습관 (15점)**: 청소 주기(7.5점) + 음주 빈도(7.5점) 차이 분석.
  - **선호 조건 (30점)**: 비흡연, 벌레 잡기, 코골이 여부 등 사용자의 선호 조건 만족 비율에 따른 배점.
  - **텍스트 유사도 (30점)**: FAISS 및 임베딩을 활용한 자기소개/룸메이트상 의미 기반 매칭.
- **필터링**: 성별 필터링은 API 요청 전 단계에서 수행됨을 가정합니다 (API 파라미터에서 제외).

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
# Upstage Console(https://console.upstage.ai/docs/getting-started)에서 API Key를 발급받으세요.
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
    "preferNonSmoker": true,
    "preferGoodAtBugs": true
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
  },
]
```

### 2. 시설 고장 신고 API
*   **URL**: `/api/repair/analyze`
*   **Method**: `POST`
*   **Content-Type**: `multipart/form-data`
*   **설명**: 고장난 시설물 이미지를 전송하면 AI가 분석 결과를 반환합니다. 중복 신고 감지 기능이 포함되어 있습니다.

#### Request
*   `file`: 이미지 파일 (binary)
*   `building`: 건물명 (예: "Dorm A")
*   `floor`: 층수 (예: "3")
*   `room_number`: (선택) 호수 (예: "1201"). 개인 시설인 경우 입력, 공용 시설인 경우 생략.

#### Response 예시
```json
{
  "analysis": {
    "category": "plumbing",
    "item": "toilet",
    "issue": "clogged",
    "severity": "CRITICAL",
    "priority_score": 9,
    "reasoning": "변기 역류로 인한 위생 문제 발생, 즉각 조치 필요.",
    "repair_suggestion": "배관 전문 업체 호출 필요.",
    "description": "변기가 막혀 물이 넘치고 있습니다."
  },
  "duplicates": [
    {
      "reportId": 1024,
      "similarity": 0.95,
      "description": "3층 화장실 변기 역류 (7분 전 신고됨)",
      "location": "Dorm A 3F"
    }
  ],
  "is_new": false
}
```

## DB 스키마 가이드 (백엔드 참고용)

### `RepairReport` (고장 신고 테이블)
| Field Name | Type | Key | Nullable | Description |
|---|---|---|---|---|
| `id` | BigInt | PK | NO | Auto Increment ID |
| `building` | Varchar(50) | IDX | NO | 건물명 (예: "Dorm A") |
| `floor` | Varchar(10) | IDX | NO | 층수 (예: "3") |
| `room_number` | Varchar(20) | | YES | 호수 (선택) |
| `category` | Varchar(50) | | NO | 고장 카테고리 (plumbing, electric 등) |
| `item` | Varchar(100) | | NO | 고장 물품 (예: toilet, sink) |
| `issue` | Varchar(100) | | NO | 증상 (예: clogged, leakage) |
| `severity` | Enum | | NO | 심각도 ('CRITICAL', 'HIGH', 'MEDIUM', 'LOW') |
| `priority_score` | Int | | NO | AI 산정 우선순위 점수 (1~10) |
| `reasoning` | Text | | YES | AI의 우선순위 판단 근거 |
| `description` | Text | | NO | 상황 설명 (AI 작성 초안 or 사용자 수정본) |
| `reporter_id` | BigInt | | NO | 신고자 User ID |
| `status` | Enum | | NO | 상태 ('PENDING', 'IN_PROGRESS', 'DONE') |
| `image_url` | Varchar(255) | | NO | 원본 이미지 저장 경로 |
| `embedding` | Vector(512)* | | YES | CLIP 이미지 임베딩 (중복 검사용) |
| `created_at` | DateTime | | NO | 생성 일시 |
