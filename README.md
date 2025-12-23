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
- **AI Vision 분석**: 고장난 시설물 사진을 업로드하면 **Gemini 3 Flash**가 이미지를 분석.
- **자동 정보 추출**: 고장 항목, 증상, 심각도, 우선순위 등을 자동으로 JSON 데이터로 반환.
- **중복 신고 감지**: **CLIP (clip-ViT-B-32)** 임베딩 기반 이미지 유사도 비교로 중복 판단.

## 기술 스택
- **프레임워크**: FastAPI
- **AI/ML**: 
  - `FAISS` (벡터 검색)
  - `Gemini 3 Flash Preview` (이미지 분석)
  - `CLIP (clip-ViT-B-32)` (이미지 임베딩 & 중복 감지)
  - `Upstage Solar` (텍스트 임베딩)
- **Data Validation**: Pydantic

## 설치 및 실행 방법

```bash
# 1. 의존성 패키지 설치
pip install -r requirements.txt

# 2. API 키 설정
# .env 파일을 생성하고 아래 내용을 추가하세요:
# UPSTAGE_API_KEY=your_upstage_key
# GOOGLE_API_KEY=your_google_key
```

## 사용 방법

```bash
# 서버 실행
uvicorn app.main:app --reload --port 8001
```

## API 문서
서버가 실행 중일 때 `http://localhost:8001/docs` 로 접속하면 Swagger UI를 통해 API를 직접 테스트해볼 수 있습니다.

---

### 1. 룸메이트 매칭 API
*   **URL**: `/api/matching/match`
*   **Method**: `POST`
*   **설명**: 내 프로필과 후보자 리스트를 받아 점수 순으로 정렬된 매칭 결과를 반환합니다.

#### 시간대 필드 스키마
시간 필드는 실제 시간이 아닌 **시간대 인덱스**를 사용합니다:

**기상 시간 (`wakeTime`)**:
- `5`: 오전 6시 이전
- `6`: 오전 6시~7시
- `7`: 오전 7시~8시
- `8`: 오전 8시~9시
- `9`: 오전 9시~10시
- `10`: 오전 10시~11시
- `11`: 오전 11시 이후

**취침 시간 (`sleepTime`)**:
- `8`: 오후 9시 이전
- `9`: 오후 9시~10시
- `10`: 오후 10시~11시
- `11`: 오후 11시~12시
- `12`: 오전 12시~1시
- `13`: 오전 1시~2시
- `14`: 오전 2시 이후

#### Request Body 예시
```json
{
  "myProfile": {
    "id": 99,
    "gender": "MALE",
    "name": "홍길동",
    "birthYear": 2002,
    "kakaoId": "hong123",
    "mbti": "INTJ",
    "smoker": false,
    "sleepTime": 11,
    "wakeTime": 7,
    "snoring": true,
    "cleaningCycle": "EVERY_TWO_DAYS",
    "drinkingStyle": "RARELY",
    "bugKiller": false,
    "absentDays": ["SUNDAY"],
    "hobby": "독서"
  },
  "preferences": {
    "preferNonSmoker": true,
    "preferGoodAtBugs": true,
    "preferQuietSleeper": false
  },
  "candidates": [
    {
      "id": 1,
      "gender": "MALE",
      "name": "후보자1",
      "birthYear": 2002,
      "kakaoId": "candidate1",
      "mbti": "ENFP",
      "smoker": false,
      "sleepTime": 12,
      "wakeTime": 8,
      "snoring": false,
      "cleaningCycle": "WEEKLY",
      "drinkingStyle": "SOMETIMES",
      "bugKiller": true,
      "absentDays": [],
      "hobby": "게임"
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
    "totalScore": 79.0,
    "rank": 1,
    "matchDetails": {
      "tagScore": 40.0,
      "prefScore": 22.5,
      "textScore": 16.5,
      "age": 23
    }
  }
]
```

---

### 2. 시설 고장 신고 API
*   **URL**: `/api/repair/analyze`
*   **Method**: `POST`
*   **Content-Type**: `application/json`
*   **설명**: 고장난 시설물 이미지를 분석하고 중복 신고 여부를 확인합니다.

#### Request 예시 (JSON)
```json
{
  "existingReportIds": [1024, 1025, 1030],
  "totalReportCount": 1030,
  "floor": "3",
  "room_number": "301"
}
```

> **이미지 경로**: 고정 경로 `storage/temp/pending.jpg` 사용 (request에서 제외)
> 프론트는 이 경로에 이미지를 저장한 후 API 호출

**필드 설명:**
| 필드명 | 타입 | 필수 | 설명 |
|--------|------|------|------|
| `existingReportIds` | int[] | ✓ | 백엔드에서 **위치(층/호수) 필터링한 기존 게시물 ID 목록** |
| `totalReportCount` | int | ✓ | 현재 총 게시물 수 (새 ID = totalReportCount + 1) |
| `floor` | string | ✓ | 층수 |
| `room_number` | string | | 호수 (공용시설이면 null 또는 생략) |

**중복 감지 흐름:**
1. 백엔드: 신규 신고의 `floor`/`room_number`와 일치하는 기존 게시물 ID 조회
2. 이 API 호출: `existingReportIds`에 해당 ID 목록 전달
3. API 내부: 신규 이미지 CLIP 벡터 계산 → 기존 게시물 벡터와 비교
4. 유사도 80% 이상이면 중복으로 판정

#### Response 예시

**중복 신고인 경우:**
```json
{
  "analysis": {
    "item": "변기",
    "issue": "배수 불량 (막힘)",
    "severity": "CRITICAL",
    "priority_score": 9,
    "reasoning": "변기 막힘은 위생 문제와 직결...",
    "description": "과도한 화장지 사용으로 인해 변기가 막혀..."
  },
  "duplicates": [
    {
      "reportId": 1024,
      "similarity": 0.95,
      "description": "화장실 변기 역류 (7분 전 신고됨)",
      "location": "3층 301호"
    }
  ],
  "is_new": false,
  "newReportId": null
}
```

**새로운 신고인 경우:**
```json
{
  "analysis": { ... },
  "duplicates": [],
  "is_new": true,
  "newReportId": 1031
}
```
> 새 신고일 때만 `newReportId`가 할당되고, 임베딩/이미지 파일이 자동 저장됩니다.
> - `storage/repair_vectors/1031.npy`
> - `storage/repair_images/1031.jpg`

---

## 백엔드 통합 가이드 (Backend Integration)

### 1. 사용자 프로필 저장 및 임베딩 생성
사용자가 회원가입하거나 프로필을 수정할 때, **DB 저장과 동시에 임베딩 벡터 파일을 생성**해야 합니다.

1.  **DB 저장**: 사용자 입력 정보(JSON)를 RDB(MySQL 등)에 저장.
2.  **벡터 생성 및 저장**:
    *   `app/users/service.py`의 `save_user_vectors` 함수 활용.
    *   `selfDescription` -> `{user_id}_self.npy` (후보자 검색용)
    *   `roommateDescription` -> `{user_id}_criteria.npy` (매칭 쿼리용)
    *   저장 위치: `storage/vectors/`

### 2. 매칭 API 호출 흐름
매칭 요청 시(`POST /api/matching/match`), DB와 벡터 저장소에서 데이터를 조회하여 API에 전달해야 합니다.

1.  **내 정보 로드**: DB에서 내 프로필(`myProfile`) 로드.
2.  **후보자 리스트 로드**: DB에서 성별 등 기본적인 필터링을 거친 후보자 리스트(`candidates`) 로드.
3.  **API 요청**: `POST /api/matching/match` 호출.

### 3. 시설 고장 신고 API 호출 흐름
1. **이미지 저장**: 사용자가 업로드한 이미지를 서버에 저장
2. **위치 필터링**: DB에서 동일 위치(층/호수)의 기존 신고 ID 조회
3. **API 호출**: `POST /api/repair/analyze` (imagePath + existingReportIds)
4. **결과 처리**: 
   - `is_new == true`: 새 신고로 DB에 저장 (임베딩도 함께 저장)
   - `is_new == false`: 중복 신고 안내

---

## DB 스키마 가이드 (백엔드 참고용)

### `RepairReport` (고장 신고 테이블)
| Field Name | Type | Key | Nullable | Description |
|---|---|---|---|---|
| `id` | BigInt | PK | NO | Auto Increment ID |
| `floor` | Varchar(10) | IDX | NO | 층수 (예: "3") |
| `room_number` | Varchar(20) | IDX | YES | 호수 (공용시설이면 NULL) |
| `item` | Varchar(100) | | NO | 고장 물품 (예: 변기, 싱크대) |
| `issue` | Varchar(100) | | NO | 증상 (예: 막힘, 누수) |
| `severity` | Enum | | NO | 심각도 ('CRITICAL', 'HIGH', 'MEDIUM', 'LOW') |
| `priority_score` | Int | | NO | AI 산정 우선순위 점수 (1~10) |
| `reasoning` | Text | | YES | AI의 우선순위 판단 근거 |
| `description` | Text | | NO | 상황 설명 (AI 작성) |
| `reporter_id` | BigInt | FK | NO | 신고자 User ID |
| `status` | Enum | | NO | 상태 ('PENDING', 'IN_PROGRESS', 'DONE') |
| `image_url` | Varchar(255) | | NO | 원본 이미지 저장 경로 |
| `created_at` | DateTime | | NO | 생성 일시 |

> **임베딩 저장 방식**: CLIP 벡터는 DB가 아닌 **로컬 파일**로 저장합니다.
> - 경로: `storage/repair_vectors/{report_id}.npy`
> - 로드: `np.load(f"storage/repair_vectors/{report_id}.npy")`

---

## 심각도 판단 기준 (Priority Logic)

| 등급 | 점수 | 기준 | 예시 |
|------|------|------|------|
| **CRITICAL** | 9-10 | 안전/위생 위협, 주거 불가능 | 변기 역류, 가스 누출, 단전/단수 |
| **HIGH** | 7-8 | 필수 생활 기능 마비 | 난방/에어컨 고장, 냉장고 고장 |
| **MEDIUM** | 4-6 | 불편하나 생활 가능 | 방문 파손, 전등 고장 |
| **LOW** | 1-3 | 미관상 문제 | 벽지 찢어짐, 스크래치 |

---

## 룸메이트 매칭 알고리즘 (Scoring Logic)

### 총점 구성 (100점 만점)

$$
\text{Total Score} = \text{Tag Score}(40) + \text{Preference Score}(30) + \text{Text Score}(30)
$$

---

### A. Tag Score (40점) - 기본 성향 일치도

나(Seeker)와 후보자(Candidate)의 **생활 패턴 차이**를 측정합니다.

#### 1. 나이 점수 (5점)
```python
age_diff = abs(seeker.age - cand.age)
age_base_score = max(0, 100 - (age_diff * 10))  # 0~100
age_score = age_base_score * 0.05               # 0~5점
```
| 나이 차이 | 기본 점수 | 최종 점수 |
|-----------|-----------|-----------|
| 0살 | 100 | 5.0점 |
| 1살 | 90 | 4.5점 |
| 5살 | 50 | 2.5점 |
| 10살+ | 0 | 0점 |

#### 2. 생활 시간 점수 (20점)
**기상 시간 (10점) + 취침 시간 (10점)**

```python
def get_scale_diff_score(val1, val2, max_diff) -> float:
    diff = abs(val1 - val2)
    return max(0.0, 1.0 - (diff / max_diff))  # 0.0~1.0

wake_score = get_scale_diff_score(seeker.wakeTime, cand.wakeTime, 6)
sleep_score = get_scale_diff_score(seeker.sleepTime, cand.sleepTime, 6)
time_score = (wake_score + sleep_score) / 2.0 * 20.0  # 0~20점
```

#### 3. 생활 습관 점수 (15점)
**청소 주기 (7.5점) + 음주 빈도 (7.5점)**

```python
# CleaningCycle: DAILY=0, EVERY_TWO_DAYS=1, WEEKLY=2, MONTHLY=3, NEVER=4
clean_score = get_scale_diff_score(seeker.cleaningCycle, cand.cleaningCycle, 4)

# DrinkingStyle: RARELY=0, SOMETIMES=1, FREQUENTLY=2
drink_score = get_scale_diff_score(seeker.drinkingStyle, cand.drinkingStyle, 2)

habit_score = (clean_score + drink_score) / 2.0 * 15.0  # 0~15점
```

---

### B. Preference Score (30점) - 선호 조건 만족도

사용자가 **체크한 선호 조건**의 만족 비율에 따라 점수 부여.

```python
active_prefs = []
if prefs.preferNonSmoker:   active_prefs.append(lambda u: not u.smoker)
if prefs.preferGoodAtBugs:  active_prefs.append(lambda u: u.bugKiller)  
if prefs.preferQuietSleeper: active_prefs.append(lambda u: not u.snoring)

if len(active_prefs) == 0:
    pref_score = 30.0  # 조건 없으면 만점
else:
    matched = sum(1 for check in active_prefs if check(cand))
    pref_score = (matched / len(active_prefs)) * 30.0
```

| 체크 조건 | 만족 개수 | 점수 |
|-----------|-----------|------|
| 3개 | 3개 | 30점 |
| 3개 | 2개 | 20점 |
| 3개 | 1개 | 10점 |
| 0개 (없음) | - | 30점 (만점) |

---

### C. Text Score (30점) - 텍스트 유사도

**Upstage Solar Embedding** + **FAISS**를 활용한 의미 기반 매칭.

```python
# 1. 벡터 로드 (사전 저장된 .npy 파일)
seeker_vec = load_user_vector(seeker.id, 'criteria')  # 원하는 룸메 설명
cand_vec = load_user_vector(cand.id, 'self')          # 자기소개

# 2. FAISS 코사인 유사도 검색
faiss.normalize_L2(seeker_vec)
faiss.normalize_L2(cand_vec)
similarity = np.dot(seeker_vec, cand_vec.T)  # 0.0~1.0

# 3. 점수 변환
text_score = similarity * 30.0  # 0~30점
```

**사용 모델:**
- `solar-embedding-1-large-passage`: 후보자 자기소개 임베딩
- `solar-embedding-1-large-query`: 내가 원하는 룸메 설명 임베딩

---

### Hard Filter (필터링)

매칭 연산 전 **제외되는 조건**:

```python
# 자기 자신 제외
if cand.id == seeker.id: continue

# 다른 성별 제외
if cand.gender != seeker.gender: continue
```
