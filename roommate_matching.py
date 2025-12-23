import os
import numpy as np
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

# .env 파일 로드
load_dotenv()

# ==========================================
# 🔑 Upstage API 설정
# ==========================================
UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")
client = OpenAI(
    api_key=UPSTAGE_API_KEY,
    base_url="https://api.upstage.ai/v1/solar"
)

# ==========================================
# 👤 사용자 클래스 정의
# ==========================================
class User:
    def __init__(self, user_id, name, gender, birth_year, sleep_time, wake_time, cleaning_cycle, drinking_freq, 
                 is_smoker, is_snorer, light_sleeper, good_at_bugs, heavy_sleeper, 
                 mbti, intro_text):
        self.id = user_id
        self.name = name
        self.gender = gender # 'Male', 'Female'
        self.birth_year = birth_year
        
        # 생활 습관 (숫자형)
        self.sleep_time = sleep_time # 0~24
        self.wake_time = wake_time   # 0~24
        self.cleaning_cycle = cleaning_cycle # 0(매일) ~ 4(안함)
        self.drinking_freq = drinking_freq   # 0(안함) ~ 2(자주)
        
        # 특징 (Boolean)
        self.is_smoker = is_smoker
        self.is_snorer = is_snorer
        self.light_sleeper = light_sleeper
        self.good_at_bugs = good_at_bugs
        self.heavy_sleeper = heavy_sleeper
        
        # 기타
        self.mbti = mbti
        self.intro_text = intro_text
        self.embedding = None # 나중에 계산

    @property
    def age(self):
        current_year = datetime.now().year
        return current_year - self.birth_year

    def __repr__(self):
        return f"<User {self.name} ({self.age}세/{self.gender})>"

# ==========================================
# 🧠 임베딩 함수
# ==========================================
def get_embedding(text, model_type):
    """
    model_type: 'passage' (DB 저장용) or 'query' (검색용)
    """
    model_name = f"solar-embedding-1-large-{model_type}"
    response = client.embeddings.create(
        input=text,
        model=model_name
    )
    return np.array(response.data[0].embedding)

# ==========================================
# 📏 거리 계산 헬퍼 함수
# ==========================================
def get_time_diff(t1, t2):
    """24시간 기준 시간 차이 계산 (예: 23시와 01시는 2시간 차이)"""
    diff = abs(t1 - t2)
    return min(diff, 24 - diff)

def get_linear_diff_score(val1, val2, max_range):
    """선형 차이 점수 (0~1점)"""
    diff = abs(val1 - val2)
    return max(0, 1 - (diff / max_range))

# ==========================================
# 🚀 매칭 점수 계산 로직
# ==========================================
def calculate_score(seeker, candidate, preferences, query_embedding):
    """
    seeker: 찾는 사람 (User 객체) - 본인의 습관 기준 비교용
    candidate: 후보자 (User 객체)
    preferences: 선호 조건 딕셔너리
    query_embedding: 찾는 사람의 '원하는 룸메' 텍스트 임베딩
    """
    
    # 1. 🛑 Hard Filter (성별) - 제거됨 (Controller 레벨에서 필터링 가정)
    # if preferences['target_gender'] != candidate.gender:
    #     return 0 

    score = 0
    total_weight = 0
    
    # 2. 📅 Age Filter (나이 차이) - 감점 방식 -> 점수화
    # 본인 나이 기준 대조 (preferences에 나이 범위 없음)
    age_diff = abs(seeker.age - candidate.age)
    
    # 나이 차이가 적을수록 점수 높음 (5살 차이까지는 어느정도 점수 부여)
    # 0살 차이: 100점, 1살: 90점 ... 10살 이상: 0점
    age_score = max(0, 100 - (age_diff * 10))
    
    score += age_score * 0.5 # 가중치 0.5로 축소
    total_weight += 0.5

    # 3. 🔢 Tag Similarity (생활 습관 숫자 비교)
    # 나와 비슷한 사람을 원한다고 가정 (seeker의 속성과 비교)
    
    # 시간 관련 (가중치 높음)
    time_score = (
        (1 - get_time_diff(seeker.sleep_time, candidate.sleep_time) / 12) + # 12시간이 최대 차이
        (1 - get_time_diff(seeker.wake_time, candidate.wake_time) / 12)
    ) / 2 * 100
    score += time_score * 2.0 # 가중치 2.0
    total_weight += 2.0
    
    # 청소/음주 (가중치 보통)
    habit_score = (
        get_linear_diff_score(seeker.cleaning_cycle, candidate.cleaning_cycle, 4) +
        get_linear_diff_score(seeker.drinking_freq, candidate.drinking_freq, 2)
    ) / 2 * 100
    score += habit_score * 1.5
    total_weight += 1.5

    # 4. ✅ Preference Check (선택 사항 우선순위)
    # 체크리스트에 해당하는 항목 하나당 큰 보너스 점수
    bonus_points = 0
    
    if preferences.get('prefer_non_smoker') and not candidate.is_smoker:
        bonus_points += 50 # 50점 보너스 (매우 큼)
    if preferences.get('prefer_good_at_bugs') and candidate.good_at_bugs:
        bonus_points += 30
    if preferences.get('prefer_quiet_sleeper') and not candidate.is_snorer:
        bonus_points += 30
    if preferences.get('prefer_non_drinker') and candidate.drinking_freq == 0:
        bonus_points += 40
        
    score += bonus_points 
    # 보너스 점수는 total_weight에 포함하지 않음 (순수 가산점)

    # 5. 📝 Text Semantic Similarity
    if query_embedding is not None and candidate.embedding is not None:
        text_sim = cosine_similarity([query_embedding], [candidate.embedding])[0][0]
        # 유사도(-1~1)를 0~100점으로 변환 (음수는 0처리)
        text_score = max(0, text_sim) * 100
        score += text_score * 1.0 # 텍스트 가중치
        total_weight += 1.0
        
    final_score = score # 가중 평균을 내지 않고 합산 점수로 (보너스가 있어서)
    
    return final_score

# ==========================================
# 🏃‍♂️ 실행 (Main)
# ==========================================
if __name__ == "__main__":
    
    # 1. 유저 데이터 베이스 생성 (User DB)
    # 2025년 기준 나이 역산
    # 24세 -> 2001, 26세 -> 1999, 21세 -> 2004, 23세 -> 2002, 22세 -> 2003
    users_db = [
        User(1, "김철수", "Male", 2001, sleep_time=23, wake_time=7, cleaning_cycle=0, drinking_freq=0,
             is_smoker=False, is_snorer=False, light_sleeper=True, good_at_bugs=False, heavy_sleeper=False,
             mbti="ISTJ", intro_text="조용하고 규칙적인 생활을 합니다. 매일 청소하고 일찍 자는 편입니다."),
             
        User(2, "이영만", "Male", 1999, sleep_time=2, wake_time=10, cleaning_cycle=2, drinking_freq=2,
             is_smoker=True, is_snorer=True, light_sleeper=False, good_at_bugs=True, heavy_sleeper=True,
             mbti="ENFP", intro_text="사람들과 어울리는걸 좋아하고 술자리도 즐깁니다. 벌레는 제가 다 잡아드려요."),
             
        User(3, "박민준", "Male", 2004, sleep_time=0, wake_time=8, cleaning_cycle=1, drinking_freq=1,
             is_smoker=False, is_snorer=False, light_sleeper=False, good_at_bugs=True, heavy_sleeper=False,
             mbti="ISFP", intro_text="적당히 깔끔하고 조용한 편입니다. 게임하는거 좋아해요."),
             
        User(4, "최준호", "Male", 2002, sleep_time=23, wake_time=7, cleaning_cycle=0, drinking_freq=0,
             is_smoker=False, is_snorer=False, light_sleeper=True, good_at_bugs=True, heavy_sleeper=False,
             mbti="ESTJ", intro_text="군필이고 생활패턴 칼같습니다. 깔끔한 방 원합니다. 비흡연자 환영."),
             
        User(5, "정수아", "Female", 2003, sleep_time=23, wake_time=7, cleaning_cycle=0, drinking_freq=0,
             is_smoker=False, is_snorer=False, light_sleeper=True, good_at_bugs=False, heavy_sleeper=False,
             mbti="INFJ", intro_text="여성 룸메이트 구해요.") # 성별 필터 테스트용
    ]

    print("⏳ 유저 데이터 임베딩 생성 중...")
    for u in users_db:
        if u.intro_text:
            u.embedding = get_embedding(u.intro_text, "passage")

    # 2. 검색 요청자 (나) 설정
    # 23세 -> 2002년생
    my_profile = User(99, "나(사용자)", "Male", 2002, 
                      sleep_time=24, # 0시
                      wake_time=7, 
                      cleaning_cycle=0, # 매일
                      drinking_freq=0,  # 안함
                      is_smoker=False, is_snorer=False, light_sleeper=True, good_at_bugs=False, heavy_sleeper=False,
                      mbti="INTJ", intro_text="")

    # 3. 나의 검색 조건 (Preferences)
    my_preferences = {
        # 'target_gender': 'Male',      # 제거
        # 'target_age_range': (20, 25), # 제거 (내 나이 기준 매칭)
        
        # 선택 사항 (체크리스트) -> 가산점
        'prefer_non_smoker': True,    # 흡연 안하는 사람 (매우 중요)
        'prefer_good_at_bugs': True,  # 벌레 잘 잡는 사람
        'prefer_quiet_sleeper': True, # 코 안고는 사람
        'prefer_non_drinker': True    # 술 안마시는 사람
    }
    
    # 내가 원하는 룸메이트 텍스트 (추가 가산점)
    my_query_text = "조용하고 깨끗한 사람, 아침형 인간을 선호합니다."
    print(f"\n🔍 검색 조건 텍스트: \"{my_query_text}\"")
    
    query_emb = get_embedding(my_query_text, 'query')

    # 4. 매칭 실행
    print("\n🔄 매칭 계산 중...")
    results = []
    
    for candidate in users_db:
        if candidate.id == my_profile.id: continue
        
        score = calculate_score(my_profile, candidate, my_preferences, query_emb)
        results.append((candidate, score))
    
    # 점수 높은 순 정렬
    results.sort(key=lambda x: x[1], reverse=True)
    
    # 5. 결과 출력
    print(f"\n=== 🏠 룸메이트 추천 결과 (User: {my_profile.name}) ===")
    print(f"내 정보: {my_profile.age}세/남, 취침 {my_profile.sleep_time}시, 기상 {my_profile.wake_time}시, 비흡연, 매일청소\n")
    
    for i, (user, score) in enumerate(results):
        print(f"{i+1}위. {user.name} ({user.age}세 - {user.birth_year}년생, {user.mbti}) | 점수: {score:.1f}점")
        print(f"    - 생활: 취침 {user.sleep_time}시, 기상 {user.wake_time}시, 흡연: {'O' if user.is_smoker else 'X'}, 음주: {user.drinking_freq}")
        print(f"    - 특징: {'벌레잘잡음 ' if user.good_at_bugs else ''}{'코골이 ' if user.is_snorer else ''}")
        print(f"    - 소개: {user.intro_text}")
        print("-" * 50)