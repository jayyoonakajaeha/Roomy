from app.users.service import save_user_vectors
import sys

def main():
    print("=== 유저 벡터 생성 도구 ===")
    
    # 1. 사용자 ID 입력
    try:
        user_id = int(input("User ID를 입력하세요 (예: 100): "))
    except ValueError:
        print("올바른 숫자를 입력해주세요.")
        return

    # 2. 텍스트 입력
    print("\n[자기소개 입력]")
    print("예: 저는 조용하고 깔끔한 성격입니다. 밤 11시면 자고 아침 7시에 일어납니다.")
    self_desc = input("CONTENT: ")

    print("\n[원하는 룸메이트상 입력]")
    print("예: 비흡연자였으면 좋겠고, 밤늦게 시끄럽게 하지 않는 분 찾습니다.")
    room_desc = input("CONTENT: ")

    # 3. 벡터 생성 및 저장
    print(f"\nUser ID {user_id}의 벡터를 생성 중입니다...")
    save_user_vectors(user_id, self_desc, room_desc)
    
    print("\n✅ 생성 완료!")
    print(f"- 자기소개 벡터: storage/vectors/{user_id}_self.npy")
    print(f"- 룸메이트상 벡터: storage/vectors/{user_id}_criteria.npy")

if __name__ == "__main__":
    main()
