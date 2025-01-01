import pandas as pd
import numpy as np
import json

def load_data(rating_file_path: str, travel_meta_path: str):
    """
    CSV로부터 평점 데이터를 로드하고,
    JSON으로부터 여행지 메타데이터를 로드한다.
    """
    # 1) CSV 로드
    ratings_df = pd.read_csv(rating_file_path)
    # 2) JSON 로드
    with open(travel_meta_path, "r", encoding="utf-8") as f:
        travel_meta = json.load(f)

    return ratings_df, travel_meta


def create_item_similarity_matrix(ratings_df: pd.DataFrame):
    """
    아이템-사용자 행렬을 만들고, 아이템 간 코사인 유사도를 계산한 후,
    아이템 유사도 행렬을 반환한다.
    """
    # user_id 행, item_id 열 형태로 pivot
    # 결측치는 0 처리(평점이 없는 경우)
    pivot_df = ratings_df.pivot_table(index="user_id", columns="item_id", values="rating").fillna(0)
    
    # 아이템별로 벡터(열) 뽑기
    item_matrix = pivot_df.T  # shape: (num_items, num_users)
    
    # 코사인 유사도 계산
    # np.dot(A, B) / (||A||*||B||)
    # 직접 계산하거나, sklearn.metrics.pairwise.cosine_similarity 사용 가능
    from sklearn.metrics.pairwise import cosine_similarity

    item_similarity = cosine_similarity(item_matrix.values)
    item_similarity_df = pd.DataFrame(item_similarity, 
                                      index=item_matrix.index,
                                      columns=item_matrix.index)
    
    return item_similarity_df


def recommend_items(
    target_item_id: str,
    item_similarity_df: pd.DataFrame,
    top_n: int = 3
):
    """
    특정 아이템과 비슷한 아이템을 상위 n개 추천한다.
    """
    if target_item_id not in item_similarity_df.index:
        return []

    # 해당 아이템과 다른 아이템 간 유사도 점수 가져오기
    sim_scores = item_similarity_df[target_item_id]

    # 자기 자신 제외, 상위 n개 내림차순
    sim_scores = sim_scores.drop(target_item_id)  # 자기 자신 제외
    sim_scores = sim_scores.sort_values(ascending=False).head(top_n)

    return list(sim_scores.index)


def recommend_for_user(
    target_user_id: str,
    ratings_df: pd.DataFrame,
    item_similarity_df: pd.DataFrame,
    top_n: int = 3
):
    """
    특정 사용자가 아직 평가하지 않은 아이템 중에서,
    사용자 취향에 가장 적합할 것으로 보이는 아이템 추천
    (가장 선호하는 아이템과 비슷한 것 중심)
    """
    # 1) 해당 사용자의 평점 정보만 가져옴
    user_ratings = ratings_df[ratings_df["user_id"] == target_user_id]
    if user_ratings.empty:
        return []

    # 2) 사용자 평점이 높은 item들만 추출 (예: 평점 4점 이상)
    user_favorites = user_ratings[user_ratings["rating"] >= 4.0]["item_id"].unique()

    # 3) (간단화) favorite 아이템들과 유사도 높은 아이템들을 합쳐서 점수화
    candidate_scores = {}
    for fav_item in user_favorites:
        if fav_item not in item_similarity_df.index:
            continue
        sim_items = item_similarity_df[fav_item].sort_values(ascending=False)

        # 자기 자신 제외
        sim_items = sim_items.drop(fav_item, errors="ignore")

        # 점수를 candidate에 누적
        for item_id, sim_score in sim_items.items():
            candidate_scores[item_id] = candidate_scores.get(item_id, 0) + sim_score

    # 4) 이미 사용자가 평가한 아이템은 제외
    already_rated_items = user_ratings["item_id"].unique()
    for item_id in already_rated_items:
        if item_id in candidate_scores:
            del candidate_scores[item_id]

    # 5) 스코어가 높은 순으로 정렬 후 상위 N개
    sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
    recommended_item_ids = [item_id for item_id, score in sorted_candidates[:top_n]]

    return recommended_item_ids


def get_item_metadata(item_id, travel_meta):
    """
    travel_meta(json) 에서 해당 item_id의 메타 정보를 찾아 반환
    """
    # travel_meta는 카테고리(자연/힐링, 전시 등) -> 목록 배열 구조이므로
    # 모든 카테고리를 순회하며 해당 item_id를 찾아야 함
    for category, items in travel_meta.items():
        for item_info in items:
            if item_info["id"] == item_id:
                return item_info
    return None


def get_items_info(item_ids, travel_meta):
    """
    여러 item_id 리스트에 대해 메타데이터 조회
    """
    results = []
    for iid in item_ids:
        info = get_item_metadata(iid, travel_meta)
        if info is not None:
            results.append(info)
    return results


# 직접 실행 예시
if __name__ == "__main__":
    # (1) 데이터 로드
    ratings_df, travel_meta = load_data("rating_matrix.csv", "travel_metadata.json")
    # (2) 아이템 유사도 행렬 계산
    item_similarity_df = create_item_similarity_matrix(ratings_df)
    # (3) 특정 아이템(CJU001)과 유사한 아이템 추천
    sim_items = recommend_items("CJU001", item_similarity_df, top_n=3)
    print("CJU001 과 유사한 아이템:", sim_items)
    # (4) 특정 사용자(U010)에게 맞춤형 아이템 추천
    recommended = recommend_for_user("U010", ratings_df, item_similarity_df, top_n=5)
    recommended_info = get_items_info(recommended, travel_meta)
    print(f"U010 사용자를 위한 추천: {recommended_info}")
