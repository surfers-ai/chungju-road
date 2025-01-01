import os
import re

import openai
import streamlit as st
from dotenv import load_dotenv

from recommender import (
    load_data,
    create_item_similarity_matrix,
    recommend_for_user,
    get_items_info,
)

load_dotenv()

# 화면 넓게 사용
st.set_page_config(
    layout="wide", page_title="충주로드 - 당신만을 위한 AI 여행 친구", page_icon="🌸"
)

# OpenAI 클라이언트 설정
openai_client = openai

# OpenAI API 키 & 어시스턴트 ID 설정
ASSISTANT_ID = os.getenv("ASSISTANT_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# --- 추천 시스템 준비 ---
ratings_df, travel_meta = load_data("rating_matrix.csv", "travel_metadata.json")
item_sim_df = create_item_similarity_matrix(ratings_df)

# (데모용) 세션에 user_id를 하나 고정한다고 가정
demo_user_id = "U001"

# ──────────────────────────────────────────────────────────────────
# 0) 페이지 헤더
# ──────────────────────────────────────────────────────────────────
st.title("충주로드 - 당신만을 위한 AI 여행 친구")
st.write("**AI 챗봇과 함께 충주 여행을 계획**해보세요!")

# 메인 레이아웃: 1:2:1 비율
col_left, col_center, col_right = st.columns([1, 2, 1])


def initialize_session_state():
    """세션 상태를 초기화하는 함수"""
    if "thread_id" not in st.session_state:
        thread = openai_client.beta.threads.create()
        st.session_state.thread_id = thread.id
    if "messages" not in st.session_state:
        st.session_state.messages = []


initialize_session_state()

# ---------------------------------------------
# 1) 왼쪽 컬럼 - 추천 목록 (협업필터링 활용)
# ---------------------------------------------
with col_left:
    st.subheader("⛰️ 맞춤 추천 여행지")
    st.caption(f"현재 데모 사용자: {demo_user_id}")

    # 협업 필터링으로 추천 아이템 가져오기
    recommended_item_ids = recommend_for_user(
        target_user_id=demo_user_id,
        ratings_df=ratings_df,
        item_similarity_df=item_sim_df,
        top_n=3
    )
    recommended_items_info = get_items_info(recommended_item_ids, travel_meta)

    if recommended_items_info:
        for rec in recommended_items_info:
            with st.container():
                # 임의의 이미지
                st.image("chungju-image.png", width=120)
                st.markdown(f"**{rec['title']}**")
                st.write(rec.get("description", "정보 없음"))
                st.markdown("---")
    else:
        st.write("개인화 추천 결과가 없습니다.")

# ---------------------------------------------
# 2) 가운데 컬럼 - 채팅창
# ---------------------------------------------
with col_center:
    st.subheader("💬 충주 AI 여행친구와 대화하기")
    st.info("충주 여행 관련 궁금한 점을 자유롭게 물어보세요.")
    st.caption("예: '물멍하기 좋은 곳 알려줘'")

    # 가장 먼저 메시지 표시용 placeholder 만들기
    chat_placeholder = st.container()

    # 사용자 입력 처리
    if prompt := st.chat_input("충주 여행 관련 궁금한 것을 물어보세요..."):
        # 채팅 기록 표시
        with chat_placeholder:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        # 사용자 메시지 추가
        st.session_state.messages.append({"role": "user", "content": prompt})
        with chat_placeholder:
            with st.chat_message("user"):
                st.markdown(prompt)

        # OpenAI API로 메시지 전송
        openai_client.beta.threads.messages.create(
            thread_id=st.session_state.thread_id, role="user", content=prompt
        )

        # 어시스턴트 응답 처리
        with chat_placeholder:
            with st.chat_message("assistant"):
                stream = openai_client.beta.threads.runs.create(
                    thread_id=st.session_state.thread_id,
                    assistant_id=ASSISTANT_ID,
                    stream=True,
                )

                import re
                placeholder = st.empty()
                full_response = ""
                for chunk in stream:
                    if chunk.event == "thread.message.delta":
                        if hasattr(chunk.data, "delta") and hasattr(
                            chunk.data.delta, "content"
                        ):
                            content_delta = chunk.data.delta.content[0].text.value
                            full_response += content_delta
                            # 정규식 패턴으로 불필요한 텍스트 제거
                            full_response = re.sub(
                                r"【\d+:\d+†.*?】", "", full_response
                            )
                            placeholder.markdown(full_response + "▌")
                placeholder.markdown(full_response)

                st.session_state.messages.append(
                    {"role": "assistant", "content": full_response}
                )

# ---------------------------------------------
# 3) 오른쪽 컬럼 - 광고(Mock)
# ---------------------------------------------
with col_right:
    st.subheader("📢 특별한 혜택")
    st.image("chungju-image.png", width=150)
    st.write("충주의 특산물, 숙박 할인, 맛집 쿠폰 등 다양한 혜택 정보를 받아보세요!")

    # 광고/이벤트 목록 (데이터 예시)
    ads_data = [
        {
            "category": "지역 이벤트",
            "title": "제8회 지현동 사과나무이야기길 축제",
            "desc": "사과나무길 걷기, 체험 부스, 푸드트럭 등 다양한 행사 진행",
            "link_text": "자세히 보기",
            "link_url": "https://www.dominilbo.com/news/articleView.html?idxno=201670",
        },
        {
            "category": "숙소 할인",
            "title": "야놀자 충주 숙박 5만원 할인 프로모션",
            "desc": "가을 시즌에 한해, 충주 지역 숙소 예약 시 즉시 할인",
            "link_text": "프로모션 상세",
            "link_url": "https://www.thefairnews.co.kr/news/articleView.html?idxno=34635",
        },
        {
            "category": "맛집 프로모션",
            "title": "중앙탑메밀마당 - 메밀 막국수 주문시 치킨 세트 할인",
            "desc": "방문 시 블로그 인증하면 할인 혜택 제공",
            "link_text": "할인정보 확인",
            "link_url": "http://jdblue2022.tistory.com/entry/충주-맛집-베스트10",
        },
    ]

    # CSS를 통한 Card 스타일
    st.markdown(
        """
        <style>
        .ad-card {
            border: 1px solid #CCC;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 15px;
            background-color: #F9F9F9;
        }
        .ad-card h4 {
            margin: 0px 0px 10px 0px;
            font-size: 1.1rem;
            color: #333;
        }
        .ad-card p {
            margin: 0px 0px 10px 0px;
            line-height: 1.4;
            color: #555;
        }
        .ad-link {
            text-decoration: none;
            color: #3273dc;
            font-weight: 500;
        }
        .ad-link:hover {
            text-decoration: underline;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # 각 광고를 Card 형식으로 렌더링
    for ad in ads_data:
        st.markdown(
            f"""
            <div class="ad-card">
                <h4>[{ad['category']}]</h4>
                <p><strong>{ad['title']}</strong></p>
                <p>{ad['desc']}</p>
                <p><a href="{ad['link_url']}" target="_blank" class="ad-link">{ad['link_text']}</a></p>
            </div>
            """,
            unsafe_allow_html=True,
        )
