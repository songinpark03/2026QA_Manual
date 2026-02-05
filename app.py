import streamlit as st
@st.dialog("📘 특허 질의응답 시스템(2-Stage Framework) 매뉴얼")
def show_manual():
    st.markdown("""
    ## 시스템 개요
    본 시스템은 **강남대학교 인공지능융합공학부** 연구팀이 제안한 **'요약-검색 결합 및 다중 문서 참조'** 기술을 기반으로 합니다.  
    기존 키워드 중심 검색의 한계를 극복하고, LLM과 RAG를 결합하여 전문적인 특허 분석 답변을 제공합니다. [cite: 4, 18]

    ---

    ### 핵심 기술 (2-Stage Framework)
    우리 시스템은 정확도를 높이기 위해 두 단계로 작동합니다. [cite: 27, 59]
    1. **Stage 1 (배경기술 요약 검색)**: 특허의 핵심 기술 정보가 담긴 '배경기술' 요약본을 바탕으로 관련 문서 3건을 빠르게 탐색합니다. [cite: 60, 61]
    2. **Stage 2 (전문 기반 답변 생성)**: 검색된 문서의 전문(기술분야, 배경기술, 실시내용)을 **400~500자 단위의 정밀한 청크(Chunk)**로 분석하여 정확한 근거를 추출합니다. [cite: 58, 62]

    ---

    ### 시스템의 강점
    * **10% 이상의 성능 향상**: Naive RAG 방식 대비 **LLM-Judge Score 기준 약 10%의 성능 향상**을 입증하였습니다. [cite: 6, 99]
    * **정보 부족 및 왜곡 방지**: 단일 문서가 아닌 **다중 문서(Top-3)**를 교차 참조하여 답변의 완전성을 높였습니다. [cite: 5, 64]
    * **환각 현상 억제**: GPT-4o-mini 모델을 기반으로 실제 특허 명세서 문헌에 근거한 답변만 생성하도록 설계되었습니다. [cite: 6, 89]

    ---

    ### 효과적인 질문 예시
    특허의 구조적 특성을 활용해 질문하면 더 좋은 결과를 얻을 수 있습니다. [cite: 45]
    * **기술 비교**: "이 특허와 유사한 선행 기술들의 해결 과제 차이점을 분석해줘." [cite: 104]
    * **구조 추출**: "명세서 내 '발명의 실시 내용'을 바탕으로 구체적인 구현 방법을 설명해줘." [cite: 62]
    * **핵심 요약**: "배경기술에 언급된 기존 기술의 문제점과 본 발명의 차별점은 뭐야?" [cite: 51]

    ---

    ### ⚠️ 안내 및 주의사항
    * **데이터 출처**: 본 시스템은 반도체 분야 등 수집된 특허 데이터셋을 기반으로 합니다. [cite: 87, 88]
    * **연구 목적**: 본 앱은 특허 질의응답 품질 향상을 위한 연구용 프로토타입이며, 답변은 법적 효력을 갖지 않습니다. [cite: 108]
    """)
    
    if st.button("닫기", use_container_width=True):
        st.rerun()
import json
from patent_qa import PatentQAChatbot
from datetime import datetime
import os
import requests

# -------------------------------
# JSON 다운로드 설정
# -------------------------------
JSON_URL = "https://drive.google.com/uc?id=1rlB_4MrzZLFXrwHgPbOQge7bDdinwyKl"
JSON_PATH = "final_patent_chunking_results.json"

def download_json():
    if not os.path.exists(JSON_PATH):
        st.info("📥 특허 데이터 로딩 중입니다. 잠시만 기다려주세요...")
        r = requests.get(JSON_URL)
        r.raise_for_status()
        with open(JSON_PATH, "wb") as f:
            f.write(r.content)

download_json()

# -------------------------------
# 페이지 설정
# -------------------------------
st.set_page_config(
    page_title="특허 질의응답 시스템",
    page_icon="💬",
    layout="wide"
)

# -------------------------------
# iMessage 스타일 CSS
# -------------------------------
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #f0f4f8 0%, #e8f0f8 50%, #f0f4f8 100%);
    }
    
    .block-container {
        padding-top: 2rem !important;
        max-width: 900px !important;
    }
    
    h1 {
        color: #1a1a1a !important;
        font-size: 1.8rem !important;
    }
    
    .stChatMessage {
        background: transparent !important;
        padding: 0.5rem 0 !important;
    }
    
    .user-message-wrapper {
        display: flex;
        justify-content: flex-end;
        margin: 0.5rem 0;
    }
    
    .user-message {
        background: linear-gradient(135deg, #007AFF 0%, #0051D5 100%);
        color: white;
        padding: 0.75rem 1rem;
        border-radius: 18px;
        border-bottom-right-radius: 4px;
        max-width: 70%;
        box-shadow: 0 1px 2px rgba(0, 122, 255, 0.2);
        font-size: 0.95rem;
        line-height: 1.4;
        word-wrap: break-word;
    }
    
    .bot-avatar {
        width: 32px;
        height: 32px;
        background: linear-gradient(135deg, #a0aec0 0%, #718096 100%);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 2px 8px rgba(113, 128, 150, 0.25);
        margin-top: 0.5rem;
    }
    
    .bot-avatar svg {
        width: 18px;
        height: 18px;
        color: white;
    }
    
    .assistant-message {
        background: #ffffff;
        color: #1a1a1a;
        padding: 0.75rem 1rem;
        border-radius: 18px;
        border-bottom-left-radius: 4px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.12);
        border: 1px solid rgba(0, 0, 0, 0.06);
        font-size: 0.95rem;
        line-height: 1.5;
        word-wrap: break-word;
    }
    
    .patent-meta-inline {
        font-size: 0.75rem;
        color: #007AFF;
        background: rgba(0, 122, 255, 0.08);
        padding: 0.35rem 0.6rem;
        border-radius: 8px;
        margin-top: 0.75rem;
        display: inline-block;
        border: 1px solid rgba(0, 122, 255, 0.2);
    }
    
    .stChatInputContainer {
        background: rgba(255, 255, 255, 0.95);
        border-top: 1px solid rgba(0, 0, 0, 0.08);
    }
    
    button[title="Copy to clipboard"] {
        display: none !important;
    }
    
    .stCodeBlock {
        display: none !important;
    }
    
    #MainMenu, footer {
        visibility: hidden !important;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# 챗봇 로딩
# -------------------------------
@st.cache_resource
def load_chatbot():
    return PatentQAChatbot(JSON_PATH)

chatbot = load_chatbot()

# -------------------------------
# 제목
# -------------------------------
st.title("💬 특허 질의응답 시스템")
st.caption("청킹 전략 기반 · 다중 특허 문서 참조 QA")
st.markdown("---")

# -------------------------------
# 세션 상태 초기화
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if len(st.session_state.messages) == 0:
    st.session_state.messages.append({
        "role": "assistant",
        "content": "안녕하세요! 특허 QA 시스템입니다. 특허에 관한 질문을 자유롭게 입력해주세요."
    })

# -------------------------------
# 대화 출력
# -------------------------------
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f'<div class="user-message-wrapper"><div class="user-message">{msg["content"]}</div></div>',
            unsafe_allow_html=True
        )
    else:
        col1, col2 = st.columns([0.05, 0.95])
        
        with col1:
            st.markdown("""
            <div class="bot-avatar">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <rect x="3" y="11" width="18" height="10" rx="2"/>
                    <circle cx="12" cy="5" r="2"/>
                    <path d="M12 7v4"/>
                </svg>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # 답변 내용과 참조 출원번호를 말풍선 안에 함께 표시
            patent_html = ""
            if "patents" in msg and msg["patents"]:
                patents_str = ", ".join(msg["patents"])
                patent_html = f'<div class="patent-meta-inline">📋 {patents_str}</div>'
            
            st.markdown(
                f'<div class="assistant-message">{msg["content"]}{patent_html}</div>',
                unsafe_allow_html=True
            )

# -------------------------------
# 질문 입력
# -------------------------------
user_input = st.chat_input("메시지를 입력하세요...")

if user_input:
    # 1. 사용자 질문 먼저 추가
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })
    
    # 2. 화면 즉시 갱신 (질문 표시)
    st.rerun()

# 답변 생성 체크 (마지막 메시지가 user이고 답변이 없는 경우)
if (len(st.session_state.messages) > 0 and 
    st.session_state.messages[-1]["role"] == "user"):
    
    last_question = st.session_state.messages[-1]["content"]
    
    # 로딩 표시
    with st.spinner("💭 답변 생성 중..."):
        result = chatbot.ask(last_question, verbose=False, max_patents=3)
    
    # 답변 추가
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "patents": result["application_numbers"]
    })
    
    # 답변 후 화면 갱신
    st.rerun()

# -------------------------------
# 요약 정보
# -------------------------------
if len(st.session_state.messages) > 1:
    st.markdown("<hr>", unsafe_allow_html=True)
    user_messages = [m for m in st.session_state.messages if m['role'] == 'user']
    st.markdown(f"""
    <div style="background: white; border-radius: 12px; padding: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.08);">
        <h4 style="margin: 0 0 0.5rem 0; color: #1a1a1a;">📊 대화 요약</h4>
        <p style="margin: 0; color: #86868b; font-size: 0.9rem;">
            총 질문 수: <strong>{len(user_messages)}개</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)

# -------------------------------
# 사이드바
# -------------------------------
with st.sidebar:
    st.markdown("### ⚙️ 설정")
    
    if st.button("📖 이용 매뉴얼 보기", use_container_width=True):
        show_manual()

    if st.button("🗑️ 대화 내역 초기화", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

