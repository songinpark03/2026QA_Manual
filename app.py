import streamlit as st
# --- 매뉴얼 팝업창 정의 ---
@st.dialog("📘 특허 질의응답 시스템 이용 매뉴얼")
def show_manual():
    st.markdown("""
    ### 1. 주요 기능
    * **다중 특허 참조**: 여러 건의 특허를 동시에 분석합니다.
    * **정밀 검색**: 청킹(Chunking) 전략을 통해 문맥에 맞는 정보를 추출합니다.
    
    ### 2. 사용 팁
    * **초기화**: 새로운 질문을 시작할 때 왼쪽의 '대화 내역 초기화'를 눌러주세요.
    * **구체적 질문**: "A 특허의 핵심 기술을 요약해줘"처럼 구체적으로 물어보세요.
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
