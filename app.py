import streamlit as st
import os
import dotenv
import uuid
import hashlib
import logging
from typing import Optional, Tuple, Dict
from datetime import datetime
import threading
import psutil
from typing import Optional, Tuple, Dict, Any
from TextCustomizationModule import customize_text
from SeoNContentAnalysisModule import SEOContentAnalyzer
from Utilities import (
    validate_api_keys,
    format_message_history,
    calculate_token_usage,
    generate_session_summary,
    sanitize_input,
    create_error_message,
    log_error,
    cleanup_session,
    get_system_status,
    optimize_response
)

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, AIMessage

from rag_methods import (
    load_doc_to_db, 
    load_url_to_db,
    stream_llm_response,
    stream_llm_rag_response,
)

# 환경 변수 로드
dotenv.load_dotenv()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 상수 정의
MODELS = [
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "anthropic/claude-3-5-sonnet-20240620",
    "google/gemini-1.5-flash",    
]

def init_session_state():
    """세션 상태 초기화"""
    try:
        initial_states = {
            "session_id": str(uuid.uuid4()),
            "password_correct": False,
            "messages": [
                {"role": "user", "content": "안녕하세요!"},
                {"role": "assistant", "content": "어떤 주제로 글을 작성할까요?"}
            ],
            "rag_sources": [],
            "current_response": "",
            "use_customization": False,
            "use_seo": False,
            "current_text": "",
            "error_log": [],
            "system_status": get_system_status()
        }
        
        for key, value in initial_states.items():
            if key not in st.session_state:
                st.session_state[key] = value
                
    except Exception as e:
        log_error(e, "init_session_state")
        st.error("세션 초기화 중 오류가 발생했습니다.")

def verify_api_keys() -> Tuple[str, str, str, Dict[str, bool]]:
    """API 키 검증 및 반환"""
    api_keys = {
        "openai": st.session_state.get('openai_api_key', os.getenv("OPENAI_API_KEY", "")),
        "anthropic": st.session_state.get('anthropic_api_key', os.getenv("ANTHROPIC_API_KEY", "")),
        "google": st.session_state.get('google_api_key', os.getenv("GOOGLE_API_KEY", ""))
    }
    
    validation_result = validate_api_keys(
        api_keys["openai"],
        api_keys["anthropic"],
        api_keys["google"]
    )
    
    return api_keys["openai"], api_keys["anthropic"], api_keys["google"], validation_result

def setup_page():
    """페이지 설정"""
    st.set_page_config(
        page_title="블로그 자동생성 앱", 
        page_icon="📚", 
        layout="centered", 
        initial_sidebar_state="expanded"
    )
    st.html("""<h2 style="text-align: center;">📚🔍 <i> 블로그 자동생성 봇 </i> 🤖💬</h2>""")

def setup_sidebar() -> Tuple[str, bool]:
    """사이드바 설정"""
    with st.sidebar:
        # API 키 입력 및 검증
        openai_api_key, anthropic_api_key, google_api_key, api_validation = verify_api_keys()
        
        with st.expander("🔑 API 키 설정"):
            st.text_input("OpenAI API Key", value=openai_api_key, type="password", key="openai_api_key")
            st.text_input("Anthropic API Key", value=anthropic_api_key, type="password", key="anthropic_api_key")
            st.text_input("Google API Key", value=google_api_key, type="password", key="google_api_key")
        
        st.divider()
        
        # 사용 가능한 모델 필터링
        available_models = [
            model for model in MODELS 
            if ("openai" in model and api_validation["openai"]) or 
               ("anthropic" in model and api_validation["anthropic"]) or 
               ("google" in model and api_validation["google"])
        ]
        
        if available_models:
            selected_model = st.selectbox("🤖 모델 선택", available_models, key="model")
        else:
            st.warning("⬅️ 계속하시려면 API Key를 입력해주세요...")
            return None, False
        
        # RAG 설정
        cols = st.columns(2)
        with cols[0]:
            use_rag = st.toggle(
                "RAG 사용",
                value="vector_db" in st.session_state,
                key="use_rag",
                disabled="vector_db" not in st.session_state,
            )
        with cols[1]:
            if st.button("전체 삭제", type="primary"):
                cleanup_session()
                st.rerun()
        
        # RAG 소스 관리
        st.header("RAG 소스:")
        setup_rag_sources()
        
        # 부가 기능 설정
        st.divider()
        setup_additional_features()
        
        # 시스템 상태 표시
        if st.checkbox("시스템 상태 보기"):
            show_system_status()
        
        # 도움말
        add_sidebar_help()
        
        return selected_model, use_rag

def setup_rag_sources():
    """RAG 소스 설정"""
    st.file_uploader(
        "📄 문서 업로드",
        type=["pdf", "txt", "docx", "md"],
        accept_multiple_files=True,
        on_change=load_doc_to_db,
        key="rag_docs",
    )
    
    st.text_input(
        "🌐 URL 넣기",
        placeholder="https://example.com",
        on_change=load_url_to_db,
        key="rag_url",
    )
    
    with st.expander(f"📚 문서 속 DB ({len(st.session_state.get('rag_sources', []))})"):
        st.write(st.session_state.get('rag_sources', []))

def setup_additional_features():
    """부가 기능 설정"""
    cols = st.columns(2)
    with cols[0]:
        st.toggle("커스텀 기능", value=False, key="use_customization")
    with cols[1]:
        st.toggle("SEO 분석", value=False, key="use_seo")

def show_system_status():
    """시스템 상태 표시"""
    status = st.session_state.system_status
    cols = st.columns(2)
    with cols[0]:
        st.metric("메모리 사용량", f"{status['memory_usage']:.1f}MB")
        st.metric("CPU 사용률", f"{status['cpu_usage']}%")
    with cols[1]:
        st.metric("디스크 사용률", f"{status['disk_usage']}%")
        st.metric("활성 스레드", status['active_threads'])

def get_llm_model(model_name: str, api_keys: Dict[str, str]) -> Optional[object]:
    """LLM 모델 초기화"""
    try:
        model_type = model_name.split("/")[0]
        model_id = model_name.split("/")[-1]
        
        model_configs = {
            "openai": {
                "class": ChatOpenAI,
                "params": {
                    "api_key": api_keys["openai"],
                    "model_name": model_id
                }
            },
            "anthropic": {
                "class": ChatAnthropic,
                "params": {
                    "api_key": api_keys["anthropic"],
                    "model": model_id
                }
            },
            "google": {
                "class": ChatGoogleGenerativeAI,
                "params": {
                    "api_key": api_keys["google"],
                    "model": model_id
                }
            }
        }
        
        config = model_configs.get(model_type)
        if not config:
            raise ValueError(f"지원하지 않는 모델 타입: {model_type}")
            
        return config["class"](
            **config["params"],
            temperature=0.3,
            streaming=True
        )
        
    except Exception as e:
        log_error(e, "get_llm_model")
        st.error(f"모델 초기화 중 오류가 발생했습니다: {create_error_message(e)}")
        return None

def process_message(prompt: str, llm_model, use_rag: bool):
    """메시지 처리 및 응답 생성"""
    try:
        # 입력 정제
        prompt = sanitize_input(prompt)
        
        # 토큰 사용량 추정
        estimated_tokens = calculate_token_usage(prompt)
        if estimated_tokens > 4000:  # 예시 제한값
            st.warning("입력이 너무 깁니다. 더 짧은 메시지를 입력해주세요.")
            return
            
        # 사용자 메시지 추가
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # assistant 응답 생성
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            st.session_state.message_placeholder = message_placeholder

            messages = [
                HumanMessage(content=m["content"]) if m["role"] == "user" 
                else AIMessage(content=m["content"]) 
                for m in st.session_state.messages
            ]

            # 응답 생성
            response_text = ""
            with st.spinner("응답 생성 중..."):
                # 스트리밍 응답 처리
                if use_rag:
                    for chunk in stream_llm_rag_response(llm_model, messages):
                        response_text += chunk
                        message_placeholder.markdown(response_text)
                else:
                    for chunk in stream_llm_response(llm_model, messages):
                        response_text += chunk
                        message_placeholder.markdown(response_text)

                if response_text:
                    # 응답 최적화 및 저장
                    optimized_response = optimize_response(response_text)
                    st.session_state.original_response = optimized_response
                    st.session_state.current_text = optimized_response
                    
                    # 메시지 플레이스홀더 업데이트
                    message_placeholder.markdown(optimized_response)
                    
                    # 커스터마이징 적용
                    if st.session_state.use_customization:
                        customized_text = customize_text(optimized_response, llm_model)
                        if customized_text and customized_text != optimized_response:
                            st.session_state.current_text = customized_text
                            # 메시지 업데이트
                            st.session_state.messages[-1]["content"] = customized_text
                            message_placeholder.markdown(customized_text)

                    # SEO 분석
                    if st.session_state.use_seo and not use_rag:
                        with st.spinner("SEO 분석 중..."):
                            seo_result = run_seo_analysis(st.session_state.current_text, llm_model)
                            if seo_result and "개선된_텍스트" in seo_result:
                                st.session_state.current_text = seo_result["개선된_텍스트"]
                                st.session_state.messages[-1]["content"] = seo_result["개선된_텍스트"]
                                message_placeholder.markdown(seo_result["개선된_텍스트"])

                    # 세션 요약 업데이트
                    st.session_state.session_summary = generate_session_summary()

    except Exception as e:
        log_error(e, "process_message")
        st.error(f"메시지 처리 중 오류가 발생했습니다: {create_error_message(e)}")
        
def run_seo_analysis(text: str, llm_model) -> Dict[str, Any]:  # llm_model 매개변수 추가
    """SEO 분석 실행"""
    try:
        st.divider()
        st.subheader("🔍 SEO 분석")

        col1, col2 = st.columns([3, 1])
        with col1:
            keyword = st.text_input(
                "주요 키워드",
                help="분석할 주요 키워드를 입력하세요",
                placeholder="예: 맛집, 여행, 리뷰 등",
                key="seo_keyword"
            )

        with col2:
            analyze_button = st.button(
                "분석 시작",
                type="primary",
                key="seo_analysis_button"
            )

        if analyze_button and keyword:
            with st.spinner("SEO 분석 중..."):
                seo_analyzer = SEOContentAnalyzer()
                keyword = sanitize_input(keyword)
                
                # 키워드 트렌드 분석
                trends = seo_analyzer.analyze_keyword_trends(keyword)
                
                # 메트릭 표시
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("월간 검색량", f"{trends['월간 검색량']:,}회")
                with col2:
                    st.metric("전월 대비 증감", f"{trends['전월 대비 증감']:.1f}%")
                
                # SEO 점수 분석
                seo_score = seo_analyzer.calculate_seo_score(text, keyword)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("SEO 총점", f"{seo_score['총점']}/100")
                with col2:
                    st.metric("SEO 등급", seo_score['등급'])
                with col3:
                    st.metric(
                        "키워드 밀도",
                        f"{seo_score['세부 점수']['키워드 밀도 점수']:.1f}%"
                    )
                
                # 태그 추천
                tags = seo_analyzer._generate_tags(text, keyword)
                st.write("### 추천 태그")
                st.write(", ".join(f"#{tag}" for tag in tags))
                
                # 개선 제안
                recommendations = seo_analyzer._analyze_content_structure(text)
                st.write("### 개선 제안")
                for suggestion in recommendations["개선 필요 사항"]:
                    if suggestion:
                        st.info(suggestion)

                # SEO 최적화된 텍스트 생성 요청
                prompt = f"""
                다음 텍스트를 SEO 최적화해주세요:
                1. 키워드 '{keyword}'의 자연스러운 배치
                2. 제목과 부제목 구조 최적화
                3. 적절한 키워드 밀도 유지
                4. 가독성 개선
                
                원본 텍스트:
                {text}
                """
                
                with st.spinner("SEO 최적화된 텍스트 생성 중..."):
                    optimized_response = llm_model.invoke(prompt).content
                    
                    return {
                        "분석_결과": seo_score,
                        "추천_태그": tags,
                        "개선_제안": recommendations["개선 필요 사항"],
                        "개선된_텍스트": optimized_response
                    }
                
        elif analyze_button:
            st.warning("키워드를 입력해주세요.")
            
    except Exception as e:
        log_error(e, "run_seo_analysis")
        st.error(f"SEO 분석 중 오류가 발생했습니다: {create_error_message(e)}")
        return None
    
def check_password(input_password: str) -> bool:
    """비밀번호 확인"""
    try:
        correct_hash = '0ffe1abd1a08215353c233d6e009613e95eec4253832a761af28ff37ac5a150c'  # 비밀번호: '1111'
        return hashlib.sha256(input_password.encode()).hexdigest() == correct_hash
    except Exception as e:
        log_error(e, "check_password")
        return False

def handle_password_protection():
    """비밀번호 보호 처리"""
    if not st.session_state.password_correct:
        st.markdown("### 🔒 접근 제어")
        password_input = st.text_input("비밀번호를 입력하세요:", type="password")
        
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("확인", type="primary"):
                if check_password(password_input):
                    st.session_state.password_correct = True
                    st.success("인증되었습니다.")
                    st.rerun()
                else:
                    st.error("잘못된 비밀번호입니다. 다시 시도해주세요.")
        return False
    return True

def add_sidebar_help():
    """사이드바 도움말 추가"""
    with st.sidebar:
        st.divider()
        st.markdown("### 🔍 도움말")
        
        with st.expander("주요 기능"):
            st.markdown("""
            - 💡 RAG를 사용하면 업로드한 문서나 URL의 내용을 참고하여 응답합니다
            - 🔄 대화 초기화 버튼으로 새로운 대화를 시작할 수 있습니다
            - ✨ 응답 텍스트는 이모지, 문체, 길이, 독해 수준을 조정할 수 있습니다
            """)
        
        with st.expander("사용 팁"):
            st.markdown("""
            1. **RAG 사용하기**:
               - PDF, TXT, DOCX, MD 파일 업로드 가능
               - URL을 통한 웹페이지 참조 가능
               - 최대 10개의 문서 동시 참조 가능
            
            2. **커스터마이징**:
               - 이모지 추가로 글 생동감 향상
               - 문체 변경으로 톤앤매너 조정
               - 길이 조절로 콘텐츠 량 최적화
               - 독해 수준 조정으로 타겟 독자층 맞춤화
            
            3. **SEO 최적화**:
               - 키워드 분석으로 검색 노출 최적화
               - 실시간 트렌드 분석
               - 맞춤형 태그 추천
               - 컨텐츠 구조 개선 제안
            """)
        
        with st.expander("문제 해결"):
            st.markdown("""
            - ❓ API 키 관련 문제는 각 제공사의 웹사이트에서 확인
            - 🔄 응답이 느린 경우 새로고침 후 다시 시도
            - 📝 오류 발생시 로그를 확인하여 문제 파악
            - 🛠 기술적 문제는 관리자에게 문의
            """)

def update_system_metrics():
    """시스템 메트릭 업데이트"""
    try:
        st.session_state.system_status = get_system_status()
    except Exception as e:
        log_error(e, "update_system_metrics")

def main():
    """메인 함수"""
    try:
        # 초기 설정
        init_session_state()
        setup_page()
        
        # 비밀번호 보호 체크
        if not handle_password_protection():
            return
        
        # 사이드바 설정
        selected_model, use_rag = setup_sidebar()
        if not selected_model:
            return
        
        # API 키 확인 및 모델 초기화
        openai_key, anthropic_key, google_key, _ = verify_api_keys()
        api_keys = {
            "openai": openai_key,
            "anthropic": anthropic_key,
            "google": google_key
        }
        
        llm_model = get_llm_model(selected_model, api_keys)
        if not llm_model:
            return
        
        # 대화 기록 표시
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # 사용자 입력 처리
        if prompt := st.chat_input(
            "메시지를 입력하세요",
            key="chat_input",
            max_chars=2000
        ):
            process_message(prompt, llm_model, use_rag)
            
        # 시스템 메트릭 업데이트
        update_system_metrics()
        
    except Exception as e:
        log_error(e, "main")
        st.error(f"애플리케이션 실행 중 오류가 발생했습니다: {create_error_message(e)}")
        if st.button("앱 초기화", type="primary"):
            cleanup_session()
            st.rerun()

if __name__ == "__main__":
    main()