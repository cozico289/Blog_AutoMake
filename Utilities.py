import streamlit as st
import os
import re
import logging
import psutil
import threading
from datetime import datetime
from typing import Dict, Any, List, Optional

def validate_api_keys(openai_key: str, anthropic_key: str, google_key: str) -> dict:
    """API 키 유효성 검증"""
    return {
        "openai": bool(openai_key and "sk-" in openai_key),
        "anthropic": bool(anthropic_key),
        "google": bool(google_key)
    }

def format_message_history(messages: list) -> str:
    """메시지 히스토리 포맷팅"""
    formatted = []
    for msg in messages:
        role = "🧑" if msg["role"] == "user" else "🤖"
        formatted.append(f"{role}: {msg['content']}")
    return "\n\n".join(formatted)

def calculate_token_usage(text: str) -> int:
    """토큰 사용량 추정 (근사치)"""
    return len(text.split()) * 1.3  # 평균적으로 단어당 1.3개의 토큰

def generate_session_summary() -> dict:
    """세션 요약 정보 생성"""
    return {
        "total_messages": len(st.session_state.messages),
        "rag_sources": len(st.session_state.get('rag_sources', [])),
        "customization_used": st.session_state.use_customization,
        "seo_analysis_used": st.session_state.use_seo,
        "current_text_length": len(st.session_state.current_text)
    }

def sanitize_input(text: str) -> str:
    """사용자 입력 정제"""
    import re
    # XSS 방지를 위한 특수문자 이스케이프
    text = re.sub(r'[<>]', '', text)
    # 연속된 공백 제거
    text = ' '.join(text.split())
    return text.strip()

def create_error_message(error: Exception) -> str:
    """사용자 친화적 에러 메시지 생성"""
    error_types = {
        ValueError: "입력값이 올바르지 않습니다",
        KeyError: "필요한 데이터가 누락되었습니다",
        ConnectionError: "네트워크 연결에 문제가 있습니다",
        TimeoutError: "응답 시간이 초과되었습니다"
    }
    return error_types.get(type(error), str(error))

def log_error(error: Exception, context: str):
    """에러 로깅"""
    import logging
    logging.error(f"Error in {context}: {str(error)}")
    st.session_state['error_log'] = st.session_state.get('error_log', [])
    st.session_state['error_log'].append({
        'timestamp': datetime.now().isoformat(),
        'context': context,
        'error': str(error)
    })

def cleanup_session():
    """세션 정리"""
    # 임시 파일 삭제
    import shutil
    if os.path.exists("source_files"):
        shutil.rmtree("source_files")
    
    # 세션 상태 정리
    keys_to_keep = {'password_correct', 'session_id'}
    for key in list(st.session_state.keys()):
        if key not in keys_to_keep:
            del st.session_state[key]

def get_system_status() -> dict:
    """시스템 상태 확인"""
    import psutil
    return {
        'memory_usage': psutil.Process().memory_info().rss / 1024 / 1024,  # MB
        'cpu_usage': psutil.cpu_percent(),
        'disk_usage': psutil.disk_usage('/').percent,
        'active_threads': threading.active_count()
    }

def optimize_response(response: str) -> str:
    """응답 최적화"""
    # 불필요한 공백 제거
    response = ' '.join(response.split())
    # 마크다운 포맷팅 정리
    response = re.sub(r'\n{3,}', '\n\n', response)
    # 이모지 정규화
    response = response.encode('unicode-escape').decode('utf-8')
    response = re.sub(r'\\U000([0-9a-fA-F]{5})', r'\\u\1', response)
    response = response.encode('utf-8').decode('unicode-escape')
    return response