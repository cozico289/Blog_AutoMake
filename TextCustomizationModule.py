import streamlit as st
from typing import Dict, Any

def create_customization_prompt(text: str, options: Dict[str, Any]) -> str:
    """선택된 커스터마이징 옵션에 따른 프롬프트를 생성합니다."""
    prompts = []
    
    if options["emoji"]["selected"]:
        prompts.append(f"""
        1. 이모지 요구사항:
        - 이모지 빈도: {options['emoji']['density']}
        - 문맥에 맞는 적절한 이모지만 사용
        - 이모지는 문장 끝이나 제목, 중요 키워드 옆에 배치
        - 이모지 중복 사용 최소화
        - {options['emoji']['density']}에 따른 이모지 수:
          * 낮음: 3-4문장당 1개
          * 중간: 2-3문장당 1개
          * 높음: 1-2문장당 1개
        """)
    
    if options["style"]["selected"]:
        style_guidelines = {
            "기본": "자연스럽고 객관적인 문체, 적절한 전문성과 대중성의 균형",
            "전문적": "학술적이고 전문적인 용어 사용, 객관적이고 논리적인 전개",
            "친근한": "일상적 어휘, 대화체, 독자와 공감대 형성하는 표현 활용",
            "격식있는": "공식적이고 예의 바른 표현, 높임말 사용, 정중한 어조"
        }
        
        prompts.append(f"""
        2. 문체 요구사항:
        - 선택된 문체: {options['style']['type']}
        - 문체 특성: {style_guidelines[options['style']['type']]}
        - 기존 이모지 유지
        - 핵심 메시지와 정보는 유지
        """)
    
    if options["length"]["selected"]:
        current_length = len(text)
        target_length = options["length"]["target"]
        action = "확장" if target_length > current_length else "축소"
        
        prompts.append(f"""
        3. 길이 조정 요구사항:
        - 목표 길이: {target_length:,}자 ({action})
        - {action}시 주의사항:
          * {'더 자세한 설명과 예시 추가' if action == '확장' else '중요한 내용 위주로 간결하게 정리'}
          * 자연스러운 흐름 유지
          * 기존 이모지와 문체 특성 유지
          * 문단 구조 유지
        """)
    
    if options["level"]["selected"]:
        level_guidelines = {
            "유치원": "기초 어휘, 짧은 문장, 반복 설명, 쉬운 예시",
            "초등학교": "기본 어휘, 간단한 문장, 구체적 예시",
            "중학교": "일상 어휘, 기본 전문용어, 논리적 구조",
            "고등학교": "다양한 어휘, 복잡한 문장, 심화 개념",
            "대학교": "전문 용어, 학술적 표현, 깊이 있는 분석",
            "전문가": "고급 전문용어, 학술적 문체, 전문적 분석"
        }
        
        prompts.append(f"""
        4. 독해 수준 조정 요구사항:
        - 목표 수준: {options['level']['target']}
        - 수준 특성: {level_guidelines[options['level']['target']]}
        - 핵심 내용 유지
        - 기존 이모지와 문체 특성 유지
        - 단계적 설명 방식 사용
        """)
    
    if prompts:
        return f"""다음 요구사항에 따라 텍스트를 수정해주세요:

{' '.join(prompts)}

원본 텍스트:
{text}

수정된 텍스트를 반환할 때 다음 사항을 지켜주세요:
1. 설명이나 메타 정보 없이 수정된 텍스트만 반환
2. 마크다운 형식 유지
3. 단락 구분 유지"""
    
    return ""

def customize_text(full_response: str, llm_stream) -> str:
    """텍스트 커스터마이징 기능을 제공합니다."""
    try:
        # 커스터마이징 UI 구성
        st.divider()
        st.subheader("✨ 텍스트 커스터마이징")
        
        # 상태 초기화
        if "current_text" not in st.session_state:
            st.session_state.current_text = full_response
            
        if "customization_options" not in st.session_state:
            st.session_state.customization_options = {
                "emoji": {"selected": False, "density": "중간"},
                "style": {"selected": False, "type": "기본"},
                "length": {"selected": False, "target": len(full_response)},
                "level": {"selected": False, "target": "고등학교"}
            }
        
        # 텍스트 표시
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 원본 텍스트")
            st.text_area(
                "원문",
                full_response,
                height=200,
                disabled=True,
                key="original_text",
                label_visibility="collapsed",
            )
            
        with col2:
            st.markdown("### 현재 텍스트")
            st.text_area(
                "현재 텍스트",
                st.session_state.current_text,
                height=200,
                disabled=True,
                key="current_text_display",
                label_visibility="collapsed",
            )

        # 커스터마이징 폼
        with st.form(key="customization_form"):
            tab1, tab2, tab3, tab4 = st.tabs(["🎯 이모지", "✍️ 문체", "📏 길이", "📚 독해수준"])
            
            # 탭 1: 이모지 설정
            with tab1:
                st.markdown("### 이모지 설정")
                emoji_selected = st.checkbox(
                    "이모지 추가 적용",
                    value=st.session_state.customization_options["emoji"]["selected"],
                    key="emoji_checkbox",
                    help="텍스트에 적절한 이모지를 추가합니다."
                )
                
                density_descriptions = {
                    "낮음": "3-4문장당 1개의 이모지 (공식적인 글에 적합)",
                    "중간": "2-3문장당 1개의 이모지 (일반적인 블로그 글에 적합)",
                    "높음": "1-2문장당 1개의 이모지 (가벼운 주제나 SNS 글에 적합)"
                }
                
                emoji_density = st.radio(
                    "이모지 빈도",
                    options=list(density_descriptions.keys()),
                    format_func=lambda x: f"{x} - {density_descriptions[x]}",
                    horizontal=True,
                    index=["낮음", "중간", "높음"].index(
                        st.session_state.customization_options["emoji"]["density"]
                    ),
                    key="emoji_density",
                )

            # 탭 2: 문체 설정
            with tab2:
                st.markdown("### 문체 설정")
                style_selected = st.checkbox(
                    "문체 변경 적용",
                    value=st.session_state.customization_options["style"]["selected"],
                    key="style_checkbox",
                    help="텍스트의 전반적인 문체를 변경합니다."
                )
                
                style_types = {
                    "기본": "자연스럽고 균형잡힌 문체",
                    "전문적": "전문성과 신뢰성이 느껴지는 문체",
                    "친근한": "편안하고 대화체 느낌의 문체",
                    "격식있는": "예의 바르고 공손한 문체"
                }
                
                style = st.select_slider(
                    "문체 선택",
                    options=list(style_types.keys()),
                    value=st.session_state.customization_options["style"]["type"],
                    format_func=lambda x: f"{x} - {style_types[x]}",
                    key="style_select"
                )

            # 탭 3: 길이 설정
            with tab3:
                st.markdown("### 길이 설정")
                length_selected = st.checkbox(
                    "길이 조정 적용",
                    value=st.session_state.customization_options["length"]["selected"],
                    key="length_checkbox",
                    help="텍스트의 전체 길이를 조정합니다."
                )
                
                current_length = len(st.session_state.current_text)
                
                # 동적 길이 범위 설정
                min_length = max(100, current_length // 2)
                max_length = min(5000, current_length * 2)
                step_size = max(50, (max_length - min_length) // 20)
                
                default_length = st.session_state.customization_options["length"]["target"]
                default_length = max(min_length, min(max_length, default_length))
                
                if min_length < max_length:
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        target_length = st.slider(
                            "목표 길이 (자)",
                            min_value=min_length,
                            max_value=max_length,
                            value=default_length,
                            step=step_size,
                            key="length_slider"
                        )
                    with col2:
                        st.metric(
                            "현재 길이",
                            f"{current_length:,}자",
                            delta=f"{target_length - current_length:,}자",
                            delta_color="normal"
                        )
                else:
                    st.warning("텍스트가 너무 짧아 길이 조정을 할 수 없습니다.")
                    target_length = current_length
                    length_selected = False

            # 탭 4: 독해 수준 설정
            with tab4:
                st.markdown("### 독해 수준 설정")
                level_selected = st.checkbox(
                    "독해 수준 조정 적용",
                    value=st.session_state.customization_options["level"]["selected"],
                    key="level_checkbox",
                    help="텍스트의 난이도와 이해하기 쉬운 정도를 조정합니다."
                )
                
                level_descriptions = {
                    "유치원": "매우 쉬운 단어와 짧은 문장 사용",
                    "초등학교": "기본적인 단어와 간단한 문장 구조",
                    "중학교": "일상적인 어휘와 기본 전문용어",
                    "고등학교": "다양한 어휘와 복잡한 문장",
                    "대학교": "전문적인 어휘와 학술적 문체",
                    "전문가": "고급 전문용어와 형식적 학술 문체"
                }
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    level = st.radio(
                        "독해 수준 선택",
                        options=list(level_descriptions.keys()),
                        index=list(level_descriptions.keys()).index(
                            st.session_state.customization_options["level"]["target"]
                        ),
                        format_func=lambda x: f"{x} - {level_descriptions[x]}",
                        key="reading_level",
                        horizontal=True
                    )
                with col2:
                    st.info(f"현재 수준: {st.session_state.customization_options['level']['target']}")

            # 버튼 컨테이너
            col1, col2 = st.columns([1, 1])
            with col1:
                reset_button = st.form_submit_button(
                    "수정 사항 초기화", 
                    type="secondary",
                    use_container_width=True,
                    help="모든 커스터마이징 옵션을 초기화합니다."
                )
            with col2:
                apply_button = st.form_submit_button(
                    "변경 사항 적용", 
                    type="primary",
                    use_container_width=True,
                    help="선택한 커스터마이징 옵션을 적용합니다."
                )

        # 버튼 동작 처리
        if reset_button:
            st.session_state.current_text = full_response
            st.session_state.customization_options = {
                "emoji": {"selected": False, "density": "중간"},
                "style": {"selected": False, "type": "기본"},
                "length": {"selected": False, "target": len(full_response)},
                "level": {"selected": False, "target": "고등학교"}
            }
            st.toast("커스터마이징 옵션이 초기화되었습니다.", icon="🔄")
            st.rerun()

        if apply_button:
            # 옵션 상태 업데이트
            options = {
                "emoji": {"selected": emoji_selected, "density": emoji_density},
                "style": {"selected": style_selected, "type": style},
                "length": {"selected": length_selected, "target": target_length},
                "level": {"selected": level_selected, "target": level}
            }
            
            # 선택된 옵션 확인
            selected_options = [
                opt for opt, val in {
                    "emoji": emoji_selected,
                    "style": style_selected,
                    "length": length_selected,
                    "level": level_selected
                }.items() if val
            ]
            
            if not selected_options:
                st.warning("적용할 변경 사항이 없습니다.")
                return st.session_state.current_text
            
            # 프롬프트 생성 및 적용
            prompt = create_customization_prompt(st.session_state.current_text, options)
            if prompt:
                try:
                    with st.spinner("커스터마이징 적용 중..."):
                        response = llm_stream.invoke(prompt)
                        modified_text = response.content
                        
                        if modified_text != st.session_state.current_text:
                            st.session_state.current_text = modified_text
                            st.session_state.messages[-1]["content"] = modified_text
                            st.session_state.message_placeholder.markdown(modified_text)
                            
                            # 적용된 옵션 표시
                            option_names = {
                                "emoji": "이모지",
                                "style": "문체",
                                "length": "길이",
                                "level": "독해수준"
                            }
                            applied_options = [option_names[opt] for opt in selected_options]
                            st.success(f"다음 옵션이 적용되었습니다: {', '.join(applied_options)}")
                        
                        return modified_text
                
                except Exception as e:
                    st.error(f"커스터마이징 적용 중 오류가 발생했습니다: {str(e)}")
                    return st.session_state.current_text
        
        return st.session_state.current_text
        
    except Exception as e:
        st.error(f"텍스트 커스터마이징 중 오류가 발생했습니다: {str(e)}")
        return full_response