import os
import dotenv
from time import time
import streamlit as st
from typing import Dict, Any

from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders import (
    WebBaseLoader, 
    PyPDFLoader, 
    Docx2txtLoader,
)
# pip install docx2txt, pypdf
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.embeddings import HuggingFaceEmbeddings

dotenv.load_dotenv()


os.environ["USER_AGENT"] = "myagent"
DB_DOCS_LIMIT = 10

# Function to stream the response of the LLM 
# def stream_llm_response(llm_stream, messages):
#     response_message = ""

#     for chunk in llm_stream.stream(messages):
#         response_message += chunk.content
#         yield chunk

#     st.session_state.messages.append({"role": "assistant", "content": response_message})

def stream_llm_response(llm_stream, messages):
    """LLM 응답을 스트리밍하고 상태를 관리합니다."""
    response_message = ""
    
    # 마지막 메시지의 커스터마이징 요청 여부 확인
    is_customization_request = any(
        keyword in messages[-1].content.lower() 
        for keyword in ["이모지", "문체", "길이", "독해"]
    )
    
    # 커스터마이징 요청이면 이전 메시지의 텍스트를 사용
    if is_customization_request and len(messages) > 1:
        base_text = messages[-2].content
        customization_options = {
            "emoji": {"selected": "이모지" in messages[-1].content.lower(), "density": "중간"},
            "style": {"selected": "문체" in messages[-1].content.lower(), "type": "기본"},
            "length": {"selected": "길이" in messages[-1].content.lower(), "target": len(base_text)},
            "level": {"selected": "독해" in messages[-1].content.lower(), "target": "고등학교"}
        }
        
        # 커스터마이징 프롬프트 생성
        prompt = get_customization_prompt(base_text, customization_options)
        if prompt:
            for chunk in llm_stream.stream(prompt):
                chunk_content = chunk.content if hasattr(chunk, 'content') else str(chunk)
                response_message += chunk_content
                yield chunk_content
    else:
        # 일반 응답 생성
        for chunk in llm_stream.stream(messages):
            chunk_content = chunk.content if hasattr(chunk, 'content') else str(chunk)
            response_message += chunk_content
            yield chunk_content

    # 상태 업데이트
    st.session_state.original_response = response_message
    st.session_state.current_text = response_message
    st.session_state.messages.append({"role": "assistant", "content": response_message})
    return response_message  # 전체 응답 반환


def load_doc_to_db():
    # Use loader according to doc type
    if "rag_docs" in st.session_state and st.session_state.rag_docs:
        docs = [] 
        for doc_file in st.session_state.rag_docs:
            if doc_file.name not in st.session_state.rag_sources:
                if len(st.session_state.rag_sources) < DB_DOCS_LIMIT:
                    os.makedirs("source_files", exist_ok=True)
                    file_path = f"./source_files/{doc_file.name}"
                    with open(file_path, "wb") as file:
                        file.write(doc_file.read())

                    try:
                        if doc_file.type == "application/pdf":
                            loader = PyPDFLoader(file_path)
                        elif doc_file.name.endswith(".docx"):
                            loader = Docx2txtLoader(file_path)
                        elif doc_file.type in ["text/plain", "text/markdown"]:
                            loader = TextLoader(file_path)
                        else:
                            st.warning(f"문서 유형 {doc_file.type}은 지원되지 않습니다.")
                            continue

                        docs.extend(loader.load())
                        st.session_state.rag_sources.append(doc_file.name)

                    except Exception as e:
                        st.toast(f"{doc_file.name} 문서를 로드하는 중 오류가 발생했습니다: {e}", icon="⚠️")
                        print(f"{doc_file.name} 문서를 로드하는 중 오류가 발생했습니다: {e}")
                    
                    finally:
                        os.remove(file_path)

                else:
                    st.error(F"문서의 최대 개수({DB_DOCS_LIMIT})에 도달했습니다.")

        if docs:
            _split_and_load_docs(docs)
            st.toast(f"문서 *{str([doc_file.name for doc_file in st.session_state.rag_docs])[1:-1]}* 성공적으로 로드되었습니다.", icon="✅")


def load_url_to_db():
    if "rag_url" in st.session_state and st.session_state.rag_url:
        url = st.session_state.rag_url
        docs = []
        if url not in st.session_state.rag_sources:
            if len(st.session_state.rag_sources) < 10:
                try:
                    loader = WebBaseLoader(url)
                    docs.extend(loader.load())
                    st.session_state.rag_sources.append(url)

                except Exception as e:
                    st.error(f"{url}에서 문서를 로드하는 중 오류가 발생했습니다: {e}")

                if docs:
                    _split_and_load_docs(docs)
                    st.toast(f"*{url}*에서 문서가 성공적으로 로드되었습니다.", icon="✅")

            else:
                st.error("문서의 최대 개수(10개)에 도달했습니다.")

def initialize_vector_db(docs):
    openai_api_key = st.session_state.get('openai_api_key')

    if openai_api_key and len(openai_api_key.strip()) > 0:
        embedding = OpenAIEmbeddings(api_key=st.session_state.openai_api_key)
    
    else:
        embedding = HuggingFaceEmbeddings(
            model_name='jhgan/ko-sroberta-nli',
            model_kwargs={'device':'cpu'},
            encode_kwargs={'normalize_embeddings':True},
            )

    vector_db = Chroma.from_documents(
        documents=docs,
        embedding=embedding,
        collection_name=f"{str(time()).replace('.', '')[:14]}_" + st.session_state['session_id'],
    )

    # We need to manage the number of collections that we have in memory, we will keep the last 20
    chroma_client = vector_db._client
    collection_names = sorted([collection.name for collection in chroma_client.list_collections()])
    print("콜렉션 수:", len(collection_names))
    while len(collection_names) > 20:
        chroma_client.delete_collection(collection_names[0])
        collection_names.pop(0)

    return vector_db
    
    

def _split_and_load_docs(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000,
        chunk_overlap=1000,
    )

    document_chunks = text_splitter.split_documents(docs)

    if "vector_db" not in st.session_state:
        st.session_state.vector_db = initialize_vector_db(docs)
    else:
        st.session_state.vector_db.add_documents(document_chunks)


# --- Retrieval Augmented Generation (RAG) Phase ---

def _get_context_retriever_chain(vector_db, llm):
    retriever = vector_db.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="messages"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get inforamtion relevant to the conversation, focusing on the most recent messages."),
    ])
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

    return retriever_chain


def get_conversational_rag_chain(llm):
    retriever_chain = _get_context_retriever_chain(st.session_state.vector_db, llm)
    
    prompt = ChatPromptTemplate.from_messages([
        ("""You are an SEO Expert and Content Storyteller, specialized in creating optimized blog posts for Naver Blog and Tistory. You are going to create blog posts with strong SEO and compelling storytelling for a Naver Smart Store operator. Here is how you will develop these blog posts:
        Step 1: Keyword Research: Identify relevant and high-traffic keywords for the Naver Smart Store's niche. Use tools like Naver Keyword Planner or Google Keyword Planner to find these keywords.
        Step 2: Content Structure Planning: Outline the structure of the blog post. Ensure it includes an engaging introduction, informative body, and a compelling conclusion.
        Step 3: SEO Optimization: Integrate the identified keywords naturally throughout the content. Ensure the use of meta tags, alt texts for images, and proper header tags (H1, H2, H3).
        Step 4: Storytelling Integration: Weave a compelling story around the product or service being promoted. Use real-life examples, customer testimonials, or hypothetical scenarios to make the content engaging.
        Step 5: Proofreading and Editing: Review the content for any grammatical errors, readability, and SEO compliance. Ensure the content is polished and professional.
        Now, proceed to execute the following task: 'AI를 활용하여 최적화된 SEO를 갖춘 네이버 블로그나 티스토리용 게시물 작성'.
        Take a deep breath and lets work this out in a step by step way to be sure we have the right answer and only use Korean.\n 
         {context}"""),
        MessagesPlaceholder(variable_name="messages"),
        ("user", "{input}"),
    ])
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)

    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

# def stream_llm_rag_response(llm_stream, messages):
#     conversation_rag_chain = get_conversational_rag_chain(llm_stream)
#     response_message = "*(RAG Response)*\n"
#     for chunk in conversation_rag_chain.pick("answer").stream({"messages": messages[:-1], "input": messages[-1].content}):
#         response_message += chunk
#         yield chunk

#     st.session_state.messages.append({"role": "assistant", "content": response_message})

def get_customization_prompt(text: str, options: Dict[str, Any]) -> str:
    """선택된 커스터마이징 옵션에 따른 프롬프트를 생성합니다."""
    prompts = []
    
    if options["emoji"]["selected"]:
        prompts.append(f"""
        다음 한국어 텍스트에 적절한 이모지를 추가해주세요.
        이모지 빈도: {options['emoji']['density']}
        주의사항:
        1. 문맥에 맞는 적절한 이모지만 사용
        2. 이모지는 문장 끝이나 제목에 배치
        3. 원문의 의미를 해치지 않도록 주의
        """)
    
    if options["style"]["selected"]:
        prompts.append(f"""
        텍스트의 문체를 '{options['style']['type']}'스타일로 변경해주세요.
        주의사항:
        1. 핵심 내용 유지
        2. 자연스러운 흐름 유지
        3. 이모지가 있다면 유지
        """)
    
    if options["length"]["selected"]:
        target_length = options["length"]["target"]
        current_length = len(text)
        prompts.append(f"""
        텍스트를 약 {target_length:,}자로 {'확장' if target_length > current_length else '축소'}해주세요.
        주의사항:
        1. 핵심 메시지 유지
        2. {'더 자세한 설명과 예시 추가' if target_length > current_length else '중요한 내용 위주로 간결하게 정리'}
        3. 자연스러운 흐름 유지
        4. 이모지와 문체 특성 유지
        """)
    
    if options["level"]["selected"]:
        prompts.append(f"""
        텍스트를 {options['level']['target']} 수준에 맞게 수정해주세요.
        주의사항:
        1. 핵심 내용 유지
        2. 목표 독자의 이해력 수준에 맞게 조정
        3. 자연스러운 흐름 유지
        4. 이모지와 문체 특성 유지
        """)
    
    if prompts:
        final_prompt = "\n".join(prompts) + f"\n\n텍스트:\n{text}"
        return final_prompt
    
    return ""

def stream_llm_rag_response(llm_stream, messages):
    """RAG 기반 LLM 응답을 스트리밍하고 상태를 관리합니다."""
    conversation_rag_chain = get_conversational_rag_chain(llm_stream)
    response_message = "*(RAG Response)*\n"
    
    # 마지막 메시지의 커스터마이징 요청 여부 확인
    is_customization_request = any(
        keyword in messages[-1].content.lower() 
        for keyword in ["이모지", "문체", "길이", "독해"]
    )
    
    if is_customization_request and len(messages) > 1:
        # 이전 메시지의 텍스트를 사용하여 커스터마이징
        base_text = messages[-2].content
        customization_options = {
            "emoji": {"selected": "이모지" in messages[-1].content.lower(), "density": "중간"},
            "style": {"selected": "문체" in messages[-1].content.lower(), "type": "기본"},
            "length": {"selected": "길이" in messages[-1].content.lower(), "target": len(base_text)},
            "level": {"selected": "독해" in messages[-1].content.lower(), "target": "고등학교"}
        }
        
        prompt = get_customization_prompt(base_text, customization_options)
        if prompt:
            for chunk in llm_stream.stream(prompt):
                chunk_content = chunk.content if hasattr(chunk, 'content') else str(chunk)
                response_message += chunk_content
                yield chunk_content
    else:
        # 일반 RAG 응답 생성
        for chunk in conversation_rag_chain.pick("answer").stream(
            {"messages": messages[:-1], "input": messages[-1].content}
        ):
            chunk_content = str(chunk) if isinstance(chunk, str) else chunk.content
            response_message += chunk_content
            yield chunk_content

    # 상태 업데이트
    st.session_state.original_response = response_message
    st.session_state.current_text = response_message
    st.session_state.messages.append({"role": "assistant", "content": response_message})
    return response_message  # 전체 응답 반환