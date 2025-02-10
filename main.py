import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
import google.generativeai as genai
import tempfile
from datetime import datetime

# 환경 변수 로드
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

# 사이드바 설정
st.sidebar.title("설정")

# 언어 설정
languages = {
    "한국어": "ko",
    "English": "en",
    "日本語": "ja"
}
selected_language = st.sidebar.selectbox("언어 선택", list(languages.keys()))

# AI 응답 설정
temperature = st.sidebar.slider("창의성 조절", 0.0, 1.0, 0.3)
max_tokens = st.sidebar.slider("최대 응답 길이", 100, 2000, 1000)

def show_pdf_info(uploaded_file):
    """PDF 파일 정보 표시"""
    st.sidebar.header("PDF 정보")
    file_details = {
        "파일명": uploaded_file.name,
        "파일크기": f"{uploaded_file.size / 1024:.2f} KB",
        "파일유형": uploaded_file.type
    }
    for key, value in file_details.items():
        st.sidebar.text(f"{key}: {value}")

def process_pdf(uploaded_file):
    """PDF 처리"""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(pages)
        
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        
        return vectorstore, chunks
        
    finally:
        os.unlink(tmp_path)

def get_gemini_response(prompt, temperature=0.3, max_tokens=1000, lang="ko"):
    """Gemini API 응답 생성"""
    model = genai.GenerativeModel('gemini-pro')
    
    # 언어 설정 추가
    if lang != "ko":
        prompt = f"Please respond in {lang}.\n\n" + prompt
        
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
    )
    return response.text

def get_relevant_context(vectorstore, question, k=5):
    """질문과 가장 관련성 높은 문맥 검색"""
    docs = vectorstore.similarity_search_with_score(question, k=k)
    
    # 유사도 점수를 기준으로 필터링 (점수가 낮을수록 더 관련성이 높음)
    relevant_docs = [(doc, score) for doc, score in docs if score < 1.0]
    
    # 상위 3개만 선택
    return relevant_docs[:3]

def main():
    st.title("PDF 문서 기반 챗봇")
    
    # 파일 업로드
    uploaded_file = st.file_uploader("PDF 파일을 업로드하세요", type="pdf")
    
    if uploaded_file is not None:
        # PDF 정보 표시
        show_pdf_info(uploaded_file)
        
        # PDF 처리
        with st.spinner("PDF 처리 중..."):
            vectorstore, chunks = process_pdf(uploaded_file)
            st.success("PDF 처리 완료!")
        
        # 탭 생성
        tab1, tab2 = st.tabs(["챗봇", "문서 분석"])
        
        with tab1:
            # 세션 상태 초기화
            if "messages" not in st.session_state:
                st.session_state.messages = []
            
            # 대화 기록 초기화 버튼
            if st.sidebar.button("대화 기록 초기화"):
                st.session_state.messages = []
                st.experimental_rerun()
            
            # 대화 기록 다운로드 버튼
            if st.sidebar.button("대화 내용 다운로드"):
                chat_history = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages])
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.sidebar.download_button(
                    "대화 내용 저장",
                    chat_history,
                    f"chat_history_{timestamp}.txt",
                    "text/plain"
                )
            
            # 채팅 기록 표시
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            # 사용자 입력
            if prompt := st.chat_input("질문을 입력하세요"):
                # 사용자 메시지 추가
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # 문맥 검색
                relevant_docs = get_relevant_context(vectorstore, prompt)
                context = "\n".join([doc.page_content for doc, _ in relevant_docs])
                
                # 프롬프트 생성
                full_prompt = f"""주어진 문맥을 바탕으로 질문에 답변해주세요.
                
                문맥:
                {context}
                
                질문: {prompt}
                """
                
                # 챗봇 응답
                with st.chat_message("assistant"):
                    with st.spinner("답변 생성 중..."):
                        response = get_gemini_response(
                            full_prompt, 
                            temperature=temperature,
                            max_tokens=max_tokens,
                            lang=languages[selected_language]
                        )
                        st.markdown(response)
                        
                        # 참조 문맥 표시
                        with st.expander("참조한 문맥 보기"):
                            if relevant_docs:
                                for i, (doc, score) in enumerate(relevant_docs, 1):
                                    st.markdown(f"**참조 {i}:** (관련도: {(1-score)*100:.1f}%)")
                                    st.markdown(f"```\n{doc.page_content}\n```")
                                    if hasattr(doc.metadata, 'page') and doc.metadata['page']:
                                        st.caption(f"(페이지: {doc.metadata['page']})")
                                    st.markdown("---")
                            else:
                                st.info("질문과 직접적으로 관련된 문맥을 찾지 못했습니다.")
                
                # 어시스턴트 메시지 추가
                st.session_state.messages.append({"role": "assistant", "content": response})
        
        with tab2:
            st.header("문서 분석")
            
            # 주요 키워드 추출
            if st.button("주요 키워드 분석"):
                with st.spinner("키워드 분석 중..."):
                    analysis_prompt = f"다음 문서에서 주요 키워드 10개를 추출해주세요:\n\n{context}"
                    keywords = get_gemini_response(analysis_prompt)
                    st.write(keywords)
            
            # 문서 요약
            if st.button("문서 요약"):
                with st.spinner("요약 생성 중..."):
                    summary_prompt = f"다음 문서를 간단히 요약해주세요:\n\n{context}"
                    summary = get_gemini_response(summary_prompt)
                    st.write(summary)

if __name__ == "__main__":
    main()
