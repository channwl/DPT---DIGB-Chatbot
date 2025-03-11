import streamlit as st
from PyPDF2 import PdfReader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

# Streamlit UI 구성
st.set_page_config(page_title="학과 챗봇", page_icon="🎓", layout="wide")
st.title("📚 학과 정보 챗봇")
st.sidebar.header("📂 PDF 업로드")

# PDF 업로드 기능
uploaded_files = st.sidebar.file_uploader("학과 정보 PDF 파일을 업로드하세요.", accept_multiple_files=True, type=["pdf"])

if uploaded_files:
    st.sidebar.success(f"{len(uploaded_files)}개의 파일 업로드 완료!")
    
    text_data = ""
    
    for pdf in uploaded_files:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text_data += page.extract_text() + "\n"
    
    # 텍스트 분할 및 임베딩 생성
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_text(text_data)
    embeddings = OpenAIEmbeddings()
    vector_db = FAISS.from_texts(texts, embeddings)
    retriever = vector_db.as_retriever()
    
    llm = OpenAI(model_name="gpt-4")
    qa_chain = RetrievalQA(llm=llm, retriever=retriever)
    
    # 챗봇 인터페이스
    st.subheader("🤖 학과 챗봇과 대화하기")
    user_input = st.text_input("질문을 입력하세요:")
    
    if user_input:
        response = qa_chain.run(user_input)
        st.write("**챗봇 답변:**", response)
        
else:
    st.sidebar.warning("PDF 파일을 업로드해주세요.")
