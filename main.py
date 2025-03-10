import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents.base import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyMuPDFLoader
from typing import List
import os
import openai

# 환경 변수 불러오기
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv("YOURKEY")

### PDF Embeddings

def pdf_to_documents(pdf_path: str) -> List[Document]:
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()
    for d in documents:
        d.metadata['file_path'] = pdf_path
    return documents

def chunk_documents(documents: List[Document]) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    return text_splitter.split_documents(documents)

def save_to_vector_store(documents: List[Document]) -> None:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.from_documents(documents, embedding=embeddings)
    vector_store.save_local("faiss_index")

def main():
    st.set_page_config(page_title="디지털경영 챗봇", layout="wide")
    st.header("디지털경영전공 챗봇")
    st.text("질문하고 싶은 내용을 입력해주세요")

    # PDF 업로드 기능 추가
    uploaded_file = st.file_uploader("PDF 파일을 업로드하세요.", type=["pdf"])
    if uploaded_file:
        pdf_path = f"temp_{uploaded_file.name}"
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # PDF 처리
        documents = pdf_to_documents(pdf_path)
        chunked_docs = chunk_documents(documents)
        save_to_vector_store(chunked_docs)
        st.success("PDF가 성공적으로 업로드 및 저장되었습니다.")

    # 질문 입력
    user_question = st.text_input("궁금한 내용을 입력하세요:")
    if user_question:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        retriever = new_db.as_retriever(search_kwargs={"k": 3})
        retrieve_docs = retriever.invoke(user_question)

        chain = get_rag_chain()
        response = chain.invoke({"question": user_question, "context": retrieve_docs})

        st.markdown("### 응답:")
        st.write(response)
        
        with st.expander("관련 문서 보기"):
            for document in retrieve_docs:
                st.write(document.page_content)

def get_rag_chain() -> Runnable:
    template = """
    아래 컨텍스트를 바탕으로 질문에 답해주세요:
    - 질문에 대한 응답은 5줄 이내로 간결하게 작성해주세요.
    - 애매하거나 모르는 내용은 "잘 모르겠습니다"라고 답변해주세요.
    - 공손한 표현을 사용해주세요.

    컨텍스트: {context}

    질문: {question}

    응답:"""
    custom_rag_prompt = PromptTemplate.from_template(template)
    model = ChatOpenAI(model="gpt-4o")
    return custom_rag_prompt | model | StrOutputParser()

if __name__ == "__main__":
    main()
    
