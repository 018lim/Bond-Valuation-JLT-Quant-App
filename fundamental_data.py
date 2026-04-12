import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import datetime
import os
import tempfile
import json
import re
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma

load_dotenv()

def extract_financials_via_rag(*args):
    st.markdown("### 📂 재무 데이터 업로드 (Gemini RAG)")
    
    data_source = st.radio("PDF 데이터 소스를 선택하세요:", ["기본 샘플 파일 사용 (현대차 사업보고서)", "직접 PDF 파일 업로드"])
    
    pdf_files_to_process = []
    
    if data_source == "기본 샘플 파일 사용 (현대차 사업보고서)":
        st.info("💡 앱에 내장된 샘플 파일(`2025_hyundai.pdf`)을 분석합니다. (샘플 분석 시 연간 수치를 우선으로 찾습니다)")
        if os.path.exists("2025_hyundai.pdf"):
            pdf_files_to_process = ["2025_hyundai.pdf"]
        else:
            st.error("⚠️ 폴더에 '2025_hyundai.pdf' 파일이 없습니다!")
            
    else:
        uploaded_files = st.file_uploader("📄 사업보고서 및 IR 자료 업로드 (여러 개 가능)", type=["pdf"], accept_multiple_files=True)
        if uploaded_files:
            pdf_files_to_process = uploaded_files
    
    if "analyze_started" not in st.session_state:
        st.session_state["analyze_started"] = False

    if st.button("🚀 AI 분석 시작"):
        if not pdf_files_to_process:
            st.warning("분석할 PDF 파일이 선택/업로드되지 않았습니다.")
            return None
        st.session_state["analyze_started"] = True
        st.session_state["rag_df"] = None 

    if st.session_state["analyze_started"]:
        if st.session_state.get("rag_df") is None:
            with st.spinner("AI가 PDF에서 연간 재무제표를 정밀 검색 중입니다..."):
                final_data = {"영업이익": 0.0, "이자비용": 0.0, "이자보상배율(배)": 0.0, "부채비율(%)": 0.0}
                
                documents = []
                for pdf_item in pdf_files_to_process:
                    if isinstance(pdf_item, str): 
                        loader = PyPDFLoader(pdf_item)
                        documents.extend(loader.load())
                    else:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                            tmp_file.write(pdf_item.getvalue())
                            tmp_file_path = tmp_file.name
                        loader = PyPDFLoader(tmp_file_path)
                        documents.extend(loader.load())
                        os.remove(tmp_file_path)
                
                # 🚀 텍스트 분할 및 정제 (IndexError 방지 방탄 코드)
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
                all_texts = text_splitter.split_documents(documents)
                
                # 내용이 비어있지 않고 특수문자가 정제된 조각만 남김
                texts = []
                for doc in all_texts:
                    clean_content = doc.page_content.replace('\x00', '').strip()
                    if len(clean_content) > 10: # 너무 짧은 조각은 임베딩 시 에러 유발 가능
                        doc.page_content = clean_content
                        texts.append(doc)

                if not texts:
                    st.error("⚠️ PDF에서 읽을 수 있는 텍스트가 없습니다. 파일 형식을 확인해주세요.")
                    return None

                # 🌟 벡터 저장소 구축
                embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2-preview")
                # 💡 개수가 안 맞아서 생기는 IndexError를 방지하기 위해 from_documents 대신 안전 필터링 거친 후 생성
                vectorstore = Chroma.from_documents(documents=texts, embedding=embeddings)
                
                retriever = vectorstore.as_retriever(search_kwargs={"k": 5}) 
                llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

                query = """
                당신은 숙련된 퀀트 애널리스트입니다. 문서 전체를 뒤져 다음 재무 지표를 찾으세요.
                1. 무조건 연간(Annual, FY) 또는 4분기 누적(12M) 수치만 가져오세요.
                2. 영업이익, 이자비용, 부채비율을 정확히 추출하세요.
                JSON 형식으로만 답변하세요:
                {"영업이익": 0, "이자비용": 0, "이자보상배율(배)": 0, "부채비율(%)": 0}
                """
                
                docs = retriever.invoke(query)
                context_text = "\n\n".join([doc.page_content for doc in docs])
                final_prompt = f"다음 [참고 문서]를 바탕으로 [질문]에 답하세요.\n\n[참고 문서]\n{context_text}\n\n[질문]\n{query}"
                response = llm.invoke(final_prompt)
                
                json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
                if json_match:
                    try:
                        pdf_data = json.loads(json_match.group())
                        for k, v in pdf_data.items():
                            if k in final_data:
                                clean_v = str(v).replace(',', '').replace('%', '').strip()
                                final_data[k] = float(clean_v)
                    except: pass

                st.session_state["rag_df"] = pd.DataFrame([final_data])

        st.markdown("### 📊 분석 완료 (수동 수정 가능)")
        edited_df = st.session_state["rag_df"].copy()
        
        col1, col2 = st.columns(2)
        with col1:
            op = st.number_input("연간 영업이익", value=float(edited_df.at[0, "영업이익"]))
            ie = st.number_input("연간 이자비용", value=float(edited_df.at[0, "이자비용"]))
        with col2:
            auto_icr = 0.0 if ie == 0 else round(op / ie, 2)
            st.number_input("이자보상배율(배) - 실시간 계산", value=auto_icr, disabled=True)
            debt = st.number_input("부채비율(%)", value=float(edited_df.at[0, "부채비율(%)"]))
        
        edited_df.at[0, "영업이익"] = op
        edited_df.at[0, "이자비용"] = ie
        edited_df.at[0, "이자보상배율(배)"] = auto_icr
        edited_df.at[0, "부채비율(%)"] = debt
        
        st.table(edited_df)
        if st.button("💾 AI 데이터 수정 및 DB 적용"):
            if 'DB' in st.session_state:
                st.session_state.DB["op"] = op
                st.session_state.DB["ie"] = ie
                st.session_state.DB["debt"] = debt
            st.success("✅ 수정한 데이터가 메인 DB에 저장되었습니다!")
            
        return edited_df
    return None

def get_company_list():
    return ["PDF 분석 (Gemini RAG)", "직접 입력"]

def generate_investment_report(h_mkt, h_fund, gap, fin_data):
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
    
    # 💡 오늘 날짜를 동적으로 생성 (예: 2026년 03월 12일)
    today_date = datetime.date.today().strftime("%Y년 %m월 %d일")
    
    prompt = f"""
    당신은 시니어 퀀트 채권 애널리스트입니다. 아래 데이터를 바탕으로 기관 투자자에게 보고할 종합 투자 리포트를 작성해주세요.

    [보고서 기본 정보]
    - 작성일자: {today_date}
    - 제목 형식: [분석 대상 기업/종목명] 퀀트 채권 평가 리포트 - [핵심 한줄 요약]

    [분석 데이터]
    - 시장 내재 부도강도 (JLT 모델): {h_mkt * 100:.2f}%
    - 재무 내재 부도강도 (스코어카드): {h_fund * 100:.2f}%
    - 괴리율 (Gap = 시장 - 재무): {gap:.0f} bp
    - 연간 영업이익: {fin_data.get('영업이익', 0):,}
    - 연간 이자비용: {fin_data.get('이자비용', 0):,}
    - 이자보상배율: {fin_data.get('이자보상배율(배)', 0)}배
    - 부채비율: {fin_data.get('부채비율(%)', 0)}%

    [리포트 작성 가이드]
    1. 핵심 요약: 시장이 신용 위험을 과대평가(저평가 구간)하는지, 과소평가(고평가 구간)하는지 진단.
    2. 재무 펀더멘털 점검: 기업의 실제 이자 지급 능력 평가.
    3. 괴리율(Gap) 분석: 시장 가격과 재무 상태 간의 차이 발생 이유 가설 제시.
    4. 최종 투자 의견: 적극 매수, 매수, 보유, 매도 중 하나 제시 및 요약.
    어조는 전문적이고 단호하게 작성.
    """
    response = llm.invoke(prompt)
    return response.content
