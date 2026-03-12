import streamlit as st
import pandas as pd
import numpy as np
import market_data
import fundamental_data
import jlt_engine
import risk_metrics  
import os  # 💡 [핵심 버그 수정] 파일 경로를 확인하기 위한 라이브러리 추가!

st.set_page_config(page_title="Adjusted J-L-T Valuation", layout="wide")

# 💡 [중앙 DB]
if 'DB' not in st.session_state:
    st.session_state.DB = {
        "op": 0.0, "ie": 0.0, "debt": 0.0,
        "fv": 10000.0, "mp": 9850.0, "cr": 3.5, "mat": 5.0, "rr": 40.0, "rf": 3.3,
        "mode": "직접 입력"
    }

st.sidebar.title("Bond Valuation Engine")
st.sidebar.markdown("**시장 & 재무 기반 하이브리드 프라이싱**")

step = st.sidebar.radio(
    "단계를 선택하세요:",
    ["Step 1: 시장 환경 설정", 
     "Step 2: 기업 재무 데이터 입력", 
     "Step 3: Adjusted J-L-T 캘리브레이션", 
     "Step 4: 괴리율 분석 및 AI 종합 리포트"] 
)

if step == "Step 1: 시장 환경 설정":
    st.title("Step 1: 현재 시장 커브 확인")
    
    # 💡 엑셀 데이터 소스 선택 옵션!
    data_source = st.radio("시장 데이터 소스를 선택하세요:", ["기본 샘플 파일 사용 (한국자산평가 최신 기준)", "직접 Excel/CSV 파일 업로드"])
    
    file_to_process = None
    
    if data_source == "기본 샘플 파일 사용 (한국자산평가 최신 기준)":
        st.info("💡 앱에 내장된 샘플 파일(`20250312.xlsx`) 데이터를 불러옵니다.")
        if os.path.exists("20250312.xlsx"):
            file_to_process = "20250312.xlsx"
        else:
            st.error("⚠️ 폴더에 '20250312.xlsx' 파일이 없습니다! 엑셀 파일 이름을 확인해주세요.")
            
    else:
        uploaded_file = st.file_uploader("KAP 데이터 업로드", type=["csv", "xlsx", "xls"])
        if uploaded_file is not None:
            file_to_process = uploaded_file
            st.success("✅ 파일이 성공적으로 업로드되었습니다! (실제 데이터 반영)")
        else:
            st.warning("⚠️ 파일을 업로드하지 않으면 기본(Dummy) 데이터로 그래프가 그려집니다.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("국고채 Zero Curve")
        yc_df = market_data.get_yield_curve(file_to_process)
        st.dataframe(yc_df)
        st.line_chart(yc_df.set_index('Maturity(Year)')['RiskFreeRate(%)'])
    with col2:
        st.subheader("등급별 신용 스프레드 (bp)")
        cs_df = market_data.get_credit_spread(file_to_process)
        st.dataframe(cs_df)
        st.line_chart(cs_df.set_index('Maturity(Year)'))

elif step == "Step 2: 기업 재무 데이터 입력":
    st.title("Step 2: 기업 재무 데이터 추출 및 입력")

    company_list = fundamental_data.get_company_list()
    try:
        idx = company_list.index(st.session_state.DB["mode"])
    except ValueError:
        idx = 1
        
    selected_company = st.selectbox("데이터 입력 방식을 선택하세요:", company_list, index=idx)
    st.session_state.DB["mode"] = selected_company

    if selected_company == "PDF 분석 (Gemini RAG)":
        fin_df = fundamental_data.extract_financials_via_rag()
        if fin_df is not None and not fin_df.empty:
            st.session_state.DB["op"] = float(fin_df.at[0, "영업이익"])
            st.session_state.DB["ie"] = float(fin_df.at[0, "이자비용"])
            st.session_state.DB["debt"] = float(fin_df.at[0, "부채비율(%)"])
            # AI 수정 및 저장은 fundamental_data 내부의 버튼에서 처리됨
            
    else:
        st.subheader("사용자 직접 입력 모드")
        st.info("💡 숫자를 입력하신 후 반드시 아래의 **[💾 데이터 저장 및 적용]** 버튼을 눌러주세요!")
        
        col1, col2 = st.columns(2)
        with col1:
            op = st.number_input("연간 영업이익", value=float(st.session_state.DB["op"]))
            ie = st.number_input("연간 이자비용", value=float(st.session_state.DB["ie"]))
        with col2:
            auto_icr = round(op / ie, 2) if ie != 0 else 0.0
            st.number_input("이자보상배율(배) - 실시간 계산", value=auto_icr, disabled=True)
            debt = st.number_input("부채비율(%)", value=float(st.session_state.DB["debt"]))
            
        st.markdown("---")
        if st.button("💾 데이터 저장 및 적용"):
            st.session_state.DB["op"] = op
            st.session_state.DB["ie"] = ie
            st.session_state.DB["debt"] = debt
            st.success(f"✅ 재무 데이터가 완벽하게 저장되었습니다! (적용된 이자보상배율: {auto_icr}배)")

elif step == "Step 3: Adjusted J-L-T 캘리브레이션":
    st.title("Step 3: Adjusted J-L-T 기반 생존확률 최적화")
    st.info("💡 값을 입력하고 아래 버튼을 누르면, **데이터가 DB에 저장된 후** 캘리브레이션이 실행됩니다.")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        fv = st.number_input("액면가 (원)", value=float(st.session_state.DB["fv"]), step=1000.0)
        mp = st.number_input("현재 시장 거래가 (원)", value=float(st.session_state.DB["mp"]), step=10.0)
        cr = st.number_input("표면이율 (Coupon, %)", value=float(st.session_state.DB["cr"]), step=0.1)
        mat = st.number_input("잔존만기 (년)", value=float(st.session_state.DB["mat"]), min_value=1.0)
        rr = st.number_input("예상 회수율 (%)", value=float(st.session_state.DB["rr"]))
        rf = st.number_input("적용 무위험 금리 (%)", value=float(st.session_state.DB["rf"]))
        
        run_calibration = st.button("💾 저장 및 🚀 캘리브레이션 실행")

    with col2:
        if run_calibration or 'mkt_hazard' in st.session_state:
            if run_calibration:
                st.session_state.DB["fv"] = fv
                st.session_state.DB["mp"] = mp
                st.session_state.DB["cr"] = cr
                st.session_state.DB["mat"] = mat
                st.session_state.DB["rr"] = rr
                st.session_state.DB["rf"] = rf

            with st.spinner("최적화 알고리즘 구동 중..."):
                hazard_rate, survival_probs = jlt_engine.calibrate_hazard_rate(
                    st.session_state.DB["mp"], st.session_state.DB["fv"], st.session_state.DB["cr"], 
                    st.session_state.DB["mat"], st.session_state.DB["rf"], st.session_state.DB["rr"] / 100
                )
                st.session_state['mkt_hazard'] = hazard_rate
                
                st.success("✅ Step 3 데이터가 DB에 저장되었고, 캘리브레이션이 완료되었습니다!")
                st.markdown(f"### 🎯 내재 부도강도 (Hazard Rate): **{hazard_rate*100:.4f}%**")
                
                sp_df = pd.DataFrame({
                    "연차": np.arange(1, st.session_state.DB["mat"] + 1),
                    "생존확률(%)": [sp * 100 for sp in survival_probs]
                })
                st.line_chart(sp_df.set_index("연차"))

elif step == "Step 4: 괴리율 분석 및 AI 종합 리포트":
    st.title("Step 4: 시장 vs 재무 괴리율 분석 및 AI 퀀트 리포트")
    
    if st.session_state.DB["ie"] == 0 and st.session_state.DB["debt"] == 0:
        st.warning("⚠️ Step 2에서 재무 데이터를 입력하고 [데이터 저장] 버튼을 먼저 눌러주세요!")
    elif 'mkt_hazard' not in st.session_state:
        st.warning("⚠️ Step 3에서 [저장 및 캘리브레이션 실행] 버튼을 먼저 눌러주세요!")
    else:
        op = st.session_state.DB["op"]
        ie = st.session_state.DB["ie"]
        debt = st.session_state.DB["debt"]
        calc_icr = round(op / ie, 2) if ie != 0 else 0.0
        
        h_mkt = st.session_state['mkt_hazard']
        h_fund, score = risk_metrics.get_fundamental_hazard_rate(debt, calc_icr)
        st.session_state['h_fund'] = h_fund
        
        st.subheader("📊 위험 지표 비교")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("시장 내재 부도강도 (JLT)", f"{h_mkt*100:.2f}%")
        with col2:
            st.metric("재무 내재 부도강도 (Scorecard)", f"{h_fund*100:.2f}%")
            st.caption(f"기업 재무 스코어: {score:.1f}/10점")

        gap = (h_mkt - h_fund) * 10000 
        st.session_state['gap'] = gap
        
        st.markdown("---")
        
        if gap > 50:
            st.success(f"📈 **시장 과매도 상태 (저평가)**: 시장 위험이 재무 위험보다 {gap:.0f}bp 높게 책정되어 있습니다.")
        elif gap < -50:
            st.error(f"🚨 **시장 과열 상태 (고평가)**: 시장 위험이 재무 위험보다 {abs(gap):.0f}bp 낮습니다.")
        else:
            st.info("⚖️ **적정 가치 구간**: 시장 가격이 기업의 재무 상태를 적절히 반영하고 있습니다.")

        st.markdown("---")
        
        st.subheader("🤖 제미나이 AI 최종 투자 의견")
        st.caption("위의 괴리율 데이터와 기업의 펀더멘털을 바탕으로 최종 리포트를 생성합니다.")

        fin_dict = {
            "영업이익": op, "이자비용": ie,
            "이자보상배율(배)": calc_icr, "부채비율(%)": debt
        }
        
        if st.button("📝 투자 의견 리포트 생성하기"):
            with st.spinner("제미나이가 데이터의 괴리율을 분석하여 리포트를 작성 중입니다..."):
                try:
                    report = fundamental_data.generate_investment_report(h_mkt, h_fund, gap, fin_dict)
                    st.markdown("---")
                    st.markdown(report)
                except Exception as e:
                    st.error(f"리포트 생성 중 오류가 발생했습니다: {e}")