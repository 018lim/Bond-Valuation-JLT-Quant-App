import pandas as pd
import numpy as np
import streamlit as st
import os

@st.cache_data(ttl=3600)
def load_kap_data(file_source):
    """내장된 샘플 경로(str) 또는 업로드된 파일(객체)을 모두 처리하여 읽어옵니다. (CSV, XLSX 지원)"""
    if file_source is None:
        return None
    
    try:
        # 파일 이름이나 객체의 이름을 통해 확장자 파악
        file_name = file_source if isinstance(file_source, str) else file_source.name
        
        if file_name.endswith('.csv'):
            try:
                df = pd.read_csv(file_source, encoding='cp949')
            except UnicodeDecodeError:
                df = pd.read_csv(file_source, encoding='utf-8')
        elif file_name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file_source)
        else:
            st.warning("⚠️ 지원하지 않는 파일 형식입니다. (csv, xlsx만 가능)")
            return None
            
    except Exception as e:
        print(f"파일 읽기 에러: {e}")
        return None
        
    df.columns = df.columns.astype(str).str.replace('\n', '').str.strip()
    return df

def get_yield_curve(file_source):
    df = load_kap_data(file_source)
    
    y1, y2, y3, y5, y10 = 3.10, 3.15, 3.20, 3.25, 3.35 
    
    if df is not None:
        try:
            mask = df.apply(lambda row: row.astype(str).str.contains('국고채').any(), axis=1)
            gov_row = df[mask].iloc[0]
            y1, y2, y3, y5, y10 = float(gov_row['1년']), float(gov_row['2년']), float(gov_row['3년']), float(gov_row['5년']), float(gov_row['10년'])
        except Exception:
            pass

    data = {
        'Maturity(Year)': [1, 2, 3, 5, 10],
        'RiskFreeRate(%)': [y1, y2, y3, y5, y10]
    }
    return pd.DataFrame(data)

def get_credit_spread(file_source):
    df = load_kap_data(file_source)
    
    sp1_aa, sp2_aa, sp3_aa, sp5_aa = 40.0, 50.0, 60.0, 70.0
    sp1_bbb, sp2_bbb, sp3_bbb, sp5_bbb = 350.0, 380.0, 430.0, 500.0
    
    if df is not None:
        try:
            mask_gov = df.apply(lambda row: row.astype(str).str.contains('국고채').any(), axis=1)
            gov_row = df[mask_gov].iloc[0]
            g1, g2, g3, g5 = float(gov_row['1년']), float(gov_row['2년']), float(gov_row['3년']), float(gov_row['5년'])
            
            mask_aa = df.apply(lambda row: row.astype(str).str.contains('AA-').any() and row.astype(str).str.contains('무보증').any(), axis=1)
            aa_row = df[mask_aa].iloc[0]
            aa1, aa2, aa3, aa5 = float(aa_row['1년']), float(aa_row['2년']), float(aa_row['3년']), float(aa_row['5년'])
            
            sp1_aa, sp2_aa, sp3_aa, sp5_aa = (aa1 - g1) * 100, (aa2 - g2) * 100, (aa3 - g3) * 100, (aa5 - g5) * 100
            
            try:
                mask_bbb = df.apply(lambda row: row.astype(str).str.contains('BBB-').any() and row.astype(str).str.contains('무보증').any(), axis=1)
                bbb_row = df[mask_bbb].iloc[0]
                bbb1, bbb2, bbb3, bbb5 = float(bbb_row['1년']), float(bbb_row['2년']), float(bbb_row['3년']), float(bbb_row['5년'])
                sp1_bbb, sp2_bbb, sp3_bbb, sp5_bbb = (bbb1 - g1) * 100, (bbb2 - g2) * 100, (bbb3 - g3) * 100, (bbb5 - g5) * 100
            except Exception:
                sp1_bbb, sp2_bbb, sp3_bbb, sp5_bbb = sp1_aa + 300, sp2_aa + 320, sp3_aa + 350, sp5_aa + 400
        except Exception:
            pass

    data = {
        'Maturity(Year)': [1, 2, 3, 5],
        'AA- (bp)': [round(sp1_aa), round(sp2_aa), round(sp3_aa), round(sp5_aa)],
        'BBB- (bp)': [round(sp1_bbb), round(sp2_bbb), round(sp3_bbb), round(sp5_bbb)]
    }
    return pd.DataFrame(data)