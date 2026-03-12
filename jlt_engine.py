import numpy as np
import pandas as pd
from scipy.optimize import minimize

def calculate_discount_factor(rate_pct, t):
    """연속 복리 기반 무위험 할인율(Discount Factor) 계산"""
    return np.exp(-(rate_pct / 100) * t)

def calculate_jlt_price(hazard_rate, face_value, coupon_rate, maturity, rf_rate, recovery_rate):
    """
    주어진 부도강도(Hazard Rate)를 바탕으로 채권의 이론가를 산출합니다.
    - 생존 시: 이자 + 원금 수령
    - 부도 시: 회수율(Recovery Rate)만큼 수령
    """
    times = np.arange(1, maturity + 1)
    price = 0
    sp_prev = 1.0 # 0년차 생존확률은 100%

    for t in times:
        df = calculate_discount_factor(rf_rate, t)
        
        # t년 시점의 생존 확률 (지수 분포 가정)
        sp_t = np.exp(-hazard_rate * t)
        
        # (t-1)년과 t년 사이에 부도가 발생할 한계 확률
        default_prob = sp_prev - sp_t
        
        # 1. 생존 시 현금흐름의 현재가치 (이표 지급)
        cf_survive = (face_value * coupon_rate / 100)
        if t == maturity:
            cf_survive += face_value # 만기 시 원금 추가
            
        price += cf_survive * df * sp_t
        
        # 2. 부도 시 현금흐름의 현재가치 (회수액)
        price += face_value * recovery_rate * df * default_prob
        
        sp_prev = sp_t
        
    return price

def calibrate_hazard_rate(market_price, face_value, coupon_rate, maturity, rf_rate, recovery_rate=0.4):
    """
    [핵심 알고리즘] 시장가(Market Price)와 모델 이론가(Model Price)의 오차가 
    0이 되도록 만드는 최적의 부도강도(Hazard Rate)를 역산(Calibration)합니다.
    """
    # 목적 함수: (모델 가격 - 실제 시장 가격)^2 을 최소화
    def objective_function(h):
        model_price = calculate_jlt_price(h[0], face_value, coupon_rate, maturity, rf_rate, recovery_rate)
        return (model_price - market_price)**2
    
    # 부도강도 초기 추정치 및 최적화 실행 (L-BFGS-B 알고리즘 사용)
    initial_guess = [0.01]
    result = minimize(objective_function, initial_guess, bounds=[(0.00001, 1.0)])
    
    calibrated_hazard_rate = result.x[0]
    
    # 캘리브레이션 된 부도강도로 연도별 생존확률 커브 생성
    times = np.arange(1, maturity + 1)
    survival_probs = [np.exp(-calibrated_hazard_rate * t) for t in times]
    
    return calibrated_hazard_rate, survival_probs