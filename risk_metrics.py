def get_fundamental_hazard_rate(debt_ratio, icr):
    """
    현업 스코어카드 방식을 단순화한 재무 부도강도 산출 로직
    """
    # 1. 부채비율 점수 (낮을수록 좋음, 1~10점)
    if debt_ratio < 50: debt_score = 10
    elif debt_ratio < 100: debt_score = 8
    elif debt_ratio < 150: debt_score = 5
    elif debt_ratio < 200: debt_score = 3
    else: debt_score = 1
    
    # 2. 이자보상배율 점수 (높을수록 좋음, 1~10점)
    if icr > 10: icr_score = 10
    elif icr > 5: icr_score = 8
    elif icr > 2: icr_score = 5
    elif icr > 1: icr_score = 3
    else: icr_score = 1
    
    # 3. 가중 합산 (부채비율 40%, 이자보상배율 60%)
    total_score = (debt_score * 0.4) + (icr_score * 0.6)
    
    # 4. 스코어별 부도강도(Hazard Rate) 매핑
    if total_score >= 9: fund_h = 0.001    # 최고 우량 (0.1%)
    elif total_score >= 7: fund_h = 0.005  # 우량 (0.5%)
    elif total_score >= 5: fund_h = 0.015  # 보통 (1.5%)
    elif total_score >= 3: fund_h = 0.040  # 주의 (4.0%)
    else: fund_h = 0.100                   # 위험 (10%)
    
    return fund_h, total_score