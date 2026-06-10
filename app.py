import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# =========================================================
# 1. 딥러닝 모델 아키텍처 정의 (기존과 완벽히 동일)
# =========================================================
class DepressionPredictor_PR(nn.Module):
    def __init__(self):
        super(DepressionPredictor_PR, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(33, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 3) 
        )
    def forward(self, x):
        return self.model(x)

# =========================================================
# 2. 모델 및 스케일러 로딩 (캐싱 적용)
# =========================================================
# st.cache_resource를 사용해 새로고침 시 모델을 매번 다시 불러오는 것을 방지
@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DepressionPredictor_PR().to(device)
    
    # 가중치 파일 로드
    try:
        model.load_state_dict(torch.load('202403403_이종원_weight.pt', map_location=device))
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"모델 가중치 로드 실패: {e}")
        return None, None

model, device = load_model()

# [주의] 이전에 주피터 노트북에서 추출했던 33개의 평균/표준편차 값을 여기에 넣어야 해!
SCALER_MEAN = np.zeros(33)  # 실제 값으로 교체 필요
SCALER_SCALE = np.ones(33)  # 실제 값으로 교체 필요

# =========================================================
# 3. Streamlit UI 구성
# =========================================================
st.set_page_config(page_title="AI 우울증 스크리닝", page_icon="🧠", layout="centered")

st.title("🧠 AI 기반 우울증 위험군 스크리닝")
st.markdown("""
현대인의 33가지 일상 습관 및 심리 데이터를 분석하여 잠재적 우울증 위험도를 판별합니다.  
**본 시스템은 고위험군 누락을 원천 차단하도록 의료 도메인 지식이 반영되어 있습니다.**
""")

st.divider()

# 설문조사 폼 시작
with st.form("survey_form"):
    st.subheader("📝 일상 데이터 입력")
    
    # 33개의 변수 입력을 깔끔하게 받기 위해 3단 컬럼 사용
    cols = st.columns(3)
    user_inputs = []
    
    # 예시: 33개의 입력 필드 생성 (실제 변수명에 맞게 텍스트 수정 필요)
    feature_names = [
        "수면 시간 (시간)", "아침 식사 빈도 (주 N회)", "주간 운동량 (일)", "스트레스 지수 (1~10)", 
        "경제활동 여부 (0:무, 1:유)" # ... 나머지 28개 변수명 추가
    ]
    
    # 변수명이 부족한 부분을 임시로 채워줌
    while len(feature_names) < 33:
        feature_names.append(f"설문 항목 {len(feature_names) + 1}")

    # UI에 33개 입력창 렌더링
    for i, name in enumerate(feature_names):
        with cols[i % 3]:
            # 데이터 성격에 따라 number_input, selectbox 등을 유동적으로 사용할 수 있음
            val = st.number_input(name, value=0.0, step=1.0, key=f"input_{i}")
            user_inputs.append(val)
            
    submit_button = st.form_submit_button("진단 결과 확인하기", use_container_width=True)

# =========================================================
# 4. 추론 및 결과 출력 로직
# =========================================================
if submit_button and model is not None:
    # 1. 데이터 전처리 (자체 정규화)
    X_raw = np.array(user_inputs)
    X_scaled = (X_raw - SCALER_MEAN) / SCALER_SCALE
    
    # 2. 딥러닝 추론
    with torch.no_grad():
        inputs = torch.FloatTensor(X_scaled).unsqueeze(0).to(device) # 배치 차원 추가
        probs = F.softmax(model(inputs), dim=1).cpu().numpy()[0]
    
    # 3. 임계값(0.3757) 적용 (프로젝트 핵심 로직)
    OPTIMAL_THRESHOLD = 0.3757
    prob_high_risk = probs[2]
    
    if prob_high_risk >= OPTIMAL_THRESHOLD:
        final_pred = 2  # 위험군
    else:
        final_pred = np.argmax(probs[:2])  # 정상(0) or 경도(1)

    # 4. 결과 시각화
    st.divider()
    st.subheader("📊 AI 스크리닝 결과")
    
    if final_pred == 0:
        st.success("🟢 **정상 (Normal)** : 현재 안정적인 심리 상태를 유지하고 있습니다.")
    elif final_pred == 1:
        st.warning("🟡 **경도 (Mild)** : 가벼운 스트레스나 우울감이 관찰됩니다. 휴식이 필요합니다.")
    elif final_pred == 2:
        st.error(f"🔴 **위험군 (High Risk)** : 잠재적 우울증 위험이 감지되었습니다. (확률: {prob_high_risk:.1%}) 전문 상담을 권장합니다.")
        
    # 세부 확률 데이터 시각화 (진단 신뢰도 제공)
    st.markdown("**클래스별 세부 예측 확률**")
    col1, col2, col3 = st.columns(3)
    col1.metric("정상 확률", f"{probs[0]:.1%}")
    col2.metric("경도 확률", f"{probs[1]:.1%}")
    col3.metric("위험군 확률", f"{probs[2]:.1%}")
    
    st.caption(f"*적용된 위험군 판별 임계값: {OPTIMAL_THRESHOLD}*")