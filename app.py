import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# =========================================================
# 1. 딥러닝 모델 아키텍처 정의
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
# 2. 모델 및 스케일러 로딩
# =========================================================
@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DepressionPredictor_PR().to(device)
    try:
        model.load_state_dict(torch.load('202403403_이종원_weight.pt', map_location=device))
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"모델 가중치 로드 실패: {e}")
        return None, None

model, device = load_model()

# 원본 데이터(chs_2025_preprocessed_v3.csv) 기반 정규화 수치
SCALER_MEAN = np.array([
    58.7392, 1.5669, 2.9600, 2.8718, 1.6117, 5.6479, 5.6001, 0.1559, 5.6237, 0.4549, 1.1693, 3.7980,
    1.4909, 0.5695, 0.9978, 2.8303, 1.6804, 1.7893, 3.2761, 2.4442, 1.6433, 1.8357, 1.8626, 3.9603,
    3.4587, 5.0072, 1.3866, 3.3248, 2.0962, 0.8629, 406.1044, 23.8640, 6.3481
])
SCALER_SCALE = np.array([
    17.1968, 0.4954, 1.6927, 0.4849, 0.4873, 2.9566, 3.0220, 0.6171, 2.9898, 3.5323, 0.3750, 2.4073,
    1.6518, 1.3208, 1.6921, 2.0623, 1.4508, 1.2081, 0.9041, 1.3330, 0.4790, 0.3704, 0.3442, 1.8132,
    1.8905, 1.7028, 0.4869, 3.3128, 1.5224, 1.2379, 334.1516, 3.9632, 1.4844
])

# =========================================================
# 3. Streamlit UI 및 33개 변수 설문 폼 구성
# =========================================================
st.set_page_config(page_title="AI 우울증 스크리닝", page_icon="🧠", layout="centered")

st.title("🧠 AI 기반 우울증 위험군 스크리닝")
st.markdown("지역사회건강조사(CHS) 기반의 33가지 일상 습관 및 심리 데이터를 분석하여 잠재적 우울증 위험도를 판별합니다.")
st.divider()

# UI 헬퍼 함수
def make_sel(label, opt_dict):
    return opt_dict[st.selectbox(label, list(opt_dict.keys()))]

with st.form("survey_form"):
    tab1, tab2, tab3, tab4 = st.tabs(["👤 기본 인적사항", "🏥 신체 건강", "🏃 생활/식습관", "🚬 흡연/음주"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("나이 (만)", value=40, min_value=19, max_value=100)
            sex = make_sel("성별", {"남자": 1.0, "여자": 2.0})
            Monthly_Income = st.number_input("가구 월 평균 소득 (만원)", value=300, min_value=0, step=50)
            sod_02z3 = make_sel("현재 혼인 상태", {"유배우(동거/별거)": 1.0, "사별": 2.0, "이혼": 3.0, "미혼": 4.0})
        with col2:
            fma_19z3 = make_sel("세대 유형", {"1인 가구": 1.0, "부부 등 2세대": 2.0, "3세대 이상": 3.0, "기타": 4.0})
            fma_04z1 = make_sel("기초생활수급 경험", {"현재 수급자": 1.0, "과거 경험 있음": 2.0, "경험 없음": 3.0})
            sob_01z1 = make_sel("최종 학력", {"무학/서당": 1.0, "초졸": 2.0, "중졸": 3.0, "고졸": 4.0, "전문대졸": 5.0, "대졸 이상": 6.0})
            soa_01z1 = make_sel("현재 경제활동 여부", {"예": 1.0, "아니오": 2.0})
            soa_06z2 = make_sel("직업군", {"전문/행정/관리직": 1.0, "사무직": 2.0, "서비스/판매직": 3.0, "농림어업숙련직": 4.0, "기능/기계조작직": 5.0, "단순노무직": 6.0, "무직/학생/주부": 7.0, "기타": 8.0})

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            BMI = st.number_input("체질량지수 (BMI)", value=23.0, step=0.1)
            oba_01z1 = make_sel("본인의 체형에 대한 인식", {"매우 마른 편": 1.0, "약간 마른 편": 2.0, "보통": 3.0, "약간 비만": 4.0, "매우 비만": 5.0})
            obb_01z1 = make_sel("최근 1년간 체중조절 노력", {"체중 감소 노력": 1.0, "체중 유지 노력": 2.0, "체중 증가 노력": 3.0, "노력 안 함": 4.0})
        with col2:
            hya_04z1 = make_sel("고혈압 의사 진단 여부", {"진단 받은 적 있음": 1.0, "없음": 2.0})
            dia_04z1 = make_sel("당뇨병 의사 진단 여부", {"진단 받은 적 있음": 1.0, "없음": 2.0})
            osa_04z1 = make_sel("골다공증 의사 진단 여부", {"진단 받은 적 있음": 1.0, "없음": 2.0})

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            Avg_Sleep = st.number_input("하루 평균 수면 시간 (시간)", value=7.0, step=0.5)
            nua_01z2 = make_sel("일주일 평균 아침식사 일수", {"주 5~7일": 1.0, "주 3~4일": 2.0, "주 1~2일": 3.0, "거의 안 먹음": 4.0})
            
            # 피드백 반영: 이웃 변수 삭제 및 가족/친구 변수명 정확히 매핑
            enb_01z1 = make_sel("가족/친척과의 접촉 빈도", {"한 달 1번 미만": 1.0, "한 달 1번": 2.0, "한 달 2~3번": 3.0, "주 1번": 4.0, "주 2~3번": 5.0, "주 4번 이상": 6.0})
            enb_03z1 = make_sel("친구와의 접촉 빈도", {"한 달 1번 미만": 1.0, "한 달 1번": 2.0, "한 달 2~3번": 3.0, "주 1번": 4.0, "주 2~3번": 5.0, "주 4번 이상": 6.0})
        
        with col2:
            st.markdown("**주간 신체활동 일수 (0~7일)**")
            # 피드백 반영: step=1.0 으로 지정하여 1일 단위로만 움직이도록 수정
            pha_04z1 = st.number_input("숨이 많이 가쁜 격렬한 활동", min_value=0.0, max_value=7.0, value=0.0, step=1.0)
            pha_07z1 = st.number_input("숨이 약간 가쁜 중등도 활동", min_value=0.0, max_value=7.0, value=0.0, step=1.0)
            phb_01z1 = st.number_input("10분 이상 걷기 실천", min_value=0.0, max_value=7.0, value=3.0, step=1.0)
            pha_19z1 = st.number_input("근력 운동 실천", min_value=0.0, max_value=7.0, value=0.0, step=1.0)

    with tab4:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**담배 (흡연)**")
            sma_01z1 = make_sel("평생 일반담배 흡연(5갑 이상) 여부", {"예": 1.0, "아니오": 2.0})
            
            # 피드백 반영: 평생 흡연 경험 없으면 하위 질문 숨김 및 자동 0 처리
            if sma_01z1 == 1.0:
                smf_01z1 = make_sel("현재 일반담배 흡연 여부", {"매일 피움": 1.0, "가끔 피움": 2.0, "과거에 피웠으나 현재 안 피움": 3.0})
                if smf_01z1 in [1.0, 2.0]: # 현재 흡연자만 흡연량 입력
                    sma_03z1 = st.number_input("하루 평균 일반담배 흡연량 (개비)", value=0.0, step=1.0)
                else: # 과거 흡연자는 흡연량 0으로 자동 세팅
                    sma_03z1 = 0.0
            else: # 평생 안 피운 사람은 항목 숨기고 자동 세팅
                smf_01z1 = 4.0 # 피운 적 없음 코드
                sma_03z1 = 0.0

            st.caption("전자담배 사용 이력")
            sma_36z1 = make_sel("궐련형 전자담배 평생 사용", {"사용해본 적 있음": 1.0, "없음": 2.0})
            if sma_36z1 == 1.0:
                sma_37z1 = make_sel("궐련형 전자담배 현재 사용", {"사용함": 1.0, "안함": 2.0})
            else:
                sma_37z1 = 8.0 # 해당 없음 코드

            sma_08z1 = make_sel("액상형 전자담배 평생 사용", {"사용해본 적 있음": 1.0, "없음": 2.0})
            if sma_08z1 == 1.0:
                sma_11z2 = st.number_input("최근 1달 액상형 전자담배 사용 일수", value=0.0, max_value=31.0, step=1.0)
            else:
                sma_11z2 = 0.0 # 사용 안 했으므로 0일

        with col2:
            st.markdown("**주류 (음주)**")
            dra_01z1 = make_sel("평생 1잔 이상 음주 여부", {"예": 1.0, "아니오": 2.0})
            
            # 피드백 반영: 평생 음주 경험 없으면 하위 질문 숨김 및 자동 '해당 없음' 처리
            if dra_01z1 == 1.0:
                drb_01z3 = make_sel("최근 1년간 음주 빈도", {"전혀 안 마심": 1.0, "월 1회 미만": 2.0, "월 1회 정도": 3.0, "월 2~4회": 4.0, "주 2~3회": 5.0, "주 4회 이상": 6.0})
                if drb_01z3 == 1.0: # 1년간 안 마심
                    drb_03z1 = 8.0 # 해당 없음
                else:
                    drb_03z1 = make_sel("1회 평균 음주량", {"1~2잔": 1.0, "3~4잔": 2.0, "5~6잔": 3.0, "7~9잔": 4.0, "10잔 이상": 5.0})
            else:
                drb_01z3 = 8.0 # 해당 없음 코드
                drb_03z1 = 8.0

    submit_button = st.form_submit_button("진단 결과 확인하기", use_container_width=True)

# =========================================================
# 4. 추론 및 결과 출력 로직
# =========================================================
if submit_button and model is not None:
    X_raw = np.array([
        age, sex, fma_19z3, fma_04z1, smf_01z1, sma_01z1, sma_36z1, sma_37z1, 
        sma_08z1, sma_11z2, dra_01z1, drb_01z3, drb_03z1, pha_04z1, pha_07z1, 
        phb_01z1, pha_19z1, nua_01z2, oba_01z1, obb_01z1, hya_04z1, dia_04z1, 
        osa_04z1, enb_01z1, enb_03z1, sob_01z1, soa_01z1, soa_06z2, sod_02z3, 
        sma_03z1, Monthly_Income, BMI, Avg_Sleep
    ])
    
    X_scaled = (X_raw - SCALER_MEAN) / SCALER_SCALE
    
    with torch.no_grad():
        inputs = torch.FloatTensor(X_scaled).unsqueeze(0).to(device)
        probs = F.softmax(model(inputs), dim=1).cpu().numpy()[0]
    
    OPTIMAL_THRESHOLD = 0.3757
    prob_high_risk = probs[2]
    
    if prob_high_risk >= OPTIMAL_THRESHOLD:
        final_pred = 2  # 위험군
    else:
        final_pred = np.argmax(probs[:2])  # 정상(0) or 경도(1)

    st.divider()
    st.subheader("📊 AI 스크리닝 결과")
    
    if final_pred == 0:
        st.success("🟢 **정상 (Normal)** : 현재 안정적인 심리 상태를 유지하고 있습니다.")
    elif final_pred == 1:
        st.warning("🟡 **경도 (Mild)** : 가벼운 스트레스나 우울감이 관찰됩니다. 휴식이 필요합니다.")
    elif final_pred == 2:
        st.error(f"🔴 **위험군 (High Risk)** : 잠재적 우울증 위험이 감지되었습니다. (위험 확률: {prob_high_risk:.1%}) 전문 상담을 권장합니다.")
        
    st.markdown("**클래스별 세부 예측 확률**")
    col1, col2, col3 = st.columns(3)
    col1.metric("정상 확률", f"{probs[0]:.1%}")
    col2.metric("경도 확률", f"{probs[1]:.1%}")
    col3.metric("위험군 확률", f"{probs[2]:.1%}")
    st.caption(f"*적용된 보수적 스크리닝 임계값: {OPTIMAL_THRESHOLD}*")
