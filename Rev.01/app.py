import streamlit as st
import pandas as pd
import requests
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import io

# 제목
st.title('오염 농도 예측 애플리케이션')

# 구글 시트 데이터 로드
@st.cache(allow_output_mutation+True)
def load_data():
   file_id = "1h2n63h2EDSA6207eF15Gk2SjsN8qjI9TrWclCMQF6X0"
   sheet_name = "Sheet1"
   url = f"https://docs.google.com/spreadsheets/d/{file_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
   res = requests.get(url)
   # df = pd.read_csv(pd.compat.StringIO(res.text))
   df = pd.read_csv(io.StringIO(res.text))
   # 데이터 처리 로직...
   return df

df = load_data()

# 데이터 표시 (옵션)
if st.checkbox('데이터 표시'):
   st.write(df.head())

# 예측을 위한 사용자 입력
TOC = st.number_input('TOC 오염도', min_value=0)
COD = st.number_input('COD 오염도', min_value=0)
TN = st.number_input('T-N 오염도', min_value=0)
Volume = st.number_input('오염물질 부피', min_value=0)

# 예측 버튼
if st.button('예측'):
   try:
       # 데이터 전처리 및 모델 훈련 로직
       df.replace(to_replace=',', value='', regex=True, inplace=True)
       df.replace('-', np.nan, inplace=True)
       df.fillna(0, inplace=True)
       df['Date'] = pd.to_datetime(df['Date'])

       # 특성과 타겟 변수 선택
       features = df[['pollute TOC', 'pollute COD', 'pollute T-N', 'pollute vol']]
       target = df['dilute vol']

       # 데이터 분할: 훈련 세트와 테스트 세트
       X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

       # 선형 회귀 모델 생성 및 훈련
       model = LinearRegression()
       model.fit(X_train, y_train)

       # 사용자 입력으로 새 데이터 생성
       new_data = pd.DataFrame([[TOC, COD, TN, Volume]], columns=['pollute TOC', 'pollute COD', 'pollute T-N', 'pollute vol'])
       
       # 예측
       predicted_dilute_vol = model.predict(new_data)
       st.write(f"예측된 dilute vol: {predicted_dilute_vol[0]}")
   except Exception as e:
       st.error(f"에러 발생: {e}")

# Streamlit 앱 실행: streamlit run app.py