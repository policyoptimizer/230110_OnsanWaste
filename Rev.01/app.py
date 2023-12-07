import streamlit as st
import pandas as pd
import requests
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# 제목
st.title('정제수 투입량 예측 애플리케이션')

# 구글 시트 데이터 로드
@st.cache
def load_data():
   file_id = "1h2n63h2EDSA6207eF15Gk2SjsN8qjI9TrWclCMQF6X0"
   sheet_name = "Sheet1"
   url = f"https://docs.google.com/spreadsheets/d/{file_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
   res = requests.get(url)
   df = pd.read_csv(pd.compat.StringIO(res.text))
   # 데이터 처리...
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
   # 모델 훈련 및 예측 로직
   # ...

# Streamlit 앱 실행: 

streamlit run app.py