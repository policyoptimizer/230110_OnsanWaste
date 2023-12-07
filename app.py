import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

# 데이터 및 모델 불러오기 (위에서 사용한 코드를 여기에 포함시킵니다)
# 예: df = pd.read_csv('file.csv') 등

# 스트림릿 제목 설정
st.title('Pollution Dilution Volume Prediction')

# 사용자 입력 받기
toc = st.number_input('Pollute TOC', min_value=0)
cod = st.number_input('Pollute COD', min_value=0)
tn = st.number_input('Pollute T-N', min_value=0)
vol = st.number_input('Pollute Vol', min_value=0)

# 예측 버튼
if st.button('Predict'):
   model = LinearRegression()
   model.fit(features, target)  # features와 target은 위에서 정의한 데이터셋을 사용합니다.
   predicted_vol = model.predict([[toc, cod, tn, vol]])[0]
   st.write(f'예측된 Dilute Volume: {predicted_vol}')
