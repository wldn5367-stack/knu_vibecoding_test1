import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# 한글 폰트 설정 (환경에 따라 'Malgun Gothic' 또는 'AppleGothic' 사용)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 페이지 설정
st.set_page_config(page_title="타이타닉 생존 예측", layout="wide")

st.title("🚢 타이타닉 생존자 예측 시스템")
st.markdown("---")

# 사이드바 메뉴
menu = st.sidebar.radio("메뉴 선택", ["데이터 로드", "데이터 분석 (EDA)", "모델 학습", "생존 예측"])

# 1. 데이터 로드 섹션
if menu == "데이터 로드":
    st.header("📂 데이터 업로드 및 확인")
    uploaded_file = st.sidebar.file_uploader("titanic.csv 파일을 업로드하세요", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state["titanic_data"] = df
        
        col1, col2 = st.columns([1, 1])
        with col1:
            with st.container(border=True):
                st.subheader("데이터 미리보기")
                st.write(df.head(10))
        with col2:
            with st.container(border=True):
                st.subheader("데이터 정보")
                st.write(f"전체 데이터 수: {len(df)}")
                st.write(f"컬럼 수: {len(df.columns)}")
                st.write(df.dtypes)
    else:
        st.info("왼쪽 사이드바에서 타이타닉 CSV 파일을 업로드해주세요.")

# 2. EDA 섹션
elif menu == "데이터 분석 (EDA)":
    if "titanic_data" in st.session_state:
        df = st.session_state["titanic_data"]
        st.header("📊 데이터 시각화 분석")

        col1, col2 = st.columns(2)

        with col1:
            with st.container(border=True):
                st.subheader("객실 등급별 생존 분포")
                fig, ax = plt.subplots()
                sns.countplot(x='Pclass', hue='Survived', data=df, palette='viridis', ax=ax)
                st.pyplot(fig)

        with col2:
            with st.container(border=True):
                st.subheader("성별에 따른 생존 분포")
                fig, ax = plt.subplots()
                sns.countplot(x='Sex', hue='Survived', data=df, palette='magma', ax=ax)
                st.pyplot(fig)
    else:
        st.warning("먼저 데이터를 업로드해주세요.")

# 3. 모델 학습 섹션
elif menu == "모델 학습":
    if "titanic_data" in st.session_state:
        df = st.session_state["titanic_data"].copy()
        st.header("⚙️ 모델 학습 및 평가")

        # 전처리 (결측치 처리 및 불필요한 컬럼 삭제)
        df['Age'] = df['Age'].fillna(df['Age'].median())
        df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
        df['Fare'] = df['Fare'].fillna(df['Fare'].median())
        df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])

        # 인코딩
        le = LabelEncoder()
        df['Sex'] = le.fit_transform(df['Sex'])
        df['Embarked'] = le.fit_transform(df['Embarked'])

        X = df.drop(columns=['Survived'])
        y = df['Survived']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 모델 학습
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # 결과 출력
        c1, c2 = st.columns(2)
        with c1:
            with st.container(border=True):
                st.subheader("모델 정확도")
                st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2%}")
                st.text("분류 보고서:")
                st.text(classification_report(y_test, y_pred))

        with c2:
            with st.container(border=True):
                st.subheader("특징 중요도 (Feature Importance)")
                feat_importances = pd.Series(model.feature_importances_, index=X.columns)
                fig, ax = plt.subplots()
                feat_importances.nlargest(10).plot(kind='barh', ax=ax)
                st.pyplot(fig)

        st.session_state["titanic_model"] = model
        st.session_state["features"] = X.columns
    else:
        st.warning("먼저 데이터를 업로드해주세요.")

# 4. 예측 섹션
elif menu == "생존 예측":
    if "titanic_model" in st.session_state:
        st.header("🔮 새로운 승객 생존 예측")
        
        with st.container(border=True):
            st.write("승객 정보를 입력하세요:")
            # 입력란을 3개의 컬럼으로 구분
            row1_col1, row1_col2, row1_col3 = st.columns(3)
            row2_col1, row2_col2, row2_col3 = st.columns(3)

            with row1_col1:
                pclass = st.selectbox("객실 등급 (Pclass)", [1, 2, 3])
            with row1_col2:
                sex = st.selectbox("성별 (Sex)", ["male", "female"])
            with row1_col3:
                age = st.number_input("나이 (Age)", min_value=0, max_value=100, value=30)

            with row2_col1:
                sibsp = st.number_input("형제/배우자 수 (SibSp)", 0, 10, 0)
            with row2_col2:
                parch = st.number_input("부모/자녀 수 (Parch)", 0, 10, 0)
            with row2_col3:
                fare = st.number_input("요금 (Fare)", 0.0, 500.0, 32.0)

            embarked = st.selectbox("탑승 항구 (Embarked)", ["S", "C", "Q"])

        if st.button("결과 확인하기", use_container_width=True):
            # 입력 데이터 변환
            sex_val = 1 if sex == "male" else 0
            emb_map = {"S": 2, "C": 0, "Q": 1}
            emb_val = emb_map[embarked]
            
            input_data = [[pclass, sex_val, age, sibsp, parch, fare, emb_val]]
            prediction = st.session_state["titanic_model"].predict(input_data)
            prob = st.session_state["titanic_model"].predict_proba(input_data)

            st.markdown("---")
            if prediction[0] == 1:
                st.success(f" 결과: 생존 가능성이 높습니다! (확률: {prob[0][1]:.2%})")
            else:
                st.error(f" 결과: 사망 가능성이 높습니다. (확률: {prob[0][0]:.2%})")
    else:
        st.warning("모델 학습을 먼저 완료해주세요.")