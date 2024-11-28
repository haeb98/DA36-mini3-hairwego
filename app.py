import streamlit as st
import os
import numpy as np
from PIL import Image

import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# 저장 경로 설정
SAVE_DIR = "upload"
os.makedirs(SAVE_DIR, exist_ok=True)  # 해당폴더가 있는경우 오류발생 억제

# 배경색 설정
main_bg_color = "#e2e4f8"  # 메인 페이지 배경색

# CSS 스타일을 적용하여 배경 색 변경
st.markdown(f"""
            <style>
            /* 메인 페이지 배경 색 설정 */
            .stApp {{
                background-color: {main_bg_color};
            }}
            </style>
            """, unsafe_allow_html=True)

# 페이지 제목
st.title("Hair We Go!")
st.title("탈모 자가진단 서비스")
st.subheader("두피 건강 상태를 확인해보세요")
st.markdown("")
# 1st 섹션
col1, col2 = st.columns([1, 2])
with col1:
    st.image("streamlit_images/hairwego_character.png", width = 220) # 가상 유저 이미지

with col2:
    st.markdown(
    """
    #### 🔎 탈모 의심 체크리스트 
    ###### 하나라도 해당된다면 지금 탈모 진단이 필요합니다! 
        ☑️ 하루에 빠지는 머리카락이 100개가 넘을 때  
        ☑️ 머리를 감거나 빗질할 때 빠지는 머리카락 수가 증가할 때  
        ☑️ 머리카락 굵기가 점점 가늘게 느껴질 때  
        ☑️ 머리카락 사이로 두피가 보일 때
    """)
st.markdown("---")

# 상단 큰 블록 (설명 문구 + 이미지 업로드)
with st.container():
    # 설명 문구
    st.markdown(
        """
        <div style="background-color: #fafafa; padding: 20px; border-radius: 10px;">
            <h2 style="text-align: center;">탈모 자가진단</h2>
            <p style="text-align: center;">
                두피 이미지를 업로드하면, <br>
                딥러닝 모델이 탈모 진행 단계를 알려드립니다.
            </p>
            <h5 style="text-align: center;">양호 ➡ 경증 ➡ 중등도 ➡ 중증</h5>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")

# 이미지 업로드 기능
with st.container():
    # 설명 - 이미지를 업로드하세요
    st.markdown(
        """
        <div style="background-color: #fcf4e8; padding: 20px; border-radius: 10px;">
            <h4 style="text-align: center;">⬇️하단에 이미지 파일을 업로드하세요⬇️</h4>
        </div>
        """,
        unsafe_allow_html=True
    )


# 이미지 업로드 기능 컨테이너
# - uploaded_file은 UploadedFile 객체이다.
# - Streamlit에서 제공하는 파일 업로드를 처리하기 위한 특수 객체로, Python의 io.BytesIO와 유사하다.
uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])
st.markdown("</div>", unsafe_allow_html=True)  # 닫는 태그

# 파일 정보 확인 토글 (Expander 사용)
if uploaded_file is not None:
    with st.expander("업로드 이미지 정보 확인"):
        st.write("파일 이름:", uploaded_file.name)
        st.write("파일 타입:", uploaded_file.type)
        st.write("파일 크기:", uploaded_file.size, "bytes")
        # 업로드된 이미지 표시
        # - 이미지경로, url, PIL Image, ndarray, List[Image], List[ndarray], UploadedFile를 지원한다.
        st.image(uploaded_file, width = 220, caption="업로드 이미지")

    # 모델 로드 및 예측
    filepath = 'MobileNetV2.loss-0.60-accuracy-0.81.h5'
    model = load_model(filepath)
    print(model)

    class_names = [
        "탈모 상태 양호",
        "탈모 진행 단계: 경증",
        "탈모 진행 단계: 중등도",
        "탈모 진행 단계: 중증",
    ]

    # 맞춤형 탈모 샴푸 추천 (이미지 경로)
    shampoo_image_paths = {
        "0": "streamlit_images/shampoo_0.png",
        "1": "streamlit_images/shampoo_1.png",
        "2": "streamlit_images/shampoo_2.png",
        "3": "streamlit_images/shampoo_3.png",
    }

    IMAGE_SIZE= 224

    # 이미지 준비
    image = Image.open(uploaded_file) # <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=1500x1500 at 0x198B95F8250>
    image_np = np.array(image)

    # cv2 대신 tf.image.resize 사용
    resized_image = tf.image.resize(image_np, (IMAGE_SIZE, IMAGE_SIZE))
    print('resized_image', type(resized_image), resized_image.shape) # <class 'tensorflow.python.framework.ops.EagerTensor'>  (224, 224, 3)
    # EagerTensor 타입을 NumPy 배열로 다시 변환
    a_image = np.array(resized_image)
    # MobileNetV2 전용 스케일링
    a_image = preprocess_input(a_image)
    batch_image = a_image.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 3)

    # 단일 버튼으로 변경
    if st.button("탈모 진행 상태 확인하기 🔎", help="모발 상태를 예측합니다."):
        # 예측 모델 로드 및 결과 출력 코드
        pred_proba = model.predict(batch_image)
        pred = np.argmax(pred_proba)
        pred_label = class_names[pred]
        pred_probability = pred_proba[0][pred]

        st.markdown(
            f"""
            <div style="background-color: #ffffff; padding: 20px; border-radius: 10px; margin-top: 20px;">
                <h3 style="text-align: center; color: #333;">🔹 모발 상태 예측 결과 🔹</h3>
                <div style="background-color: #aeb4f5; padding: 15px; border-radius: 10px; margin-bottom: 10px;">
                    <h4 style="color: #1c1d29;">예측: {pred_label}</h4>
                </div>
                <div style="background-color: #f9fcbb; padding: 15px; border-radius: 10px;">
                    <h4 style="color: #191a11;">예측 확률: {pred_probability * 100:.2f}%</h4>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # 샴푸 이미지 출력
        shampoo_image_html = f"""
        <div style="background-color: #ffffff; padding: 20px; border-radius: 15px; text-align: center; margin-top: 20px;">
            <h4 style="color: #333; font-size: 22px;">🧴 Hair We Go 맞춤형 탈모 샴푸 추천 🧴</h4>
        </div>
        """
        # 제목 출력
        st.markdown(shampoo_image_html, unsafe_allow_html=True)

        # 경로 확인: shampoo_image_paths[str(pred)]가 올바른 경로를 참조하는지 확인
        image_path = shampoo_image_paths[str(pred)]

        # 3개의 컬럼으로 나누기
        col1, col2, col3 = st.columns([1, 3, 1])  # 가운데 컬럼을 더 넓게 설정

        # 중간 컬럼에 이미지 넣기
        with col2:
            st.image(image_path, width=550)  # 이미지 크기 조정

        # 구매 링크 버튼
        # 3개의 컬럼으로 나누기 (중앙에 배치하기 위해)
        col1, col2, col3 = st.columns([3, 2, 3])  # 가운데 컬럼을 더 넓게 설정

        # 중간 컬럼에 버튼 넣기
        with col2:
            # HTML 버튼으로 새 탭에서 페이지 이동
            st.markdown(
                """
                <div style="text-align: center;">
                    <a href="https://www.oliveyoung.co.kr/store/planshop/getPlanShopDetail.do?dispCatNo=500000102250043&trackingCd=Home_Catchkeyword" 
                       target="_blank" 
                       style="display: inline-block; background-color: #f9fcbb; color: black; padding: 10px 20px; text-decoration: none; 
                              border-radius: 8px; font-size: 18px; font-weight: bold;">
                        지금 구매하기 🏃🏻‍♀️➡️
                    </a>
                </div>
                <br>
                """,
                unsafe_allow_html=True
            )

    # 서버에 저장
    save_path = os.path.join(SAVE_DIR, uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"이미지가 성공적으로 저장되었습니다: {save_path}")


# 데이터 정보 확인
st.markdown("---")


# 탈모 건강 상식
st.markdown("### 📚 탈모 건강 상식")
# 블록 1: 좌측 - 이미지, 우측 - 설명
col1, col2 = st.columns([1, 2])  # 왼쪽 열은 1배, 오른쪽 열은 2배 크기

# 블록 1 내용
with col1:
    st.image("streamlit_images/hair_tip_1.jpg", caption="Tip 1: 탈모 예방 두피 마사지", width=180)

with col2:
    st.markdown(
        """
        두피를 마사지 해주면 혈액순환을 촉진하여 
        모공 속 노폐물과 공해물질을 배출해 영양 공급에 도움이 됩니다.
        적당한 지압은 두피를 활성화시켜 탈모 예방에 유익합니다.
        """
    )

# 블록 2: 좌측 - 이미지, 우측 - 설명
col1, col2 = st.columns([1, 2])  # 왼쪽 열은 1배, 오른쪽 열은 2배 크기

# 블록 2 내용
with col1:
    st.image("streamlit_images/hair_tip_2.jpg", caption="Tip 2: 올바른 샴푸 사용법", width=180)

with col2:
    st.markdown(
        """
        - **Step 1:** 샴푸 전 브러싱하기  
        - **Step 2:** 모발 충분히 적셔주기  
        - **Step 3:** 샴푸 덜어내기  
        - **Step 4:** 손으로 충분히 거품 내어주기... 
        """
    )


st.link_button("두피 이미지 데이터셋 바로가기", "https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=216")