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
            <h5 style="text-align: center;">양호 >> 경증 >> 중등도 >> 중증</h5>
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

    # "탈모 진행 상태 확인하기"
    with st.form(key="my_form"):
        # 버튼 추가 (스타일 조정)
        st.markdown(
            """
            <div style="display: flex; justify-content: center; align-items: center; margin-top: 20px;">
                <button type="submit" style="
                    background-color: #007BFF;
                    color: white;
                    padding: 20px 40px;
                    font-size: 24px;
                    border: none;
                    border-radius: 10px;
                    cursor: pointer;
                    text-align: center;
                ">
                    탈모 진행 상태 확인하기 🔎
                </button>
            </div>
            """,
            unsafe_allow_html=True
        )

        # 여기서 버튼 클릭 시의 이벤트 처리
        submit_button = st.form_submit_button("탈모 진행 상태 확인하기🔎")

        if submit_button:
            pred_proba = model.predict(batch_image)
            pred = np.argmax(pred_proba)
            pred_label = class_names[pred]
            pred_probability = pred_proba[0][pred]

            # 결과 표시 컨테이너
            st.markdown(
                """
                <div style="background-color: #ffffff; padding: 20px; border-radius: 10px; margin-top: 20px;">
                    <h3 style="text-align: center; color: #333;">모발 상태 예측 결과</h3>
                    <div style="background-color: #e0f7fa; padding: 15px; border-radius: 10px; margin-bottom: 10px;">
                        <h4 style="color: #00796b;">예측: {}</h4>
                    </div>
                    <div style="background-color: #ffe0b2; padding: 15px; border-radius: 10px;">
                        <h4 style="color: #e65100;">예측 확률: {:.2f}%</h4>
                    </div>
                </div>
                """.format(pred_label, pred_probability * 100),
                unsafe_allow_html=True
            )

    # 서버에 저장
    save_path = os.path.join(SAVE_DIR, uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"이미지가 성공적으로 저장되었습니다: {save_path}")


# 데이터 정보 확인
st.markdown("---")

st.link_button("두피 이미지 데이터셋 바로가기", "https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=216")