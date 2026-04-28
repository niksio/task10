import streamlit as st
import requests
from PIL import Image
import io
import matplotlib.pyplot as plt
import numpy as np
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Классификация изображений", layout="wide")

st.title("Практическая работа №10: Демонстрация модели классификации")

# Используйте этот URL локально или замените на Render.com/HuggingFace после деплоя
API_URL = "http://localhost:7860/predict/"

option = st.radio("Выберите способ ввода:", ("Загрузить файл", "Нарисовать на холсте"))

uploaded_image = None

if option == "Загрузить файл":
    file = st.file_uploader("Загрузите изображение", type=["png", "jpg", "jpeg"])
    if file:
        uploaded_image = Image.open(file)
        st.image(uploaded_image, caption="Загруженное фото", width=300)

elif option == "Нарисовать на холсте":
    # Холст для рисования (идеально для MNIST / чисел)
    st.write("Нарисуйте объект/число по центру:")
    canvas_result = st_canvas(
        fill_color="white",
        stroke_width=15,
        stroke_color="black",
        background_color="white",
        width=280,
        height=280,
        drawing_mode="freedraw",
        key="canvas",
    )
    if canvas_result.image_data is not None:
        # Конвертация numpy массива из canvas в PIL Image
        uploaded_image = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')


if uploaded_image:
    if st.button("Отправить в API"):
        with st.spinner("⏳ Обработка на сервере..."):
            try:
                # Преобразование изображения в байты для отправки
                img_byte_arr = io.BytesIO()
                # Для canvas может быть RGBA, сохраним в PNG
                if uploaded_image.mode in ("RGBA", "P"):
                    uploaded_image = uploaded_image.convert("RGB")
                    
                uploaded_image.save(img_byte_arr, format='PNG')
                img_byte_arr = img_byte_arr.getvalue()
                
                # Отправка POST запроса к FastAPI бэкенду
                files = {'file': ('image.png', img_byte_arr, 'image/png')}
                response = requests.post(API_URL, files=files)
                
                if response.status_code == 200:
                    result = response.json()
                    st.success("✅ Успешно классифицировано!")
                    
                    # Извлечение результатов
                    predicted_class = result["predicted_class"]
                    confidence = result["confidence"]
                    all_probs = result["all_probabilities"]
                    
                    st.markdown(f"### **Предсказанный класс: {predicted_class}**")
                    st.markdown(f"**Уверенность:** {confidence * 100:.2f}%")
                    
                    # Визуализация вероятностей
                    st.markdown("#### Распределение вероятностей по всем классам:")
                    fig, ax = plt.subplots(figsize=(10, 4))
                    classes = list(range(len(all_probs)))
                    ax.bar(classes, all_probs, color='skyblue')
                    ax.set_xticks(classes)
                    ax.set_xlabel('Класс')
                    ax.set_ylabel('Вероятность')
                    ax.set_title('Уверенность модели для каждого класса')
                    
                    st.pyplot(fig)
                    
                else:
                    st.error(f"❌ Ошибка API (Статус-код: {response.status_code})")
                    st.write(response.text)
                    
            except requests.exceptions.ConnectionError:
                st.error(f"❌ Не удалось подключиться к API по адресу {API_URL}.")
                st.info("Убедитесь, что backend запущен локально (uvicorn main:app) или вы обновили API_URL.")

