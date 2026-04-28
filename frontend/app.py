import streamlit as st
import requests
from PIL import Image
import io
import matplotlib.pyplot as plt
import numpy as np
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Мультимодельная классификация", layout="wide")
st.title("Практическая работа №10: Демонстрация моделей")

# API_URL = "http://localhost:7860/predict/" # Для локального тестирования
API_URL = "https://task10-supv.onrender.com/predict/" # После деплоя

st.sidebar.header("Настройки")
model_choice = st.sidebar.selectbox(
    "Выберите модель для задачи:",
    ("Классификация изображений (7 классов)", "Классификация цифр (MNIST)")
)

# Переменная 'images' или 'digits', которая будет передаваться в API
model_type_key = "images" if "изображений" in model_choice else "digits"

# Меняем инструкцию на основе выбранной модели
st.write(f"### Текущая задача: {model_choice}")
if model_type_key == "images":
    st.info("Классификация объектов: bike, cars, cats, dogs, flowers, horses, human.")
else:
    st.info("Распознавание рукописных цифр: от 0 до 9.")

option = st.radio("Выберите способ ввода:", ("Загрузить файл", "Нарисовать на холсте"))

uploaded_image = None

if option == "Загрузить файл":
    file = st.file_uploader("Загрузите изображение", type=["png", "jpg", "jpeg"])
    if file:
        uploaded_image = Image.open(file)
        st.image(uploaded_image, caption="Загруженное фото", width=300)

elif option == "Нарисовать на холсте":
    # Меняем текст и кисть в зависимости от модели
    if model_type_key == "digits":
        st.write("Нарисуйте цифру:")
        stroke_w = 20
        stroke_c = "white"
        bg_c = "black"
    else:
        st.write("Попробуйте нарисовать:")
        stroke_w = 10
        stroke_c = "black"
        bg_c = "white"
        
    canvas_result = st_canvas(
        fill_color=bg_c,
        stroke_width=stroke_w,
        stroke_color=stroke_c,
        background_color=bg_c,
        width=400,
        height=400,
        drawing_mode="freedraw",
        key=f"canvas_{model_type_key}",
    )
    
    if canvas_result.image_data is not None:
        image_data = canvas_result.image_data
        image = Image.fromarray(image_data.astype('uint8'), 'RGBA')
        # Создаем фон, чтобы исключить прозрачность (черный для цифр, белый для объектов)
        bg_color = (0, 0, 0) if model_type_key == "digits" else (255, 255, 255)
        background = Image.new('RGB', image.size, bg_color)
        background.paste(image, mask=image.split()[3]) 
        uploaded_image = background


if uploaded_image:
    if st.button("Отправить в API"):
        with st.spinner(f"⏳ Обработка на сервере ({model_type_key})..."):
            try:
                img_byte_arr = io.BytesIO()
                if uploaded_image.mode != "RGB":
                    uploaded_image = uploaded_image.convert("RGB")
                    
                uploaded_image.save(img_byte_arr, format='JPEG') 
                img_byte_arr = img_byte_arr.getvalue()
                
                # В FastAPI теперь нужно передавать файл и form-data!
                files = {'file': ('image.jpg', img_byte_arr, 'image/jpeg')}
                data = {'model_type': model_type_key} # Сообщаем серверу какую модель грузить
                
                response = requests.post(API_URL, files=files, data=data)
                
                if response.status_code == 200:
                    result = response.json()
                    st.success("✅ Успешно классифицировано!")
                    
                    predicted_class_name = result["predicted_class_name"]
                    confidence = result["confidence"]
                    all_probs = result["all_probabilities"]
                    returned_classes = result["classes"]
                    
                    st.markdown(f"### **Предсказанный класс: {predicted_class_name}**")
                    st.markdown(f"**Уверенность:** {confidence * 100:.2f}%")
                    
                    st.markdown("#### Распределение вероятностей:")
                    fig, ax = plt.subplots(figsize=(10, 4))
                    
                    y_pos = np.arange(len(returned_classes))
                    ax.bar(y_pos, all_probs, color='skyblue')
                    ax.set_xticks(y_pos)
                    
                    # Для изображений поворачиваем подписи, для цифр оставляем прямо
                    rotation = 45 if model_type_key == "images" else 0
                    ha = 'right' if model_type_key == "images" else 'center'
                    ax.set_xticklabels(returned_classes, rotation=rotation, ha=ha)
                    
                    ax.set_xlabel('Класс')
                    ax.set_ylabel('Вероятность')
                    ax.set_title(f'Уверенность модели ({model_type_key})')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                else:
                    st.error(f"❌ Ошибка API (Статус-код: {response.status_code})")
                    details = response.json().get("detail", response.text)
                    st.write(f"Детали: {details}")
                    
            except requests.exceptions.ConnectionError:
                st.error(f"❌ Не удалось подключиться к API по адресу {API_URL}.")
                st.info("Убедитесь, что backend запущен (или API_URL правильный).")