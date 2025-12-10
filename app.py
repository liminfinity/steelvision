import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import numpy as np
import pandas as pd
from collections import Counter
import os

# === ПУТИ К МОДЕЛЯМ (ONNX) ===
MODEL_PATHS = {
    "YOLO11s (основная)": "models/yolo11s.onnx",
    "YOLO11n (быстрая)": "models/yolo11n.onnx",
    "YOLOv8s (baseline)": "models/yolov8s.onnx",
    "YOLOv8n (baseline)": "models/yolov8n.onnx",
}

DEFAULT_MODEL_NAME = "YOLO11s (основная)"

# === ТЕСТОВЫЕ ИЗОБРАЖЕНИЯ ===
DEMO_IMAGES = {
    "Crazing": "demo_images/crazing_1.jpg",
    "Inclusion": "demo_images/inclusion_1.jpg",
    "Patches": "demo_images/patches_1.jpg",
    "Pitted surface": "demo_images/pitted_1.jpg",
    "Rolled-in scale": "demo_images/rolledin_1.jpg",
    "Scratches": "demo_images/scratches_1.jpg",
}


# === КЭШ ЗАГРУЗКИ МОДЕЛЕЙ ПО КЛЮЧУ ===
@st.cache_resource
def load_model(model_key: str):
    """
    Загружаем модель один раз по ключу (имени),
    дальше берём из кэша.
    """
    model_path = MODEL_PATHS[model_key]
    return YOLO(model_path, task="detect")


# === БОКОВАЯ ПАНЕЛЬ С НАСТРОЙКАМИ ===
st.sidebar.title("⚙️ Настройки детектора")

model_name = st.sidebar.selectbox(
    "Выберите модель",
    list(MODEL_PATHS.keys()),
    index=list(MODEL_PATHS.keys()).index(DEFAULT_MODEL_NAME),
)

# выбор источника изображения
image_source = st.sidebar.radio(
    "Источник изображения",
    ["Загрузить своё", "Тестовые фото"],
    index=0,
)

conf_thres = st.sidebar.slider(
    "Порог уверенности (confidence)",
    min_value=0.1,
    max_value=0.9,
    value=0.25,
    step=0.05,
)

iou_thres = st.sidebar.slider(
    "IoU threshold (NMS)",
    min_value=0.1,
    max_value=0.9,
    value=0.45,
    step=0.05,
)

show_boxes = st.sidebar.checkbox("Показывать изображение с боксами", value=True)
show_table = st.sidebar.checkbox("Показывать таблицу с детекциями", value=True)
show_counts = st.sidebar.checkbox(
    "Показывать количество дефектов по классам", value=True
)

st.sidebar.markdown("---")
st.sidebar.write("Текущая модель:")
st.sidebar.code(f"{model_name}\n→ {MODEL_PATHS[model_name]}", language="text")

# === ЗАГРУЗКА МОДЕЛИ ПО ВЫБОРУ С ОБРАБОТКОЙ ОШИБОК ===
try:
    model = load_model(model_name)
except Exception as e:
    st.sidebar.error(f"Ошибка при загрузке модели `{model_name}`: {e}")
    st.stop()

# Блок с информацией о модели
st.sidebar.markdown("### ℹ️ Информация о модели")
st.sidebar.write(f"Число классов: {len(model.names)}")
st.sidebar.write("Классы:")
st.sidebar.write(", ".join(model.names.values()))

# === ГЛАВНЫЙ ЗАГОЛОВОК ===
st.title("NEU-DET Steel Defects Detector")

st.markdown(
    """
Это веб-интерфейс для детекции дефектов на изображениях стали.  
Выберите модель и источник изображения:
- загрузите своё фото **или** выберите один из тестовых примеров,
и получите:
    - картинку с bboxes,
    - таблицу с типами дефектов и уверенностью,
    - summary по количеству дефектов.
"""
)

# === ВЫБОР ИЗОБРАЖЕНИЯ ===
img = None
img_caption = ""

if image_source == "Загрузить своё":
    uploaded_file = st.file_uploader(
        "Загрузите изображение", type=["jpg", "jpeg", "png"]
    )
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        img_caption = "Входное изображение (загруженное)"
else:
    demo_name = st.selectbox("Выберите тестовое изображение", list(DEMO_IMAGES.keys()))
    demo_path = DEMO_IMAGES[demo_name]
    if not os.path.exists(demo_path):
        st.error(f"Тестовое изображение не найдено: {demo_path}")
    else:
        img = Image.open(demo_path).convert("RGB")
        img_caption = f"Тестовое изображение: {demo_name}"

if img is not None:
    st.image(img, caption=img_caption, width=700)

    # временно сохраняем картинку, потому что YOLO ждёт путь или np.array
    with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:
        img.save(tmp.name)

        # запускаем детекцию
        results = model(
            tmp.name,
            conf=conf_thres,
            iou=iou_thres,
            save=False,
            verbose=False,
        )

    res = results[0]  # один кадр

    # === 1. Картинка с боксами ===
    if show_boxes:
        res_img = res.plot()  # numpy array (BGR)
        # конвертация BGR -> RGB
        if isinstance(res_img, np.ndarray):
            res_img = res_img[:, :, ::-1]
        st.image(res_img, caption=f"Результат детекции ({model_name})", width=700)

    # Если нет боксов — можно сразу сказать об этом
    if res.boxes is None or len(res.boxes) == 0:
        st.warning("Дефекты не найдены при текущем пороге уверенности.")
    else:
        # === 2. Таблица с детекциями ===
        boxes = res.boxes  # ultralytics.yolo.engine.results.Boxes
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        xyxy = boxes.xyxy.cpu().numpy()  # [x1,y1,x2,y2]

        # имена классов из модели
        id2name = model.names

        rows = []
        for i in range(len(cls_ids)):
            cls_id = cls_ids[i]
            x1, y1, x2, y2 = xyxy[i]
            rows.append(
                {
                    "Класс": id2name.get(cls_id, str(cls_id)),
                    "Уверенность": float(confs[i]),
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                }
            )

        df = pd.DataFrame(rows)

        if show_table:
            st.subheader("Детекции (таблица)")
            st.dataframe(df.style.format({"Уверенность": "{:.3f}"}), width="stretch")

        # === 3. Количество дефектов по типам ===
        if show_counts:
            st.subheader("Количество дефектов по классам")
            counts = Counter(df["Класс"])
            counts_df = (
                pd.DataFrame([{"Класс": k, "Количество": v} for k, v in counts.items()])
                .sort_values("Количество", ascending=False)
                .reset_index(drop=True)
            )
            st.table(counts_df)

            st.markdown(
                f"**Всего детекций:** {len(df)}; "
                f"уникальных классов: {counts_df.shape[0]}"
            )
else:
    st.info(
        "Загрузите изображение или выберите тестовое, чтобы выполнить детекцию дефектов."
    )
