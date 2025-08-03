import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tempfile
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import Model

tf.config.run_functions_eagerly(True)

def grad_cam(model, image, layer_names=['conv2d', 'conv2d_1', 'conv2d_2', 'conv2d_3', 'conv2d_4', 'conv2d_5', 'conv2d_8', 'conv2d_9', 'conv2d_7']):
    image = tf.Variable(image)
    inputs = [image] if isinstance(model.input, list) else image

    heatmaps = {}

    for layer_name in layer_names:
        conv_layer = model.get_layer(layer_name)
        grad_model = Model(inputs=model.inputs, outputs=[conv_layer.output, model.output])

        with tf.GradientTape() as tape:
            tape.watch(image)
            conv_outputs, predictions = grad_model(inputs)
            class_index = tf.argmax(predictions[0])
            loss = predictions[:, class_index]

        grads = tape.gradient(loss, conv_outputs)
        if grads is None:
            heatmaps[layer_name] = None
            continue

        grads = grads[0]
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        pooled_grads = pooled_grads[..., tf.newaxis]

        heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
        heatmap = tf.maximum(heatmap, 0)
        heatmap_max = tf.reduce_max(heatmap)

        if heatmap_max == 0:
            heatmap = tf.zeros_like(heatmap)
        else:
            heatmap /= heatmap_max

        heatmap_resized = cv2.resize(heatmap.numpy(), (image.shape[2], image.shape[1]))
        heatmaps[layer_name] = heatmap_resized
        #st.write(f"[DEBUG] Layer: {layer_name}, Grad max: {tf.reduce_max(grads)}, min: {tf.reduce_min(grads)}")

    return heatmaps


st.set_page_config(
    page_title="Microscopic Fungus Colony Classifier",
    page_icon="🧫",
    layout="wide"
)

@st.cache_resource
def loading_model():
    try:
        model = load_model('./model/model4.h5')
        return model
    except Exception as e:
        st.write(f'Fail to Load Model. Error: {e}')

model = loading_model()
class_ids = ['H1', 'H2', 'H3', 'H5', 'H6']
# ========== TITLE ==========
st.markdown(
    "<h1 style='text-align: center; color: #4B8BBE;'>🧫 Microscopic Fungus Colony Classification with CNN Resedual Network </h1>",
    unsafe_allow_html=True
)
st.markdown("<hr>", unsafe_allow_html=True)

# ========== SIDEBAR ==========
st.sidebar.markdown("## 📤 Upload Microscopic Image")
uploaded_file = st.sidebar.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    original_img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    height, width, channels = original_img_rgb.shape
    if height !=224 or width !=224:
        original_img_rgb = cv2.resize(original_img_rgb, (224,224), interpolation=cv2.INTER_LANCZOS4)
    
    file_type = uploaded_file.type
    file_type = file_type.split('/')[-1]

    img_input = original_img_rgb.astype(np.float32) / 255.0
    img_input = np.expand_dims(img_input, axis=0)
    
    results = model.predict(img_input)
    predicted_class = class_ids[results.argmax()]
    conf = results.max()
    #st.write(results)
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 🧪 Prediction Result")
    st.sidebar.success(f"**Predicted Class:** {predicted_class}")
    st.sidebar.success(f"**Confidence:** {conf:.2f}")

    heatmaps = grad_cam(model, img_input)
    
    available_layers = [layer for layer in heatmaps if heatmaps[layer].sum() != 0.0]
    #layer = st.selectbox("Choose a layer:", available_layers)
    st.markdown("### Select GradCAM Layer")
    layer = st.radio("", available_layers, horizontal=True)

    col1, col2 = st.columns(2)
    with col1:
        st.image(Image.fromarray(original_img_rgb), caption="📷 Original Image", width=500)
        #st.write("Predictions:", results.numpy())
        #st.write("Loss:", results)

    with col2:
        grad_cam_img = heatmaps[layer]

         # Convert to uint8 and apply colormap
        heatmap_uint8 = np.uint8(255 * grad_cam_img)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for PIL
        st.image(Image.fromarray(heatmap_colored), caption = f"🎯 GradCAM Layer {layer}", width=500)


