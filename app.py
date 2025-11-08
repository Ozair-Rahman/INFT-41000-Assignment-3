import streamlit as st
from streamlit_cropper import st_cropper
from ultralytics import YOLO
from PIL import Image
import io
import numpy as np
import cv2
model = YOLO("yolo11n.pt")

# Function to run YOLO inference
def run_yolo_inference(uploaded_image):
    try:
        # Convert uploaded image to a format suitable for YOLO
        img = Image.open(uploaded_image)
        img_array = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)  # Convert to OpenCV format

        # Run YOLO on the image
        results = model(img_array)

        # Get the result image with bounding boxes (in BGR format)
        result_img_bgr = results[0].plot()

        # Convert from BGR to RGB
        result_img_rgb = cv2.cvtColor(result_img_bgr, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image
        result_pil = Image.fromarray(result_img_rgb)
        return result_pil

    except Exception as e:
        st.error(f"Error during inference: {e}")
        return None

# Function to save the image
def save_image(image, filename="saved_image.png"):
    image.save(filename)
    st.success(f"Image saved as {filename}")

    
# Page selection (Navigation Bar)
page = st.radio("Select a page", ("YOLO Object Detection", "Crop Image"))

if page == "YOLO Object Detection":
    # YOLO Object Detection Page
    st.title("YOLO Object Detection Web App")
    st.subheader("Upload an image to detect objects")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image.", use_container_width=True)
        st.write("")
        st.write("Classifying...")

        with st.spinner('Running YOLO inference...'):
            result_image = run_yolo_inference(uploaded_file)

        if result_image is not None:
            st.success('Inference complete!')
            st.image(result_image, caption="Detected Image with Bounding Boxes", use_container_width=True)

            # Allow users to download the result image
            img_buffer = io.BytesIO()
            result_image.save(img_buffer, format="PNG")
            img_buffer.seek(0)
            st.download_button(
                label="Download Detected Image",
                data=img_buffer,
                file_name="detected_image.png",
                mime="image/png"
            )

elif page == "Crop Image":
    # Image Cropping Page
    st.title("Image Cropping Tool")

    # Upload an image and set some options for demo purposes
    st.header("Cropper Demo")
    img_file = st.sidebar.file_uploader(label='Upload a file', type=['png', 'jpg'])
    realtime_update = st.sidebar.checkbox(label="Update in Real Time", value=True)
    box_color = st.sidebar.color_picker(label="Box Color", value='#0000FF')
    aspect_choice = st.sidebar.radio(label="Aspect Ratio", options=["1:1", "16:9", "4:3", "2:3", "Free"])
    aspect_dict = {
        "1:1": (1, 1),
        "16:9": (16, 9),
        "4:3": (4, 3),
        "2:3": (2, 3),
        "Free": None
    }
    aspect_ratio = aspect_dict[aspect_choice]

    if img_file:
        img = Image.open(img_file)
        # Get a cropped image from the frontend
        cropped_img = st_cropper(img, realtime_update=realtime_update, box_color=box_color,
                                aspect_ratio=aspect_ratio) 

        if not realtime_update:
            st.write("Double click to save crop")
        # Button to save the image
        # Allow users to download the result image
            img_buffer = io.BytesIO()
            cropped_img.save(img_buffer, format="PNG")
            img_buffer.seek(0)
            st.download_button(
                label="Download Cropped Image",
                data=img_buffer,
                file_name="cropped_image.png",
                mime="image/png"
            )
        # Manipulate cropped image at will
        st.write("Preview")
        _ = cropped_img.thumbnail((150,150))
        st.image(cropped_img)

