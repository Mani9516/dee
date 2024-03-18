import cv2
import numpy as np
import streamlit as st

# Function to detect deepfake in a video
def detect_video_deepfake(video):
    frames = []
    while True:
        ret, frame = video.read()
        if not ret:
            break
        frames.append(cv2.resize(frame, (224, 224)))  # Assuming the model takes input size (224, 224))
    video.release()

    frames = np.array(frames)
    frames = frames / 255.

    # Example: Simple rule-based detection based on frame analysis
    # You would need a more sophisticated method for real deepfake detection
    is_deepfake = True
    if len(frames) > 1000:  # Example: If the video is longer than 1000 frames, classify as deepfake
        is_deepfake = False

    return is_deepfake

# Function to detect deepfake in an image using computer vision techniques
def detect_image_deepfake(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Calculate variance of Laplacian
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Example threshold for variance
    threshold = 100
    if variance < threshold:
        return True  # If variance is low, likely a deepfake
    else:
        return False  # If variance is high, likely not a deepfake

# Streamlit app
def main():
    st.title("Deepfake Detection App")

    # Option to upload an image or video
    option = st.selectbox("Choose an option:", ["Image", "Video"])

    if option == "Image":
        st.subheader("Detect Deepfake in Image")
        uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

        if uploaded_image is not None:
            image_bytes = uploaded_image.read()
            image_array = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            st.write("Shape of uploaded_image:", image.shape if image is not None else "None")
            if image is not None and image.size > 0:  # Check if image is not None and has data
                st.image(image, caption="Uploaded Image", use_column_width=True)
                is_deepfake = detect_image_deepfake(image)
                st.write("Is the image a deepfake?", is_deepfake)
            else:
                st.write("Invalid or empty image uploaded.")

    elif option == "Video":
        st.subheader("Detect Deepfake in Video")
        uploaded_video = st.file_uploader("Upload a video", type=["mp4"])

        if uploaded_video is not None:
            # Save the uploaded video to a temporary file
            with open("temp_video.mp4", "wb") as f:
                f.write(uploaded_video.read())

            # Read the video from the temporary file
            video = cv2.VideoCapture("temp_video.mp4")

            # Check if the video is opened successfully
            if video.isOpened():
                st.video("temp_video.mp4")
                is_deepfake_video = detect_video_deepfake(video)
                st.write("Is the video a deepfake?", is_deepfake_video)
            else:
                st.write("Failed to open the uploaded video.")

if __name__ == "__main__":
    main()
