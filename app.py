
import numpy as np
import cv2
import time
import os
import pandas as pd
import altair as alt
from ter import get_prediction_proba,predict_emotions_text,emotions_emoji_dict,pipe_lr
import tempfile
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase, WebRtcMode
from streamlit_option_menu import option_menu
from audio_recorder_streamlit import audio_recorder
from ser import speech_predict_emotion, model


st.set_page_config(page_title="Emotion Detective App", page_icon=":camera::microphone:",layout="wide")

with st.spinner("Loading models and other dependencies. Please Wait...."):
    from fer import classifier, predict_emotion
    from keras.models import load_model
    from tensorflow import keras
    from tensorflow.keras.preprocessing.image import img_to_array


st.toast('Model loaded successfully')
time.sleep(0.3)
st.toast('Application is Ready to go', icon='🎉',)



page_bg_col = """
<style>
[data-testid="stAppViewContainer"] {
background-image:url("https://www.magicpattern.design/_next/image?url=https%3A%2F%2Fstorage.googleapis.com%2Fbrandbird%2Fmagicpattern%2Fwallpapers%2Fmagicpattern-mesh-gradient-1635770568709-preview.jpg&w=3840&q=75");
background-size:cover;
}

[data-testid="stSidebar"]{
}

div[data-testid=stToast] {
                
                background-color: #33CC5A;
            }
[data-testid=toastContainer] [data-testid=stMarkdownContainer] > p{
font-size:18px;
color: #FFFFFF;
}
</style>
"""

emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']
emotion_emoji = ['😡','🤢','😨','😃','😐','😢','😲']


try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
except Exception:
    st.write("Error loading cascade classifiers")


class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        #image gray
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            image=img_gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img=img, pt1=(x, y), pt2=(
                x + w, y + h), color=(0, 255, 255), thickness=2)
            roi_gray = img_gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                prediction = classifier.predict(roi)[0]
                maxindex = int(np.argmax(prediction))
                finalout = emotion_labels[maxindex]
                output = str(finalout)
            label_position = (x, y-10)
            cv2.putText(img, output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return img

def main():
    st.markdown(page_bg_col,unsafe_allow_html=True)
    
    # activiteis = ["Home", "Home 2","Real Time Face Emotion Detection", "Emotion Detection from Video"]
    # with st.sidebar:
    selected = option_menu(
    menu_title=None,
    options=["Home", "Text Based Emotion Detection","Speech Emotion Detection","Real Time Emotion Detection", "Emotion Detection from Video","About"],
    icons=["house-fill","chat-right-quote-fill", "mic-fill","camera-reels-fill","collection-play-fill","file-person-fill"],
    default_index=0,
    orientation='horizontal'
)
    # choice = st.sidebar.selectbox("Select Activity", activiteis)
    st.title("Emotion Detection Application")
    # Homepage.
    if selected == "Home":
        html_temp_home1 = """<div style="background-color:#FC4C02;padding:0.5px">
                             <h4 style="color:white;text-align:center;">
                            Let's Understand people's emotion better.
                             </h4>
                             </div>
                             </br>"""

        st.markdown(html_temp_home1, unsafe_allow_html=True)
         # Description
        st.markdown(f'<style>.green-text{{color:green;font-size:18px}}</style><p class="green-text">This web app uses text, facial and audio recognition technology to detect emotions in real-time.</p>', unsafe_allow_html=True)

        
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
        ## 1. Anger 😠
    ## 2. Happy 😃 
    ## 3. Sad 😢
    ## 4. Disgust 🤢
""")
        with col2:
            st.markdown("""
        ## 5. Fear 😨
        ## 6. Neutral 😐
        ## 7. Surprise 😲
""")
            
    elif selected == "Speech Emotion Detection":
        st.title("Speech Emotion Recognition")
        audio_bytes = audio_recorder(energy_threshold=(-1.0, 1.0), pause_threshold=13.0)
        if audio_bytes:
            with open("recorded_audio.wav", "wb") as f:
                f.write(audio_bytes)
            st.success("Audio recorded...")

            st.audio(audio_bytes, format="audio/wav")
            
            
            
            # Display emotion predictions as a table
            # st.subheader("Emotion Predictions")
            # for emotion, probability in predictions.items():
            #     st.write(f"{emotion}: {probability}")

            submit_pred = st.button(label='Submit')
            if submit_pred:
                # Predict emotions
                with st.spinner("Predicting Emotions....."):
                    predictions = speech_predict_emotion(audio_bytes)
                    st.write(predictions)
                    
                col1,col2 = st.columns(2)
                with col1:
                    # Display the predicted emotion
                    predicted_emotion = max(predictions, key=predictions.get)
                    st.success("Predicted Emotion")
                    st.write(predicted_emotion)

                with col2:
                    # Create DataFrame for plotting
                    st.success("Prediction Probability Graph")
                    proba_df = pd.DataFrame(predictions.values(), index=predictions.keys(), columns=["probability"])
                    proba_df.reset_index(inplace=True)
                    proba_df.rename(columns={"index": "emotions"}, inplace=True)

                    # Plotting using Altair
                    fig = alt.Chart(proba_df).mark_bar().encode(x='emotions', y='probability', color='emotions')
                    st.altair_chart(fig, use_container_width=True)


    elif selected == "Text Based Emotion Detection":
        st.title("Text Emotion Detection")
        st.subheader("Detect Emotions In Text")

        with st.form(key='my_form'):
            raw_text = st.text_area("Type Here")
            submit_text = st.form_submit_button(label='Submit')

        if submit_text:
            col1, col2 = st.columns(2)

            prediction = predict_emotions_text(raw_text)
            probability = get_prediction_proba(raw_text)
            st.write(probability)

            with col1:
                st.success("Original Text")
                st.write(raw_text)

                st.success("Prediction")
                emoji_icon = emotions_emoji_dict[prediction]
                st.write("{}:{}".format(prediction, emoji_icon))
                st.write("Confidence:{}".format(np.max(probability)))

            with col2:
                st.success("Prediction Probability")
                #st.write(probability)
                proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
                #st.write(proba_df.T)
                proba_df_clean = proba_df.T.reset_index()
                proba_df_clean.columns = ["emotions", "probability"]

                fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions', y='probability', color='emotions')
                st.altair_chart(fig, use_container_width=True)

    elif selected == "Real Time Emotion Detection":
        st.header("Webcam Real Time Feed")
        webrtc_streamer(key="example", video_processor_factory=VideoTransformer)

    elif selected == "Emotion Detection from Video":
        st.markdown("### Upload a video to predict emotions.")
        # Upload video
        video_file = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov', 'mkv'])
        st.button("Reset", type="primary")
        if st.button('Start Prediction'):
            if video_file is not None:
                # Save the uploaded video to a temporary file
                temp_video_file = tempfile.NamedTemporaryFile(delete=False)
                temp_video_file.write(video_file.read())

                # Process video frames
                cap = cv2.VideoCapture(temp_video_file.name)
                video_placeholder = st.empty()
                emoji_placeholder = st.empty()

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Make prediction and display annotated frame
                    annotated_frame,emoji_out = predict_emotion(frame)
                    video_placeholder.image(annotated_frame, channels="BGR")
                    emoji_placeholder.markdown(f"<span style='font-size: 40px;'>Predicted Emotion is:</span><span style='font-size: 70px;'>{emoji_out}</span>", unsafe_allow_html=True)
                # Close the temporary file
                temp_video_file.close()
                os.unlink(temp_video_file.name)

            else:
                st.write("Please upload a video file.")
    elif selected == "About":
        st.write(f"#### Project ID: P7033")
        st.subheader("Project Description:")
        st.write("Our project is about training models with facials expression data (FER 2013 DATA) and RAVDESS (Speech data) to recognize and detect human emotions visually and via audio. (eg: for better recommendation based on user's emotion feedback.)") 

        # Team members section
        st.subheader("Meet the Team:")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.write("**Linga Richitha**")
            st.write("222010303029")

        with col2:
            st.write("**Manish Ram**")
            st.write("222010303047")

        

        with col3:
            st.write("**Anish Sanghai**")
            st.write("222010304024")

        with col4:
            st.write("**Karan Singha**")
            st.write("222010303022")


        guide_details= "Guide: S Vijaya Kumar"

        if guide_details:
            st.subheader("Guide Details:")
            st.write(guide_details)
        st.markdown("""
        <style>
        body {
            text-align: center;
        }
        </style>
        """, unsafe_allow_html=True)

    else:
        pass


if __name__ == "__main__":
    main()