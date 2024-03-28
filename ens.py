from test import video_to_audio, audio_to_text
import ter
import ser


model_weights={
    'audio':0.7,
    'video':0.7,
    'text':0.5
}
def ensembling(video_path):

    audio_path = f"output/audio.wav"
    text_path = f"output/text.txt"
    video_to_audio(video_path,audio_path)
    audio_to_text(audio_path,text_path)
    with open(audio_path, "rb") as audio_file:
        audio_data = audio_file.read()
    #extract features and everything

    audio_pred=ser.speech_predict_emotion(audio_data)
    print(audio_pred)
    video_pred=video_model.predict(video_input)
    text_pred=ter.get_prediction_proba(text_path)
    print(text_pred)

    audio_pred_confidence=123
    video_pred_confidence=456
    text_pred_confidence=789

    weighted_scores = {
        'audio'+str(audio_pred):model_weights['audio']*audio_pred_confidence,
        'video'+str(video_pred):model_weights['video']*video_pred_confidence,
        'text'+str(text_pred):model_weights['text']*text_pred_confidence
    }
    print("hello",weighted_scores)

    final_emotion = max(weighted_scores, key=weighted_scores.get)
    print(final_emotion)


ensembling('D:\\Facial Emotion detection\\output\\video.mp4')
