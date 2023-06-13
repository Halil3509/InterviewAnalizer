from emotionAnalizer import EmotionAnalizer

MODEL_PATH = "./transfer_learning_model.h5"

emotionAnalizer = EmotionAnalizer(MODEL_PATH)

video_path = "./video.mp4"

emotion_score = emotionAnalizer.get_emotion_score(video_path)

print(emotion_score)