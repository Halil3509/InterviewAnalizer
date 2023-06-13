from handAnalizer import HandAnalizer

video_path = "./video3.mp4"

handAnalizer = HandAnalizer()

result = handAnalizer.get_hand_score(video_path, fps = 1)

print(result)