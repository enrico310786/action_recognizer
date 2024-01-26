import cv2
from pytorchvideo.data.encoded_video import EncodedVideo

path_video = "/home/enrico/Dataset/Actions/test_actions/mp4/avvitare/IMG_5662.mp4"

def load_video(video_path):

    #print("Load the video: ", video_path)

    frame_list = []
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_list.append(frame)
    cap.release()

    video = EncodedVideo.from_path(video_path)
    return frame_list, len(frame_list), fps, int(video.duration)


frame_list, num_frames, fps, video_duration = load_video(path_video)

print("num_frames: ", num_frames)
print("fps: ", fps)
print("video_duration: ", video_duration)


for item in frame_list:
    print("height: {} - width: {} - color: {}".format(item.shape[0], item.shape[1], item.shape[2]))
    break