import cv2
from pytorchvideo.data.encoded_video import EncodedVideo
import numpy as np
import skvideo.io

'''
Script to test the transformation of a mp3 video of the dataset from rectangle to square. 
In this case I noted that the recordered video have the heigth and width inverted
'''

path_video = "/home/enrico/Dataset/Actions/test_actions/IMG_5576.mp4"
sampling_rate = 1 # if 1 the take all frames
path_new_video = "/home/enrico/Dataset/Actions/test_actions/IMG_5576_square.mp4"
fps_rate = 30 # 30 if sampling rate is 1, 10 if sampling rate is 3
offset = 250
#################################################################

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


def create_video_clip(frame_list, path_saving_video, number_files_frames, frame_size, fps_rate):
    '''
    :param frame_list: list of RGB frames
    :param path_saving_video:
    :param number_files_frames:
    :param frame_size:
    :param fps_rate:
    :return:
    '''
    out_video = np.empty([number_files_frames, frame_size, frame_size, 3], dtype=np.uint8)
    out_video = out_video.astype(np.uint8)

    i = 0
    for img in frame_list:
        out_video[i] = img
        i += 1

    skvideo.io.vwrite(path_saving_video,
                      out_video,
                      inputdict={'-r': str(int(fps_rate))})
    print("Created clip video: ", path_saving_video)
    return


# 1) load video as list of RGB frames
frame_list, num_frames, fps, video_duration = load_video(path_video)

print("num_frames: ", num_frames)
print("fps: ", fps)
print("video_duration: ", video_duration)

# 2) center crop and sampling of frames
squared_frames_list = []
for i, frame in enumerate(frame_list):
    if i % sampling_rate == 0:
        # pass to BGR
        #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # center crop
        h, w, c = frame.shape
        center_x = int(w / 2)
        center_y = int(h / 2)

        if center_x > center_y:
            # the frame is larger
            frame = frame[0: 2 * center_y, center_x - center_y: center_x + center_y]
        elif center_x < center_y:
            # the frame is higher
            frame = frame[center_y - center_x - offset: center_y + center_x - offset, 0 : 2 * center_x]

        #print("frame.shape: ", frame.shape)
        squared_frames_list.append(frame)
        # resize
        #frame = cv2.resize(frame, (new_size, new_size), interpolation=cv2.INTER_AREA)
        # save the frame

print("length of squared_frames_list: ", len(squared_frames_list))
height = squared_frames_list[0].shape[0]
width = squared_frames_list[0].shape[1]
print("height: {} - width:{}".format(height, width))

# 3) create new video clip
create_video_clip(squared_frames_list, path_new_video, len(squared_frames_list), squared_frames_list[0].shape[0], fps_rate)