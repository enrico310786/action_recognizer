import cv2
import argparse
import os
from pytorchvideo.data.encoded_video import EncodedVideo
import numpy as np
import skvideo.io


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


def make_square_video(path_original_video, path_square_video, size, offset, sampling_rate=1, fps_rate=30):

    # 1) load video as list of RGB frames
    frame_list, num_frames, fps, video_duration = load_video(path_original_video)

    #print("num_frames: ", num_frames)
    #print("fps: ", fps)
    #print("video_duration: ", video_duration)

    # 2) center crop and sampling of frames
    squared_frames_list = []
    for i, frame in enumerate(frame_list):
        if i % sampling_rate == 0:
            # pass to BGR
            # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            # center crop
            h, w, c = frame.shape
            center_x = int(w / 2)
            center_y = int(h / 2)

            if center_x > center_y:
                # the frame is larger
                frame = frame[0: 2 * center_y, center_x - center_y: center_x + center_y]
            elif center_x < center_y:
                # the frame is higher
                frame = frame[center_y - center_x - offset: center_y + center_x - offset, 0: 2 * center_x]

            # resize the frame to the selected size
            frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
            squared_frames_list.append(frame)

    # 3) create new video clip
    create_video_clip(frame_list=squared_frames_list,
                      path_saving_video=path_square_video,
                      number_files_frames=len(squared_frames_list),
                      frame_size=size,
                      fps_rate=fps_rate)


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


if __name__ == "__main__":
    '''
    Script to transform the video of the dataset from the rectangle to the square shape
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_rect_dataset', type=str, help='directory where is stored the rectangle shape dataset')
    parser.add_argument('--dir_square_dataset', type=str, help='directory where to save the square shape dataset')
    parser.add_argument('--size', type=int, default=1080, help='finale size for the frames')
    parser.add_argument('--offset', type=int, default=250, help='offset to ad to the crop')

    opt = parser.parse_args()
    dir_rect_dataset = opt.dir_rect_dataset
    dir_square_dataset = opt.dir_square_dataset
    size = int(opt.size)
    offset = int(opt.offset)

    print("final size: ", size)

    print("create the directory: {}".format(dir_square_dataset))
    os.makedirs(dir_square_dataset, exist_ok=True)

    # Iterate in the subdirs of the dataset. Each subdir is a class
    for _, action_dir, _ in os.walk(dir_rect_dataset):
        for action in action_dir:
            path_rect_action = os.path.join(dir_rect_dataset, action)
            CHECK_FOLDER = os.path.isdir(path_rect_action)

            counter = 0
            if CHECK_FOLDER:
                print("ACTION: ", action)

                # create the action directory for the square shape
                path_square_action = os.path.join(dir_square_dataset, action)
                print("create the directory: {}".format(path_square_action))
                os.makedirs(path_square_action, exist_ok=True)

                video_rect_list = os.listdir(path_rect_action)
                print("Number of rect shape video: {}".format(len(video_rect_list)))

                for name_rect_video in video_rect_list:
                    path_rect_video = os.path.join(path_rect_action, name_rect_video)
                    name_square_video = name_rect_video
                    path_square_video = os.path.join(path_square_action, name_square_video)

                    if os.path.exists(path_square_video):
                        print("The square video '{}' already exist. Go to the next".format(path_square_video))
                    else:
                        make_square_video(path_original_video=path_rect_video,
                                          path_square_video=path_square_video,
                                          size=size,
                                          offset=offset)
                    counter += 1

            print("Number of squared video: ", counter)
            print("---------------------------------------------")