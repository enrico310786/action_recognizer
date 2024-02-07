import argparse
import os
import shutil

from pytorchvideo.data.encoded_video import EncodedVideo
import numpy as np
import skvideo.io
import cv2
import albumentations as A
import pandas as pd

transform = A.ReplayCompose([
    A.GridDistortion(distort_limit=0.2, p=0.7),
    A.Rotate(limit=7, p=0.6),
    A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=10, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
    A.CLAHE(p=0.7),
    A.PixelDropout(drop_value=0, dropout_prob=0.02, p=0.3),
    A.PixelDropout(drop_value=255, dropout_prob=0.02, p=0.3),
    A.Blur(blur_limit=(2, 4), p=0.5),
])


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


def augment_frames(frame_list):
    data = None
    augmented_frame_list = []

    for i, item in enumerate(frame_list):
        if i == 0:
            first_image = item
            data = transform(image=first_image)
            new_image = data['image']
        else:
            new_image = A.ReplayCompose.replay(data['replay'], image=item)['image']
        augmented_frame_list.append(new_image)
    return augmented_frame_list


def create_augmented_video(frame_list, path_augmented_video, fps):

    writer = skvideo.io.FFmpegWriter(path_augmented_video,
                                     inputdict={'-r': str(fps)},
                                     outputdict={'-r': str(fps), '-c:v': 'libx264', '-preset': 'ultrafast', '-pix_fmt': 'yuv444p'})

    for i, image in enumerate(frame_list):
        image = image.astype('uint8')
        writer.writeFrame(image)

    # close writer
    writer.close()


if __name__ == "__main__":
    '''
    Script take the video of the recordered dataset grouped by classes and tranform them from mov to mp4 extension
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_orig_dataset', type=str, help='directory where is stored the original dataset')
    parser.add_argument('--dir_augment_dataset', type=str, help='directory where to save the augmented dataset')
    parser.add_argument('--n_applications', type=int, help='number of applications per video')

    opt = parser.parse_args()
    dir_orig_dataset = opt.dir_orig_dataset
    dir_augment_dataset = opt.dir_augment_dataset
    n_applications = int(opt.n_applications)

    print("create the directory: {}".format(dir_augment_dataset))
    os.makedirs(dir_augment_dataset, exist_ok=True)

    # create two dataset. The augmented video in the train set, the original video in the test set
    df_train = pd.DataFrame(columns=['CLASS', 'LABEL', 'PATH_VIDEO', 'ENG_CLASS'])
    df_val = pd.DataFrame(columns=['CLASS', 'LABEL', 'PATH_VIDEO', 'ENG_CLASS'])

    # load the dataframe of the original dataset
    path_dataframe = os.path.join(dir_orig_dataset, "dataset_info.csv")
    df = pd.read_csv(path_dataframe)

    # iter over the row of the dataframe
    class2label = {}
    for index, row in df.iterrows():
        class_name = row["CLASS"]
        eng_class_name = row["ENG_CLASS"]
        label = row["LABEL"]
        path_video = os.path.join(dir_orig_dataset, row["PATH_VIDEO"])
        name_video = path_video.split("/")[-1]

        # create the directory on the augmented dataset if it not exist
        augmented_directory = os.path.join(dir_augment_dataset, class_name)
        CHECK_FOLDER = os.path.isdir(augmented_directory)
        if not CHECK_FOLDER:
            print("create the directory: {}".format(augmented_directory))
            os.makedirs(augmented_directory, exist_ok=True)

        # copy the original video and set it in the validation dataset
        new_file_path = os.path.join(augmented_directory, name_video)
        relative_path = ("/").join(new_file_path.split("/")[-2:])
        print("Copio il file originale '{}' in '{}' ".format(path_video, new_file_path))
        shutil.copyfile(path_video, new_file_path)

        df_val = df_val.append({'CLASS': class_name,
                                'PATH_VIDEO': relative_path,
                                'LABEL': label,
                                'ENG_CLASS': eng_class_name}, ignore_index=True)

        # load the video and divide it into frames. The output are RGB frames
        frame_list, _, fps, _ = load_video(path_video)

        for i in range(n_applications):
            # 2: augment each frames
            augmented_frame_list = augment_frames(frame_list)

            # 3: generate the new video with the augmented frames
            name_augmented_video = str(name_video.split(".")[0]) + "_" + str(i + 1) + ".mp4"
            path_augmented_video = os.path.join(augmented_directory, name_augmented_video)
            relative_path = ("/").join(path_augmented_video.split("/")[-2:])
            create_augmented_video(augmented_frame_list, path_augmented_video, fps)

            df_train = df_train.append({'CLASS': class_name,
                                        'PATH_VIDEO': relative_path,
                                        'LABEL': label,
                                        'ENG_CLASS': eng_class_name}, ignore_index=True)

    print("save the train csv")
    path_train_csv = os.path.join(dir_augment_dataset, "df_train.csv")
    df_train.to_csv(path_train_csv, index=False)

    print("****************************************************************")

    print("TRAIN DATASET FEATURES")
    print(df_train.info())
    print("")
    print("CLIP DISTRIBUTION BY CLASS")
    print("")
    desc_grouped = df_train[['CLASS']].value_counts()
    print(desc_grouped)

    print("save the valid csv")
    path_valid_csv = os.path.join(dir_augment_dataset, "df_val.csv")
    df_val.to_csv(path_valid_csv, index=False)

    print("****************************************************************")

    print("VALIDATION DATASET FEATURES")
    print(df_val.info())
    print("")
    print("CLIP DISTRIBUTION BY CLASS")
    print("")
    desc_grouped = df_val[['CLASS']].value_counts()
    print(desc_grouped)




