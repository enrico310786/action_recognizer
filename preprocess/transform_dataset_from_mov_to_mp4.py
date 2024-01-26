from moviepy.editor import VideoFileClip
import argparse
import os


def convert_mov_in_mp4(input_file, output_file):
    try:
        # Load the MOV video
        video_clip = VideoFileClip(input_file)

        # Attention: usually  -> width, height = video_clip.size. But, i have recordered the video with the
        # iphone rotated, thus the width become the height and vice versa!!!
        #width, height = video_clip.size
        height, width = video_clip.size
        print("width: {} - height: {}".format(width, height))

        # Set manually the height and width to maintain the aspect ration
        video_clip.write_videofile(output_file,
                                   codec='libx264',
                                   audio_codec='aac',
                                   fps=int(video_clip.fps),
                                   preset='ultrafast',
                                   threads=4,
                                   logger=None,
                                   ffmpeg_params=["-vf", f"scale={width}:{height}"])

        print(f"Conversion ended. The video is saved to: {output_file}")

    except Exception as e:
        print(f"Error during the conversion: {str(e)}")


if __name__ == "__main__":
    '''
    Script take the video of the recordered dataset grouped by classes and tranform them from mov to mp4 extension
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_mov_dataset', type=str, help='directory where is stored the mov dataset')
    parser.add_argument('--dir_mp4_dataset', type=str, help='directory where to the mp4 dataset')

    opt = parser.parse_args()
    dir_mov_dataset = opt.dir_mov_dataset
    dir_mp4_dataset = opt.dir_mp4_dataset

    print("create the directory: {}".format(dir_mp4_dataset))
    os.makedirs(dir_mp4_dataset, exist_ok=True)

    # Iterate in the subdirs of the dataset. Each subdir is a class
    for _, action_dir, _ in os.walk(dir_mov_dataset):
        for action in action_dir:
            path_mov_action = os.path.join(dir_mov_dataset, action)
            CHECK_FOLDER = os.path.isdir(path_mov_action)

            counter = 0
            if CHECK_FOLDER:
                print("ACTION: ", action)

                # create the action directory for the mp4 extension
                path_mp4_action = os.path.join(dir_mp4_dataset, action)
                print("create the directory: {}".format(path_mp4_action))
                os.makedirs(path_mp4_action, exist_ok=True)

                video_mov_list = os.listdir(path_mov_action)
                print("Number of mov video: {}".format(len(video_mov_list)))

                for name_mov_video in video_mov_list:
                    path_mov_video = os.path.join(path_mov_action, name_mov_video)
                    name_mp4_video = name_mov_video.split(".")[0] + ".mp4"
                    path_mp4_video = os.path.join(path_mp4_action, name_mp4_video)

                    if os.path.exists(path_mp4_video):
                        print("The mp4 '{}' video already exist. Go to the next".format(path_mp4_video))
                    else:
                        convert_mov_in_mp4(path_mov_video, path_mp4_video)
                    counter += 1

            print("Number of converted video: ", counter)
            print("---------------------------------------------")