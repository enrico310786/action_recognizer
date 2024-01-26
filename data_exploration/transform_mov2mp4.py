from moviepy.editor import VideoFileClip


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
        print(f"Si è verificato un errore durante la conversione: {str(e)}")

#################################


input_video = "/home/enrico/Dataset/Actions/test_actions/IMG_5576.mov"
output_video = "/home/enrico/Dataset/Actions/test_actions/IMG_5576.mp4"

convert_mov_in_mp4(input_video, output_video)