from moviepy.editor import VideoFileClip, concatenate_videoclips

'''
Script to generate a GIF from a video. The GIF take some frames at the starting, some frames in the middle and some frames at the end
'''

def convert_video_to_gif(input_video_path, output_gif_path, num_frames):
    clip = VideoFileClip(input_video_path)
    clip_without_audio = clip.set_audio(None)  # Remove audio

    # Determine the starting, the middle and the ending point
    total_duration = clip_without_audio.duration
    start_time = 0
    middle_time = total_duration / 2
    end_time = total_duration

    # Generate a subclip for the start, middle and ending
    start_clip = clip_without_audio.subclip(0, start_time + 1)
    middle_clip = clip_without_audio.subclip(middle_time - 1, middle_time + 1)
    end_clip = clip_without_audio.subclip(end_time - 1, end_time)

    # Attach the clips
    final_clip = concatenate_videoclips([start_clip, middle_clip, end_clip])

    # Write the GIF
    final_clip.write_gif(output_gif_path, fps=10)  # Adjust fps as needed


input_video_path = '/home/enrico/Dataset/Actions/test_actions/square_256_mp4/avvitare_antiorario/IMG_5609.mp4'
output_gif_path = '/home/enrico/Dataset/Actions/test_actions/GIF/screw_1432.gif'
num_frames = 20  # Number of frames to take from the video to build the GIF

convert_video_to_gif(input_video_path, output_gif_path, num_frames)