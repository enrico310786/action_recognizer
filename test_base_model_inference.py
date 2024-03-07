import torch
import argparse
from pytorchvideo.data.encoded_video import EncodedVideo
import numpy as np
import pandas as pd
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    UniformTemporalSubsample
)

from torchvision.transforms import (
    Compose,
    Lambda,
    Resize
)

from model import R3D_slowfast, TimeSformer, R2plus1d_18

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device: ", device)


class PackPathway(torch.nn.Module):
    """
    Transform for converting video frames as a list of tensors.
    """
    def __init__(self, alpha_slowfast):
        super().__init__()
        self.alpha_slowfast = alpha_slowfast

    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // self.alpha_slowfast
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list


def load_video(video_path, permute_color_frame, transform):

    video = EncodedVideo.from_path(video_path)
    start_time = 0
    # follow this post for clip duration https://towardsdatascience.com/using-pytorchvideo-for-efficient-video-understanding-24d3cd99bc3c
    clip_duration = int(video.duration)
    end_sec = start_time + clip_duration
    video_data = video.get_clip(start_sec=start_time, end_sec=end_sec)
    video_data = transform(video_data)
    video_tensor = video_data["video"]
    if permute_color_frame:
        video_tensor = torch.permute(video_tensor, (1, 0, 2, 3))

    return video_tensor



if __name__ == '__main__':

    """
    The script select a base model, the number of frames to sample and a video. It generate the embedding of the video
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_video', type=str, help='path of the dataset')
    parser.add_argument('--name_model', type=str, help='name of the model to use to find the embedding')
    parser.add_argument('--num_frames_to_sample', type=int, help='number of frames to use')

    opt = parser.parse_args()

    path_video = opt.path_video
    name_model = opt.name_model
    num_frames_to_sample = opt.num_frames_to_sample

    # fixed constants
    alpha_slowfast = 4
    is_slowfast = False
    permute_color_frame = False

    model = None
    # depending on the name of the model fix the other constants
    if name_model == "3d_slowfast":
        is_slowfast = True
        print("Set is_slowfast to True")
        mean = [0.45, 0.45, 0.45]
        std = [0.225, 0.225, 0.225]
        size = 256
        model = R3D_slowfast()

    elif name_model == "timesformer":
        permute_color_frame = True
        print("Set permute_color_frame to True")
        mean = [0.45, 0.45, 0.45]
        std = [0.225, 0.225, 0.225]
        size = 224
        model = TimeSformer()

    elif name_model == "r2+1":
        mean = [0.43216, 0.394666, 0.37645]
        std = [0.22803, 0.22145, 0.216989]
        size = 112
        model = R2plus1d_18()

    # set the model in eval mode
    model = model.eval()
    model = model.to(device)

    # set the transformations
    if not is_slowfast:
        transform = ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(num_frames_to_sample),
                    Lambda(lambda x: x / 255.0),
                    Normalize(mean, std),
                    Resize((size, size))
                ]
            ),
        )
    else:
        transform = ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(num_frames_to_sample),
                    Lambda(lambda x: x / 255.0),
                    Normalize(mean, std),
                    Resize((size, size)),
                    PackPathway(alpha_slowfast)
                ]
            ),
        )

    # load the video
    tensor_video = load_video(path_video, permute_color_frame, transform)

    # add one extra dimension to make the inference
    if is_slowfast:
        print("tensor_video[0].size(): ", tensor_video[0].size())
        print("tensor_video[1].size(): ", tensor_video[1].size())
        tensor_video = [i.to(device)[None, ...] for i in tensor_video]
    else:
        print("tensor_video.size(): ", tensor_video.size())
        tensor_video = tensor_video[None].to(device)

    # inference to get the embedding
    with torch.no_grad():
        embedding = model(tensor_video)[0]

    print("embedding.size(): ", embedding.size())


