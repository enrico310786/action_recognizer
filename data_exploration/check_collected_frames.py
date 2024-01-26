from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    UniformTemporalSubsample
)

from torchvision.transforms import (
    Resize
)

import os
import imageio
import argparse
import torch
from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision.transforms import Compose, Lambda


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


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


def apply_transformation(video_data, num_frames_to_sample, mean, std, resize_to, is_slowfast):

    if not is_slowfast:
        transform = ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(num_frames_to_sample),
                    Lambda(lambda x: x / 255.0),
                    Normalize(mean, std),
                    # RandomShortSideScale(min_size=min_size, max_size=max_size),
                    # RandomCrop(resize_to),
                    Resize((resize_to, resize_to))
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
                    # RandomShortSideScale(min_size=min_size, max_size=max_size),
                    # RandomCrop(resize_to),
                    Resize((resize_to, resize_to)),
                    PackPathway(alpha_slowfast)
                ]
            ),
        )

    return transform(video_data)


def unnormalize_img(img):
    """Un-normalizes the image pixels."""
    img = (img * std) + mean
    img = (img * 255).astype("uint8")
    return img.clip(0, 255)


def save_frames(video_tensor, path_to_save):
    count = 0
    for video_frame in video_tensor:
        frame_unnormalized = unnormalize_img(video_frame.permute(1, 2, 0).numpy())
        path_frame = os.path.join(path_to_save, str(count) + ".png")
        imageio.imwrite(path_frame, frame_unnormalized)
        count += 1


if __name__ == "__main__":
    '''
    Script to collect the frames of a video in the same manner of the dataloader of the action recognition models
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_video_to_analyze', type=str, help='path where is saved the video to sample')
    parser.add_argument('--dir_sampled_frames', type=str, help='directory where to store the sampled frames')
    parser.add_argument('--num_frames_to_sample', type=int, help='number of frames to sample')
    parser.add_argument("--is_slowfast", type=str2bool, nargs='?', const=True, default=False, help="Activate slowfast sampling")
    parser.add_argument("--resize_to", type=int, help="Size of the final square")

    opt = parser.parse_args()
    path_video_to_analyze = opt.path_video_to_analyze
    dir_sampled_frames = opt.dir_sampled_frames
    num_frames_to_sample = int(opt.num_frames_to_sample)
    is_slowfast = opt.is_slowfast
    resize_to = int(opt.resize_to)

    print("resize_to: ", resize_to)
    print("num_frames_to_sample: ", num_frames_to_sample)

    mean = [0.45, 0.45, 0.45]
    std = [0.225, 0.225, 0.225]
    alpha_slowfast = 4

    dir_sampled_frames_0 = None
    dir_sampled_frames_1 = None
    if is_slowfast:
        dir_sampled_frames_0 = os.path.join(dir_sampled_frames, "list_0")
        print("create the directory: {}".format(dir_sampled_frames_0))
        os.makedirs(dir_sampled_frames_0, exist_ok=True)
        dir_sampled_frames_1 = os.path.join(dir_sampled_frames, "list_1")
        print("create the directory: {}".format(dir_sampled_frames_1))
        os.makedirs(dir_sampled_frames_1, exist_ok=True)
    else:
        print("create the directory: {}".format(dir_sampled_frames))
        os.makedirs(dir_sampled_frames, exist_ok=True)

    video = EncodedVideo.from_path(path_video_to_analyze)
    start_time = 0
    # follow this post for clip duration https://towardsdatascience.com/using-pytorchvideo-for-efficient-video-understanding-24d3cd99bc3c
    clip_duration = int(video.duration)
    # print("clip_duration: ", clip_duration)
    end_sec = start_time + clip_duration

    print("end_sec: ", end_sec)
    video_data = video.get_clip(start_sec=start_time, end_sec=end_sec)
    print("type(video_data): ", type(video_data))
    print("video_data.keys(): ", video_data.keys())
    print("type(video_data['video']): ", type(video_data['video']))
    print("video_data['video'].size(): ", video_data['video'].size())
    print("apply transformations")
    video_data = apply_transformation(video_data=video_data,
                                      num_frames_to_sample=num_frames_to_sample,
                                      mean=mean,
                                      std=std,
                                      resize_to=resize_to,
                                      is_slowfast=is_slowfast)

    video_tensor = video_data["video"]

    if is_slowfast:
        # in this case video_tensor is a list [(C, N, H, W), (C, N, H, W)]
        video_tensor_0 = video_tensor[0]
        print("video_tensor_0.size(): ", video_tensor_0.size())
        video_tensor_1 = video_tensor[1]
        print("video_tensor_1.size(): ", video_tensor_1.size())

        # permute the color channel with the number of frame channel from (C, N, H, W) to (N, C, H, W)
        print("apply permutation")
        video_tensor_0 = video_tensor_0.permute(1, 0, 2, 3)
        video_tensor_1 = video_tensor_1.permute(1, 0, 2, 3)
        print("video_tensor_0.size(): ", video_tensor_0.size())
        print("video_tensor_1.size(): ", video_tensor_1.size())

        # save the collected frames
        save_frames(video_tensor_0, path_to_save=dir_sampled_frames_0)
        save_frames(video_tensor_1, path_to_save=dir_sampled_frames_1)
    else:
        # in this case i have a tensor (C, N, H, W)
        print("video_tensor.size(): ", video_tensor.size())
        # permute the color channel with the number of frame channel from (C, N, H, W) to (N, C, H, W)
        print("apply permutation")
        video_tensor = video_tensor.permute(1, 0, 2, 3)
        print("video_tensor.size(): ", video_tensor.size())
        save_frames(video_tensor, path_to_save = dir_sampled_frames)