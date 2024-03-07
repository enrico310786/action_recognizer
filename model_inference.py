import argparse
import os
import yaml
import cv2
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    UniformTemporalSubsample
)
import torchvision.transforms as T
import torchvision

from torchvision.transforms import (
    Compose,
    Lambda,
    Resize
)
from pytorchvideo.data.encoded_video import EncodedVideo
import torch
import torch.nn as nn
from model import SpaceTimeAutoencoder, find_last_checkpoint_file
import time

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def load_config(path) -> dict:
    """
    Loads and parses a YAML configuration file.

    :param path: path to YAML configuration file
    :return: configuration dictionary
    """
    with open(path, "r", encoding="utf-8") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg


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


def load_frames_video(video_path):

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
        # here is the normalization between 0 and 1
        frame = T.ToTensor()(frame)
        frame_list.append(frame)
    cap.release()

    video = EncodedVideo.from_path(video_path)
    return frame_list, len(frame_list), fps, int(video.duration)


def load_clip_video(path_video):
    video = EncodedVideo.from_path(path_video)
    start_time = 0
    # follow this post for clip duration https://towardsdatascience.com/using-pytorchvideo-for-efficient-video-understanding-24d3cd99bc3c
    clip_duration = int(video.duration)
    end_sec = start_time + clip_duration
    video_data = video.get_clip(start_sec=start_time, end_sec=end_sec)
    return video_data


if __name__ == '__main__':

    '''
    The script executes the inference of the SpaceTimeAutoencoder over a video clip. 
    We can load the video in two way. As a .mp4 video clip or as a train of frames
    
    Whit this script we can note that the two ways of loading a video are different. 
    Working with the load_frames_video allow to load all the frames. This should be used also in the dataloader
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_config_file', type=str, help='Path of the config file to use')
    parser.add_argument('--path_video', type=str, help='Path to the clip or the directory with frames')
    opt = parser.parse_args()

    # 1 - load config file and the model
    path_config_file = opt.path_config_file
    path_video = opt.path_video
    print("path_config_file: {}".format(path_config_file))
    cfg = load_config(path_config_file)
    print("path_video: ", path_video)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: ", device)

    # 2 - build the model
    print("Build the model")
    model = SpaceTimeAutoencoder(model_config=cfg["model"]).to(device)
    checkpoint_dir = os.path.join(cfg["model"]['saving_dir_experiments'], cfg["model"]['saving_dir_model'])
    path_checkpoint = find_last_checkpoint_file(checkpoint_dir, use_best_checkpoint=True)
    print("Load the chekpoint at '{}'".format(path_checkpoint))
    checkpoint = torch.load(path_checkpoint, map_location=torch.device(device))
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model = model.eval()

    # 3 - set the parameters
    is_slowfast = False
    if cfg["model"]['name_time_model'] == "3d_slowfast":
        is_slowfast = True
        print("Set is_slowfast to True")

    mean = [float(i) for i in cfg["data"]["mean"]]
    std = [float(i) for i in cfg["data"]["std"]]
    num_frames_to_sample = cfg["data"]["num_frames_to_sample"]
    alpha_slowfast = cfg["data"].get("alpha_slowfast", None)
    resize_to = cfg["data"]["resize_to"]
    permute_color_frame = cfg["data"].get("permute_color_frame", 1.0) > 0.0  # this is true just for the timesformer

    print("num_frames_to_sample: ", num_frames_to_sample)

    # 4- set the error function
    err_function = nn.MSELoss(reduction='sum')

    # 5 - set the transformations
    # transformation for clip videos
    if is_slowfast:
        transform_video = ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(num_frames_to_sample),
                    Lambda(lambda x: x / 255.0),
                    Normalize(mean, std),
                    Resize((resize_to, resize_to)),
                    PackPathway(alpha_slowfast)
                ]
            ),
        )
    else:
        transform_video = ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(num_frames_to_sample),
                    Lambda(lambda x: x / 255.0),
                    Normalize(mean, std),
                    Resize((resize_to, resize_to)),
                ]
            ),
        )

    # transformation for frames
    if is_slowfast:
        transform_frames = T.Compose([
            T.Normalize(mean=mean, std=std),
            T.Resize(resize_to),
            torchvision.ops.Permute([1, 0, 2, 3]),
            UniformTemporalSubsample(num_frames_to_sample),
            PackPathway(alpha_slowfast)
        ])
    else:
        transform_frames = T.Compose([
            T.Normalize(mean=mean, std=std),
            T.Resize(resize_to),
            torchvision.ops.Permute([1, 0, 2, 3]),
            UniformTemporalSubsample(num_frames_to_sample),
        ])

    # 6 - load the video
    video_tensor = load_clip_video(path_video)
    print("Before transformation - video_tensor['video'].size(): ", video_tensor['video'].size()) # [N, C, H, W]
    video_tensor = transform_video(video_tensor)
    video_tensor = video_tensor["video"]
    print("After transformation: video_tensor.size(): ", video_tensor.size())  # [C, N, H, W]

    # this part simulate the real condition where i have not an mp4 but a train of frames. Thus i load the video as a train of frames
    frames_tensor, _, _, _ = load_frames_video(path_video)
    frames_tensor = torch.stack(frames_tensor)
    print("Before transformation - frames_tensor.size(): ", frames_tensor.size()) # [N, C, H, W]
    frames_tensor = transform_frames(frames_tensor)
    print("After transformation: frames_tensor.size(): ", frames_tensor.size()) # [C, N, H, W]

    # 7 - permute color if permute_color_frame
    if permute_color_frame:
        # the timesformer gets first the number of frames then the number of channels
        video_tensor = torch.permute(video_tensor, (1, 0, 2, 3))  # [N, C, H, W]
        frames_tensor = torch.permute(frames_tensor, (1, 0, 2, 3))  # [N, C, H, W]

    # 7 - augment one dimension
    if is_slowfast:
        video_tensor = [item[None].to(device) for item in video_tensor]
        frames_tensor = [item[None].to(device) for item in frames_tensor]
        print("torch.equal(video_tensor[0], frames_tensor[0]): ", torch.equal(video_tensor[0], frames_tensor[0]))
        print("torch.equal(video_tensor[1], frames_tensor[1]): ", torch.equal(video_tensor[1], frames_tensor[1]))
    else:
        video_tensor = video_tensor[None].to(device)
        frames_tensor = frames_tensor[None].to(device)
        print("torch.equal(video_tensor, frames_tensor): ", torch.equal(video_tensor, frames_tensor))

    print("------------------------------------")
    print("INFERENCE WITH VIDEO TENSOR")
    print("------------------------------------")

    # make inference for video
    print("start inference")
    start_time = time.time()
    with torch.no_grad():
        embeddings, reconstructed_embeddings, _ = model(video_tensor)

    # calculate the error
    error = err_function(reconstructed_embeddings[0], embeddings[0]).item()
    end_time = time.time()

    print("error: ", error)
    print("inference_time: ", end_time - start_time)

    print("------------------------------------")
    print("INFERENCE WITH FRAMES TENSOR")
    print("------------------------------------")

    # make inference for video
    print("start inference")
    start_time = time.time()
    with torch.no_grad():
        embeddings, reconstructed_embeddings, _ = model(frames_tensor)

    # calculate the error
    error = err_function(reconstructed_embeddings[0], embeddings[0]).item()
    end_time = time.time()

    print("error: ", error)
    print("inference_time: ", end_time - start_time)






