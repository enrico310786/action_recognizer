import torch
import os
import cv2
from pytorchvideo.data.encoded_video import EncodedVideo
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torchvision.transforms as T
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    UniformTemporalSubsample
)
import torchvision
from torchvision.transforms import (
    Compose,
    Lambda,
    Resize
)


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


class Dataset(torch.utils.data.Dataset):
    def __init__(self, df_dataset, data_cfg, dataset_path, is_train=False, is_slowfast=False) -> None:
        super().__init__()

        self.df_dataset = df_dataset
        self.dataset_path = dataset_path
        self.data_cfg = data_cfg
        self.num_frames_to_sample = self.data_cfg["num_frames_to_sample"]
        self.mean = [float(i) for i in self.data_cfg["mean"]]
        self.std = [float(i) for i in self.data_cfg["std"]]
        self.resize_to = self.data_cfg["resize_to"]
        self.permute_color_frame = self.data_cfg.get("permute_color_frame", 1.0) > 0.0
        self.is_train = is_train
        self.is_slowfast = is_slowfast
        self.alpha_slowfast = data_cfg.get("alpha_slowfast", None)

        # in this case i do not perform any runtime transformations. The data augmentation has been done offline
        if not self.is_slowfast:
            self.transform = ApplyTransformToKey(
                key="video",
                transform=Compose(
                    [
                        UniformTemporalSubsample(self.num_frames_to_sample),
                        Lambda(lambda x: x / 255.0),
                        Normalize(self.mean, self.std),
                        Resize((self.resize_to, self.resize_to))
                    ]
                ),
            )
        else:
            self.transform = ApplyTransformToKey(
                key="video",
                transform=Compose(
                    [
                        UniformTemporalSubsample(self.num_frames_to_sample),
                        Lambda(lambda x: x / 255.0),
                        Normalize(self.mean, self.std),
                        Resize((self.resize_to, self.resize_to)),
                        PackPathway(self.alpha_slowfast)
                    ]
                ),
            )

    def __len__(self):
        return len(self.df_dataset)

    def __getitem__(self, idx):
        video_path = os.path.join(self.dataset_path, self.df_dataset.iloc[idx]["PATH_VIDEO"])
        label = self.df_dataset.iloc[idx]["LABEL"]
        classe = self.df_dataset.iloc[idx]["ENG_CLASS"]

        video = EncodedVideo.from_path(video_path)
        start_time = 0
        # follow this post for clip duration https://towardsdatascience.com/using-pytorchvideo-for-efficient-video-understanding-24d3cd99bc3c
        clip_duration = int(video.duration)
        end_sec = start_time + clip_duration
        video_data = video.get_clip(start_sec=start_time, end_sec=end_sec)
        video_data = self.transform(video_data)
        video_tensor = video_data["video"]


        if self.permute_color_frame:
            video_tensor = torch.permute(video_tensor, (1, 0, 2, 3))

        return video_tensor, label, classe


class Dataset_V2(torch.utils.data.Dataset):
    def __init__(self, df_dataset, data_cfg, dataset_path, is_train=False, is_slowfast=False) -> None:
        super().__init__()

        self.df_dataset = df_dataset
        self.dataset_path = dataset_path
        self.data_cfg = data_cfg
        self.num_frames_to_sample = self.data_cfg["num_frames_to_sample"]
        self.mean = [float(i) for i in self.data_cfg["mean"]]
        self.std = [float(i) for i in self.data_cfg["std"]]
        self.resize_to = self.data_cfg["resize_to"]
        self.permute_color_frame = self.data_cfg.get("permute_color_frame", 1.0) > 0.0
        self.is_train = is_train
        self.is_slowfast = is_slowfast
        self.alpha_slowfast = data_cfg.get("alpha_slowfast", None)

        # in this case i do not perform any runtime transformations. The data augmentation has been done offline
        if not self.is_slowfast:
            self.transform = T.Compose([
                T.Normalize(mean=self.mean, std=self.std),
                T.Resize(self.resize_to),
                torchvision.ops.Permute([1, 0, 2, 3]),
                UniformTemporalSubsample(self.num_frames_to_sample),
                PackPathway(self.alpha_slowfast)
            ])
        else:
            self.transform = T.Compose([
                T.Normalize(mean=self.mean, std=self.std),
                T.Resize(self.resize_to),
                torchvision.ops.Permute([1, 0, 2, 3]),
                UniformTemporalSubsample(self.num_frames_to_sample)
            ])

    def __len__(self):
        return len(self.df_dataset)

    def __getitem__(self, idx):
        video_path = os.path.join(self.dataset_path, self.df_dataset.iloc[idx]["PATH_VIDEO"])
        label = self.df_dataset.iloc[idx]["LABEL"]
        classe = self.df_dataset.iloc[idx]["ENG_CLASS"]

        frames_tensor, _, _, _ = self.load_frames_video(video_path)
        frames_tensor = torch.stack(frames_tensor)
        print("Before transformation - frames_tensor.size(): ", frames_tensor.size())  # [N, C, H, W]
        frames_tensor = self.transform(frames_tensor)
        print("After transformation: frames_tensor.size(): ", frames_tensor.size())  # [C, N, H, W]

        if self.permute_color_frame:
            frames_tensor = torch.permute(frames_tensor, (1, 0, 2, 3))

        return frames_tensor, label, classe

    def load_frames_video(self, video_path):

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


def create_loaders(df_dataset_train, df_dataset_val, df_dataset_anomaly, data_cfg, dataset_path, batch_size, is_slowfast=False):

    # 1 - istanzio la classe dataset di train, val e test
    classification_dataset_train = Dataset_V2(df_dataset=df_dataset_train,
                                           data_cfg=data_cfg,
                                           dataset_path=dataset_path,
                                           is_train=True,
                                           is_slowfast=is_slowfast)
    classification_dataset_val = Dataset_V2(df_dataset=df_dataset_val,
                                         data_cfg=data_cfg,
                                         dataset_path=dataset_path,
                                         is_slowfast=is_slowfast)

    classification_dataset_anomaly = None
    if df_dataset_anomaly is not None:
        classification_dataset_anomaly = Dataset_V2(df_dataset=df_dataset_anomaly,
                                                 data_cfg=data_cfg,
                                                 dataset_path=dataset_path,
                                                 is_slowfast=is_slowfast)


    # 2 - istanzio i dataloader
    classification_dataloader_train = DataLoader(dataset=classification_dataset_train,
                                                 batch_size=batch_size,
                                                 shuffle=True,
                                                 num_workers=0,
                                                 drop_last=False)

    classification_dataloader_val = DataLoader(dataset=classification_dataset_val,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=0,
                                               drop_last=False)

    classification_dataloader_anomaly = None
    if classification_dataset_anomaly is not None:
        classification_dataloader_anomaly = DataLoader(dataset=classification_dataset_anomaly,
                                                       batch_size=batch_size,
                                                       shuffle=True,
                                                       num_workers=0,
                                                       drop_last=False)

    return classification_dataloader_train, classification_dataloader_val, classification_dataloader_anomaly