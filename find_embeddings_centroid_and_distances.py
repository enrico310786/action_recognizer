import os
import torch
import argparse
from sklearn.manifold import TSNE
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


from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
import seaborn as sns

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
    The script take a model, the directory where are stored the videos, the list of classes to analyze.
    For the collected video the script calculates the embedding with the loaded model.
    Is then calculated the centroid of each class and
    1) for each centroid are calculated the distances of all the embeddings grouped by class and check the corresponding distributions
    2) for each centroid I consider the farthest points of the same class of the centroids and counts the number of embedding inside that hypersphere gouped by class
    3) for each centroid I consider the closest points not belonging to the same class of the centroids and counts the number of embedding inside that hypersphere gouped by class
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_dataset', type=str, help='path of the dataset')
    parser.add_argument('--path_csv_dataset', type=str, help='path of the csv where are stored the info of the video')
    parser.add_argument('--name_model', type=str, help='name of the model to use to find the embedding')
    parser.add_argument('--dir_storing_results', type=str, help='directory where to store the results')
    parser.add_argument('--num_frames_to_sample', type=int, help='number of frames to use')

    opt = parser.parse_args()

    path_dataset = opt.path_dataset
    path_csv_dataset = opt.path_csv_dataset
    name_model = opt.name_model
    dir_storing_results = opt.dir_storing_results
    num_frames_to_sample = int(opt.num_frames_to_sample)

    print("create the directory: {}".format(dir_storing_results))
    os.makedirs(dir_storing_results, exist_ok=True)

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

    # load the csv as pandas dataframe
    df = pd.read_csv(path_csv_dataset)

    # build the class2label dict
    class2label = {}
    for index, row in df.iterrows():
        class_name = row["ENG_CLASS"]
        label = row["LABEL"]

        if class_name not in class2label:
            class2label[class_name] = label

    #sort the value of the label
    class2label = dict(sorted(class2label.items(), key=lambda item: item[1]))
    label2class = {v: k for (k, v) in class2label.items()}
    list_classes = list(class2label.keys())
    print("class2label: ", class2label)
    print("label2class: ", label2class)
    print("list_classes: ", list_classes)

    # iter over the videos
    embeddings_array = None
    class_labels = []
    for index, row in df.iterrows():

        relative_path = row['PATH_VIDEO']
        path_video = os.path.join(path_dataset, relative_path)
        classe = row['ENG_CLASS']
        label = row['LABEL']

        # load the video
        if not os.path.exists(path_video):
            continue
        else:
            tensor_video = load_video(path_video, permute_color_frame, transform)

            # add one extra dimension to make the inference
            if is_slowfast:
                tensor_video = [i.to(device)[None, ...] for i in tensor_video]
            else:
                tensor_video = tensor_video[None].to(device)

            # inference to extract the embedding
            with torch.no_grad():
                embedding = model(tensor_video)

            # stack the embedding
            if embeddings_array is None:
                embeddings_array = embedding.detach().cpu().numpy()
            else:
                embeddings_array = np.vstack((embeddings_array, embedding.detach().cpu().numpy()))

            class_labels.append(label)

    class_labels = np.array(class_labels)

    print("embeddings_array.shape: ", embeddings_array.shape)
    print("class_labels.shape: ", class_labels.shape)

    # for each class i construct the centroid of the corresponding embeddings
    embedding_centroids = []
    # calculate the centroid of the embedding vectors for each class. Iter over the label of each class from 0 to N
    for idx in label2class.keys():
        filter_idxs = np.where(class_labels == idx)[0]
        print("filter_idxs: ", filter_idxs)
        embeddings_array_filtered = np.take(embeddings_array, filter_idxs, 0)
        centroid = np.mean(embeddings_array_filtered, axis=0)
        embedding_centroids.append(centroid)
    embedding_centroids = np.array(embedding_centroids)

    print("embedding_centroids.shape: ", embedding_centroids.shape)

    # iter over the centroid
    for idx in label2class.keys():
        print("select the {}-th centroid for the class '{}'".format(str(idx), label2class[idx]))
        centroid = embedding_centroids[idx]

        # for the current centroid initialize a dataframe to collect the distance
        df_distribution = pd.DataFrame(columns=['LABEL_EMBEDDING', 'CLASS_EMBEDDING', 'EMBEDDING_DISTANCE'])

        # iter over the classes label
        for label in label2class.keys():
            filter_idxs = np.where(class_labels == label)[0]
            embeddings_array_filtered = np.take(embeddings_array, filter_idxs, 0)

            # iter over the embeddings_array_filtered to find the distance from the current centroid
            for emb in embeddings_array_filtered:
                emb_dist = np.linalg.norm(emb - centroid)
                df_distribution = df_distribution.append({'CLASS_EMBEDDING': label2class[label],
                                                          'LABEL_EMBEDDING': label,
                                                          'EMBEDDING_DISTANCE': emb_dist}, ignore_index=True)

        # plot and save the boxplot
        print("Plot the distance distribution from the centroid of the class: {}".format(label2class[idx]))
        name_file = name_model + "_emb_dist_centroid_" + label2class[idx] + ".png"
        title = "Embedding distances from the centroid of class '" + label2class[idx] + "'"
        plt.figure(figsize=(12, 12))
        sns.boxplot(data=df_distribution, x="CLASS_EMBEDDING", y="EMBEDDING_DISTANCE").set(title=title)
        plt.xticks(rotation=30, ha='right', rotation_mode='anchor')
        plt.savefig(os.path.join(dir_storing_results, name_file))

        # analyze the hypersphere of the farthest point
        # 1. find the farthest point of the same class of the centroid
        df_filtered = df_distribution[df_distribution['CLASS_EMBEDDING'] == label2class[idx]]
        max_in_distance = df_filtered["EMBEDDING_DISTANCE"].max()
        # 2. find the number of points grouped by class in the hypersphere
        df_farthest = df_distribution[df_distribution["EMBEDDING_DISTANCE"] < max_in_distance]
        # boxplot
        plt.figure(figsize=(12, 12))
        sns.countplot(x=df_farthest["CLASS_EMBEDDING"])
        plt.xticks(rotation=30, ha='right', rotation_mode='anchor')
        name_file = name_model + "_num_emb_farthest_" + label2class[idx] + ".png"
        title = "Number of samples inside the farthest point from the centroid of class '" + label2class[idx] + "'"
        plt.title(title, fontsize=16)
        plt.savefig(os.path.join(dir_storing_results, name_file))

        # analyze the hypersphere of the closest point belonging to a class different from that of the centroid
        # 1. find the closest point of the other classes of the centroid
        df_filtered = df_distribution[df_distribution['CLASS_EMBEDDING'] != label2class[idx]]
        min_out_distance = df_filtered["EMBEDDING_DISTANCE"].min()
        min_distance_row = df_filtered[df_filtered['EMBEDDING_DISTANCE'] == min_out_distance].iloc[0]
        min_class = min_distance_row['CLASS_EMBEDDING']
        print("Min value excluding '{}': {} - Class of min value: {}".format(label2class[idx], min_out_distance, min_class))

        # 2. find the number of points grouped by class in the hypersphere
        df_closest = df_distribution[df_distribution["EMBEDDING_DISTANCE"] < min_out_distance]
        if df_closest.empty:
            print("The dataframe df_closest is empty: no point inside the min sphere")
        else:
            # boxplot
            plt.figure(figsize=(12, 12))
            sns.countplot(x=df_closest["CLASS_EMBEDDING"])
            plt.xticks(rotation=30, ha='right', rotation_mode='anchor')
            name_file = name_model + "_num_emb_closest_" + label2class[idx] + ".png"
            title = "Number of samples inside the closest point from the centroid of class " + label2class[idx]
            plt.title(title, fontsize=16)
            plt.savefig(os.path.join(dir_storing_results, name_file))

        print("-----------------------------------------------------------------------")