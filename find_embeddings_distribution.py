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
    The script take a model, the directory where a stored the video, the list of classes to analyze.
    Fot the collected video the script calculate the embedding with the loaded model and generate the TSNE 2d distribution
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_dataset', type=str, help='path of the dataset')
    parser.add_argument('--path_csv_dataset', type=str, help='path of the csv where are stored the info of the video')
    parser.add_argument('--list_classes', type=str, default="", help='list of classes to use separed by commas')
    parser.add_argument('--name_model', type=str, help='name of the model to use to find the embedding')
    parser.add_argument('--dir_storing_results', type=str, help='directory where to store the results')
    parser.add_argument('--name_result_image', type=str, help='name of the image with the TSNE distribution')
    parser.add_argument('--num_frames_to_sample', type=int, help='number of frames to use')

    opt = parser.parse_args()

    path_dataset = opt.path_dataset
    path_csv_dataset = opt.path_csv_dataset
    list_classes = opt.list_classes
    name_model = opt.name_model
    dir_storing_results = opt.dir_storing_results
    name_result_image = opt.name_result_image
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

    # select the classis
    list_classes = list_classes.split(",")
    print("Selected classess: ", list_classes)
    df = df[df['CLASS'].isin(list_classes)]
    df.reset_index(drop=True, inplace=True)

    # iter over the videos
    embeddings_array = None
    labels_list = []
    classes_list = []
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
                #print("tensor_video[0].size(): ", tensor_video[0].size())
                #print("tensor_video[1].size(): ", tensor_video[1].size())
                tensor_video = [i.to(device)[None, ...] for i in tensor_video]
            else:
                #print("tensor_video.size(): ", tensor_video.size())
                tensor_video = tensor_video[None].to(device)

            # inference to extract the embedding
            with torch.no_grad():
                embedding = model(tensor_video)

            # stack the embedding
            if embeddings_array is None:
                embeddings_array = embedding.detach().cpu().numpy()
            else:
                embeddings_array = np.vstack((embeddings_array, embedding.detach().cpu().numpy()))

            labels_list.append(label)
            classes_list.append(classe)

    print("embeddings_array.shape: ", embeddings_array.shape)
    print("len(labels_list): ", len(labels_list))

    # applicazione tsne
    # costruisco un dataframe per collezionare le classi e le coordinate 2s di tsne
    number_of_classes = len(list_classes)
    #df_results = pd.DataFrame(data={'label':  np.array(labels_list)})
    df_results = pd.DataFrame(data={'classes': np.array(classes_list)})

    # applico t-sne
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(embeddings_array)
    df_results['axis_1'] = tsne_results[:, 0]
    df_results['axis_2'] = tsne_results[:, 1]

    plt.figure(figsize=(9, 9))
    # t-sne plot
    sns.scatterplot(
        x="axis_1", y="axis_2",
        hue="classes",
        palette=sns.color_palette("hls", n_colors=number_of_classes),
        data=df_results,
        s=50).set_title('2D t-SNE')
    #plt.legend(loc=1)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    path_result_image_1 = os.path.join(dir_storing_results, name_result_image)
    plt.savefig(path_result_image_1, bbox_inches='tight')