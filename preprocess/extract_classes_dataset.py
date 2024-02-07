import argparse
import os
import pandas as pd


if __name__ == "__main__":
    '''
    Script to generate a csv to collect the path of each video and its corresponding class
    '''

    ita2engclass = {"avvitare": "screw",
                    "svitare": "unscrew",
                    "avvitare_orario": "screw_1234",
                    "avvitare_antiorario": "screw_1432",
                    "avvitare_incrociato_2413": "screw_2413",
                    "avvitare_incrociato_1342": "screw_1342",
                    "avvita_brugola_1": "screw_allenkey_1",
                    "svita_brugola_1": "unscrew_allenkey_1",
                    "avvita_brugola_2": "screw_allenkey_2",
                    "svita_brugola_2": "unscrew_allenkey_2"}

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_dataset', type=str, help='directory of the dataset')

    opt = parser.parse_args()
    dir_dataset = opt.dir_dataset

    df = pd.DataFrame(columns=['CLASS', 'LABEL', 'PATH_VIDEO'])
    class2label = {}
    counter = 0

    # Iterate in the subdirs of the dataset. Each subdir is a class
    for _, action_dir, _ in os.walk(dir_dataset):
        for action in action_dir:
            path_action = os.path.join(dir_dataset, action)
            CHECK_FOLDER = os.path.isdir(path_action)

            if CHECK_FOLDER:
                print("ACTION: ", action)

                if action not in class2label.keys():
                    class2label[action]=counter
                    counter += 1

                video_list = os.listdir(path_action)
                print("Number of video: {}".format(len(video_list)))

                for name_video in video_list:
                    path_video = os.path.join(path_action, name_video)
                    relative_path = ("/").join(path_video.split("/")[-2:])

                    df = df.append({'CLASS': action,
                                    'PATH_VIDEO': relative_path,
                                    'LABEL': class2label[action]}, ignore_index=True)

    print("Add eng classes")
    df['ENG_CLASS'] = df['CLASS'].map(ita2engclass)
    print("save the csv")
    df.to_csv("dataset_info.csv", index=False)

    print("****************************************************************")

    print("DATASET FEATURES")
    print(df.info())
    print("")
    print("CLIP DISTRIBUTION BY CLASS")
    print("")
    desc_grouped = df[['ENG_CLASS']].value_counts()
    print(desc_grouped)
    print("")
    print("class2label: ", class2label)
