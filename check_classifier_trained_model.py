import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import math
import numpy as np
from sklearn import metrics
from sklearn.metrics import accuracy_score


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def assign_label(value, reference_class):
    if value == reference_class:
        return 0
    else:
        return 1


if __name__ == "__main__":

    '''
    Script to analyze the trained model starting fromn the csv with the error distribution
    Once setting the theshorld, the reference class and if using the log scale or not, it generate the confusion matrix
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_results', type=str, help='Directory where are stored the results')
    parser.add_argument('--theshorld', type=float, help='theshorld. If error is lower then theshorld then is the desired class, otherwise is another class')
    parser.add_argument("--log_scale", type=str2bool, nargs='?', const=True, default=False, help="Use the log scale to evaluate the errors")
    parser.add_argument('--reference_class', type=str, help='Class under analysis')
    parser.add_argument('--competitive_class', type=str, help='Class against the target class')

    opt = parser.parse_args()
    dir_results = opt.dir_results
    theshorld = opt.theshorld
    log_scale = opt.log_scale
    reference_class = opt.reference_class
    competitive_class = opt.competitive_class

    if theshorld is not None:
        theshorld = float(theshorld)

    print("Thershold: ", theshorld)
    print("log_scale: ", log_scale)
    print("reference_class: ", reference_class)
    print("competitive_class: ", competitive_class)

    plot_log_class_pos = "error_distribution_class_log.png"
    log_labeled_csv = "log_label_error_distribution.csv"

    # load the dataset with the results
    path_dataset_results = os.path.join(dir_results, "reconstruction_error_dataset.csv")
    df_distribution = pd.read_csv(path_dataset_results)

    #######
    # NORMAL SCALE
    #######

    print("RECONSTRUCTION_ERROR DISTRIBUTION GROUPED BY CLASS")
    print("")
    desc_grouped = df_distribution.groupby('ENG_CLASS')[
        "RECONSTRUCTION_ERROR"].describe()[['mean', 'std', 'min', '75%', 'max']]
    print(desc_grouped)
    print("")

    print("-------------------------------------------------------------------")
    print("-------------------------------------------------------------------")

    #######
    # LOG SCALE
    #######

    df_distribution['LOG_RECONSTRUCTION_ERROR'] = df_distribution['RECONSTRUCTION_ERROR'].map(lambda x: math.log(x))

    print("LOG_RECONSTRUCTION_ERROR DISTRIBUTION GROUPED BY CLASS")
    print("")
    desc_grouped = df_distribution.groupby('ENG_CLASS')[
        "LOG_RECONSTRUCTION_ERROR"].describe()[['mean', 'std', 'min', '75%', 'max']]
    print(desc_grouped)
    print("")

    # boxplot
    plt.figure(figsize=(15, 15))
    sns.boxplot(data=df_distribution, x="ENG_CLASS", y="LOG_RECONSTRUCTION_ERROR")
    plt.xticks(rotation=45)
    plt.title('val set', fontsize=12)
    plt.savefig(os.path.join(dir_results, plot_log_class_pos))

    print("-------------------------------------------------------------------")
    print("-------------------------------------------------------------------")


    # generate the label coresponding to the class
    class2label = {reference_class: 0, competitive_class: 1}
    list_classes = [reference_class, competitive_class]
    # select just the rows with the selected classes
    df_reduced = df_distribution[df_distribution['ENG_CLASS'].isin(list_classes)].reset_index(drop=True)
    df_reduced['LABEL'] = df_reduced['ENG_CLASS'].map(class2label)
    print("class2label: ", class2label)
    df_distribution.to_csv(os.path.join(dir_results, log_labeled_csv), index=False)

    if reference_class is not None and theshorld is not None:
        print("Classification analysis")

        true_label_list = []
        pred_label_list = []
    
        for index, row in df_reduced.iterrows():
            classe = row['ENG_CLASS']
            label = row['LABEL']

            if log_scale:
                error = row['LOG_RECONSTRUCTION_ERROR']
            else:
                error = row['RECONSTRUCTION_ERROR']

            true_label_list.append(label)
    
            if error < theshorld:
                pred_label_list.append(0)
            else:
                pred_label_list.append(1)
    
        true_label_list = np.array(true_label_list)
        pred_label_list = np.array(pred_label_list)
        print('true_label_list.shape: ', true_label_list.shape)
        print('pred_label_list.shape: ', pred_label_list.shape)
    
        print('Accuracy: ', accuracy_score(true_label_list, pred_label_list))
        print(metrics.classification_report(true_label_list, pred_label_list))
    
        ## Plot confusion matrix
        cm = metrics.confusion_matrix(true_label_list, pred_label_list)
    
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.set(font_scale=1.3)  # Adjust to fit
        sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap=plt.cm.Blues,
                    cbar=False)
        ax.set(xlabel="Pred", ylabel="True", xticklabels=class2label.keys(),
               yticklabels=class2label.keys())
        # plt.yticks(fontsize=10, rotation=0)
        plt.yticks(fontsize=11, rotation=-30, ha='right', rotation_mode='anchor')
        # plt.xticks(fontsize=10, rotation=90)
        plt.xticks(fontsize=11, rotation=30, ha='right', rotation_mode='anchor')
        name_confusion_matrix = str(theshorld) + "_confusion_matrix.png"
        fig.savefig(os.path.join(dir_results, name_confusion_matrix))

        ## Save report in a txt
        target_names = list(class2label.keys())
        cr = metrics.classification_report(true_label_list, pred_label_list, target_names=target_names)
        f = open(os.path.join(dir_results, "classification_report"), 'w')
        f.write('Title\n\nClassification Report\n\n{}'.format(cr))
        f.close()