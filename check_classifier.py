import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import math
import numpy as np
from sklearn import metrics
from sklearn.metrics import accuracy_score


dir_save_results = "/home/enrico/Dataset/Actions/test_actions/results/timesformer/avvitare/v1"
path_dataset_errors = os.path.join(dir_save_results, "reconstruction_error_dataset.csv")
df_errors = pd.read_csv(path_dataset_errors)
name_plot_log_errors = "log_error_distribution_grouped_by_classes.png"
name_confusion_matrix = "classification_confusion_matrix.png"
name_report = "report_classification.txt"


TARGET_CLASS = "avvitare"
COMPETITIVE_CLASS = "svitare"


class2label = {TARGET_CLASS: 0, COMPETITIVE_CLASS: 1}

#######
# NORMAL SCALE
#######


# TRAIN ERROR_POSITIVE NORMAL SCALE

print("-------------------------------------------------------------------")
print("RECONTRUCTION_ERROR DISTRIBUTION GROUPED BY CLASS")
print("")
desc_grouped = df_errors[df_errors['TYPE_DATASET'] == 'ANOMALY'].groupby('CLASS')["RECONSTRUCTION_ERROR"].describe()[['mean', 'min', '75%', 'max']]
print("RECONTRUCTION_ERROR distrbution for dataset ANOMALY: ")
print(desc_grouped)
print("-------------------------------------------------------------------")


#######
# LOG SCALE
#######

# passa a scala logaritmica
df_errors['LOG_ERROR_RECONSTRUCTION_ERROR'] = df_errors['RECONSTRUCTION_ERROR'].map(lambda x: math.log(x))

####### TRAIN ERROR_POSITIVE

print("-------------------------------------------------------------------")
print("LOG_ERROR_RECONSTRUCTION_ERROR DISTRIBUTION GROUPED BY CLASS")
print("")
desc_grouped = df_errors[df_errors['TYPE_DATASET'] == 'ANOMALY'].groupby('CLASS')["LOG_ERROR_RECONSTRUCTION_ERROR"].describe()[['mean', 'std', 'min', '75%', 'max']]
print("LOG_ERROR_RECONSTRUCTION_ERROR distrbution for dataset ANOMALY: ")
print(desc_grouped)
print("-------------------------------------------------------------------")

# boxplot
print("Plot log error distribution grouped by dataset and actors")
plt.figure(figsize=(15, 15))
sns.boxplot(data=df_errors[df_errors['TYPE_DATASET'] == 'ANOMALY'], x="CLASS", y="LOG_ERROR_RECONSTRUCTION_ERROR")
plt.xticks(rotation=30, ha='right', rotation_mode='anchor')
plt.title('LOG Reconstruction error grouped by classes', fontsize=12)

plt.savefig(os.path.join(dir_save_results, name_plot_log_errors))


print("-------------------------------------------------------------------")
print("-------------------------------------------------------------------")


# ANALISI CON SOGLIA NORMALE

th_min_error = 0.907211
th_max_error = 1.280789

print("th_min_error: ", th_min_error)
print("th_max_error: ", th_max_error)

num_target_class = len(df_errors[df_errors['CLASS'] == TARGET_CLASS])
print('Numero item target class {}: {}'.format(TARGET_CLASS, num_target_class))
num_competitive_class = len(df_errors[df_errors['CLASS'] == TARGET_CLASS])
print('Numero item competitive class {}: {}'.format(COMPETITIVE_CLASS, num_competitive_class))

print("")
# numero di item della classe target al di sotto del valore minimo della classe competitiva
num_under_th_min_error = len(df_errors[(df_errors['CLASS'] == TARGET_CLASS) & (df_errors['RECONSTRUCTION_ERROR'] <= th_min_error)])
print("Numero di elementi della classe target '{}' sotto il valore minimo della classe '{}': {}".format(TARGET_CLASS, COMPETITIVE_CLASS, num_under_th_min_error))

# numero di item della classe competitiva al di sotto del valore massimo della classe target
num_under_th_max_error = len(df_errors[(df_errors['CLASS'] == COMPETITIVE_CLASS) & (df_errors['RECONSTRUCTION_ERROR'] <= th_max_error)])
print("Numero di elementi della classe '{}' sotto il valore massimo della classe target'{}': {}".format(COMPETITIVE_CLASS, TARGET_CLASS, num_under_th_max_error))


#####################
list_classes = [TARGET_CLASS, COMPETITIVE_CLASS]

df_reduced = df_errors[df_errors['CLASS'].isin(list_classes)].reset_index(drop=True)
df_reduced['LABEL'] = df_reduced['CLASS'].map(class2label)

true_label_list = []
pred_label_list = []

for index, row in df_reduced.iterrows():
    classe = row['CLASS']
    label = row['LABEL']
    error = row['RECONSTRUCTION_ERROR']

    true_label_list.append(label)

    if error < th_max_error:
        pred_label_list.append(class2label[TARGET_CLASS])
    else:
        pred_label_list.append(class2label[COMPETITIVE_CLASS])

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
fig.savefig(os.path.join(dir_save_results, name_confusion_matrix))

## Save report in a txt
target_names = list(class2label.keys())
cr = metrics.classification_report(true_label_list, pred_label_list, target_names=target_names)
f = open(os.path.join(dir_save_results, name_report), 'w')
f.write('Title\n\nClassification Report\n\n{}'.format(cr))
f.close()