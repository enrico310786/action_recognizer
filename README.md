# action_recognizer

# Categories

avvitare_incrociato_2413,
avvitare_incrociato_1342,
avvitare_antiorario,
svitare,
avvita_brugola_1,
svita_brugola_1,
avvita_brugola_2,
svita_brugola_2,
avvitare,
avvitare_orario




## Embeddings distribution with TSNE

python find_embeddings_distribution.py --path_dataset /mnt/efs/enrico/action_recognizer/dataset/square_256_mp4 --path_csv_dataset /mnt/efs/enrico/action_recognizer/dataset/square_256_mp4/dataset_info.csv --list_classes avvita_brugola_1,svita_brugola_1 --name_model timesformer --dir_storing_results /mnt/efs/enrico/action_recognizer/results/avvita_svita_brugola_1 --name_result_image tf_128 --num_frames_to_sample 128

## Embeddings centroids

python find_embeddings_centroid_and_distances.py --path_dataset /mnt/efs/enrico/action_recognizer/dataset/square_256_mp4 --path_csv_dataset /mnt/efs/enrico/action_recognizer/dataset/square_256_mp4/dataset_info.csv --name_model timesformer --dir_storing_results /mnt/efs/enrico/action_recognizer/results/distances_tf_8 --num_frames_to_sample 8