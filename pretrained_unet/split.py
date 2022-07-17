import splitfolders

input_folder = "../Datasets/256_Water/"
output_folder = "../Datasets/shoreline_ready_data/"
train_ratio = .5
val_ratio =.5

splitfolders.ratio(
        input_folder, 
        output = output_folder, 
        seed = 42, 
        ratio=(train_ratio, val_ratio),
        group_prefix = None
    )