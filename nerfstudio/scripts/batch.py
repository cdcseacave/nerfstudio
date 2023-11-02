import datetime
import os
import gc

# User inputs that usually stay the same
path_to_polycam_repo = "~/workspace/polycam/"
base_data_dir = "~/workspace/data/"
base_out_dir = "~/workspace/data/batch_out/"
top_datasets = ["CellTower2", "IcyTree", "Castle", "Cockpit", "UnderwaterCar", "Shipwreck", "StatuePond", "Smoker", "ElliottCar", "OilRig"]
secondary_datasets = ["Crystal", "PumpkinTree", "ChickenCoop", "ChrisCar", "PricklyYucca", "Semi"]
max_dimension = 2048
zip_output = True

# User inputs to change
datasets_run = ["OilRig", "Castle", "Cockpit", "UnderwaterCar", "Shipwreck", "StatuePond", "ElliottCar", "Crystal", "ChrisCar", "Semi", "PricklyYucca", "IcyTree"] # top_datasets + secondary_datasets
training_options = ["--pipeline.model.sh_degree=3", "--pipeline.model.sh_degree=2"] # ["--pipeline.model.init-pts-hemisphere=False", "--pipeline.model.init-pts-hemisphere=True"] # Note: changing n_iterations here doesn't work


def GetDownsizeDir(dataset):
    return os.path.expanduser(base_data_dir + str(max_dimension) + "/" + dataset)


def DownsizeCopyDataset(dataset):
    dataset_dir0 = os.path.expanduser(base_data_dir + dataset)
    dataset_dir = GetDownsizeDir(dataset)
    if os.path.isdir(dataset_dir):
        return 0
    im_dir0 = dataset_dir0 + "/input"
    im_dir = dataset_dir + "/input"
    os.makedirs(dataset_dir)
    os.makedirs(im_dir)
    im_files = [os.path.join(im_dir0,f) for f in os.listdir(im_dir0) if os.path.isfile(os.path.join(im_dir0,f))]
    cnt = 0
    import cv2
    for im_file in im_files:
        if not im_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        im = cv2.imread(im_file)
        height, width, channels = im.shape
        max_dim = max(height, width)
        scale = max_dimension / max_dim
        if scale < 1.0:
            im = cv2.resize(im, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        im_out_file = im_dir + "/" + os.path.basename(im_file)
        cv2.imwrite(im_out_file, im)
        cnt += 1
        print("Copied image " + str(cnt) + "/" + str(len(im_files)) + " from dataset: " + dataset)
    print(str(cnt) + " images copied & downsized\n")
    return 1


def RunDataset(dataset, config, out_dir):
    dataset_dir = GetDownsizeDir(dataset)

    # Run colmap if needed
    colmap_done_dir = dataset_dir + "/sparse/0"
    if not os.path.isdir(colmap_done_dir):
        command = "ns-colmap --path " + dataset_dir
        print("\n" + "Call colmap command:\n" + command + "\n")
        os.system(command)

    # Run gaussian splatting training
    os.system("ns-train gaussian-splatting --data " + dataset_dir + " --output-dir " + dataset_dir + " --viewer.quit-on-train-completion=True " + config)

    # Find the gaussian splatting model directory name
    model_dir0 = dataset_dir + "/" + dataset + "/gaussian-splatting"
    model_dir = max([os.path.join(model_dir0,d) for d in os.listdir(model_dir0)], key=os.path.getmtime) # Find the most recently created subdirectory

    # Export the model
    os.system("ns-export gaussian-splat --output-dir " + model_dir + " --load-config " + model_dir + "/config.yml")

    # Copy the model to the output directory
    model_file = model_dir + "/point_cloud.ply"
    out_file = out_dir + "/" + dataset + ".ply"
    os.system("cp " + model_file + " " + out_file)

    # Convert .ply -> .splat (follow instructions in main polycam repo to run: yarn setup)
    cwd = os.getcwd()
    os.chdir(os.path.expanduser(path_to_polycam_repo))
    os.system("yarn gulp convert-ply-to-splat --ply " + out_file)
    os.chdir(cwd)

    # Clear memory
    gc.collect()
    return model_dir


if __name__ == "__main__":
    # Copy & downsize all images
    cnt = 0
    for dataset in datasets_run:
        print("Dataset " + str(cnt+1) + "/" + str(len(datasets_run)))
        cnt += DownsizeCopyDataset(dataset)

    if cnt == 0: # For some reason import cv2 messes up colmap
        # Create a new folder for each run based on current time
        now = datetime.datetime.now() # Get the current time
        folder_name = now.strftime("%Y-%m-%d_%H-%M-%S") # Create a folder name based on the current time
        out_dir = os.path.expanduser(base_out_dir + folder_name)
        os.makedirs(out_dir)

        # For each configuration
        n_configs = len(training_options)
        for i in range(n_configs):
            out_dir_i = out_dir + "/" + str(i)
            os.makedirs(out_dir_i)

            # Run each dataset
            for dataset in datasets_run:
                model_dir = RunDataset(dataset, training_options[i], out_dir_i)
            
            # Copy over the configuration file
            os.system("cp " + model_dir + "/config.yml " + out_dir_i + "/config.yml")

        # Zip the output folder
        if zip_output:
            zip_dir = out_dir + ".zip"
            exit_code = os.system("zip -r " + zip_dir + " " + out_dir)
            print("\n\nCommand to copy this dataset:\nscp ubuntu@141.148.59.221:" + zip_dir + " splats.zip")