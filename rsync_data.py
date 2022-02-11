import os
import argparse

rsync_base = "rsync -azhP "
hab_lab_local_path = "/Users/alexclegg/Documents/dev2/scott_ant_task/habitat-lab/"
hab_lab_remote_path = "alexclegg@devfair:~/scott_ant_dir/habitat-lab/"
sub_directories = {
    "archive": ["data/ant_exp_archive/"],
    "videos": ["data/videos/"],
    "results": ["data/checkpoints/", "data/tb/"],
    "hab_lab_code": ["configs/", "habitat/", "habitat_baselines/", "run_exp.py", "rsync.py"],
    "config_only": ["habitat_baselines/config/ant_v2/"],
}

directions = ["up", "down"]
types = ["results", "code", "config"]

def rsync_remote(direction="down", type="results"):
    assert direction in directions
    assert type in types

    full_command = rsync_base
    local_path = hab_lab_local_path
    remote_path = hab_lab_remote_path
    target_sub_directories = sub_directories["results"]
    if type == "code":
        target_sub_directories = sub_directories["hab_lab_code"]
    if type == "config":
        target_sub_directories = sub_directories["config_only"]

    for sub_dir in target_sub_directories:
        final_command = full_command + remote_path + sub_dir + " " + local_path + sub_dir
        if direction == "up":
            final_command = full_command + local_path + sub_dir + " " + remote_path + sub_dir
        print(final_command)
        os.system(final_command)

def clean_data():
    rm_base = "rm -r "
    full_command = rm_base
    target_sub_directories = sub_directories["results"] + sub_directories["videos"]
    for sub_dir in target_sub_directories:
        final_command = full_command + hab_lab_local_path + sub_dir
        print(final_command)
        os.system(final_command)

def archive_data(archive_directory):
    rsync_base = "rsync -azhP "
    target_sub_directories = sub_directories["results"] + sub_directories["videos"]
    target_sub_directories.append("run_exp.py")
    target_sub_directories.append("habitat_baselines/config/ant_v2/ppo_ant_v2_train.yaml")
    target_sub_directories.append("configs/tasks/ant-v2.yaml")
    target_sub_directories.append("habitat/tasks/ant_v2/ant_v2.py")
    for sub_dir in target_sub_directories:
        source = hab_lab_local_path + sub_dir
        destination = archive_directory
        if sub_dir.endswith("/"):
            destination = archive_directory+sub_dir.split("/")[-2] + "/"
        final_command = rsync_base + source + " " + destination
        #print(f"source = {source}")
        #print(f"destination = {destination}")
        if not os.path.exists(destination):
            create_dir_command = "mkdir -p " + destination
            print(f"create_dir_command = {create_dir_command}")
            os.system(create_dir_command)
        print(f"final_command = {final_command}")
        os.system(final_command)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--archive", type=str,
        required=False,
        help="If provided, copy all data to a specified directory. Overrides other options.",)
    parser.add_argument("--up", action="store_true")
    parser.add_argument("--code", action="store_true")
    parser.add_argument("--config", action="store_true")
    #default is 'download' 'results' from remote
    #optional params change direction and upload type
    args = parser.parse_args()

    if args.archive:
        archive_data(args.archive)
    elif args.clean:
        clean_data()
    else:
        direction = "down" if not args.up else "up"
        type = "results"
        if args.config:
            type="config"
        elif args.code:
            type="code"
        rsync_remote(direction, type)