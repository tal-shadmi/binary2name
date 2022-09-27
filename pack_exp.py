import shutil
from os import makedirs
from os.path import isdir, join as path_join, isfile
from time import strftime, gmtime

# TODO will be move
from shutil import move #, copytree



from common import stats_file_name, log_file_name, ref_file_name, predicted_file_name, default_model_save_dir


def make_sure_dir_exists(path, delete_if_exists=False):
    if isdir(path):
        if delete_if_exists:
            shutil.rmtree(path)
            makedirs(path)
    else:
        makedirs(path)


def get_date_string():
    return strftime("%Y-%m-%d_%H-%M-%S", gmtime())


packs_dir_name = "Packs"
make_sure_dir_exists(packs_dir_name)
this_pack_dir = path_join(packs_dir_name, get_date_string())
make_sure_dir_exists(this_pack_dir)

move(stats_file_name, this_pack_dir)
#move(log_file_name, this_pack_dir)
move(ref_file_name, this_pack_dir)
move(predicted_file_name, this_pack_dir)
move(default_model_save_dir, path_join(this_pack_dir,default_model_save_dir))

screen_log_file_name = "screenlog.0"

if isfile(screen_log_file_name):
    move(screen_log_file_name, this_pack_dir)

move("start_exp_common.py", this_pack_dir)
