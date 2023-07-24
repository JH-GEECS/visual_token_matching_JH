import pandas as pd
import os
import csv
import taskonomy_constants
from tqdm import tqdm
import shutil
import ray
from PIL import Image

"""
각각의 raw data를 dataframe으로 변환한 이후에 master dataframe(- rgb가 좋을 것이다.)을 기준으로 sampling을 할 필요가 있다.
"""


def img_path_list_processor(root_dir='./data_path', write_root_dir='./processed_data_path'):
    task_list = taskonomy_constants.TASKS_GROUP_NAMES_SUB
    task_num_under_bar = taskonomy_constants.TASKS_GROUP_NAMES_NUM_UNDER

    df_dict = {}

    for each_task, each_num_under_bar in tqdm(zip(task_list, task_num_under_bar), total=len(task_list)):
        df_dict[each_task] = pd.read_csv(os.path.join(root_dir, each_task + '_img_list.txt'), header=None)

        # file path parsing
        df_dict[each_task].columns = ['path']
        df_dict[each_task][['dir', 'file']] = df_dict[each_task]['path'].str.rsplit('/', 1, expand=True)
        df_dict[each_task].drop(['path'], axis=1, inplace=True)

        df_dict[each_task][["basename", "ext"]] = df_dict[each_task]['file'].str.split('.', 1, expand=True)
        df_dict[each_task].drop(['ext'], axis=1, inplace=True)

        # task wise하게 정의한다.
        df_dict[each_task][["scene_name", "leftover"]] = df_dict[each_task]['basename'].str.split('_', 1, expand=True)

        if each_num_under_bar == 0:
            df_dict[each_task][["idx", "task_name"]] = df_dict[each_task]['leftover'].str.rsplit('_', 1, expand=True)
        elif each_num_under_bar == 1:
            df_dict[each_task][["idx", "task_name_part1", "task_name_part2"]] = df_dict[each_task][
                'leftover'].str.rsplit('_', n=2, expand=True)
            df_dict[each_task]["task_name"] = df_dict[each_task][["task_name_part1", "task_name_part2"]].apply(
                lambda row: '_'.join(row), axis=1)
            df_dict[each_task].drop(['task_name_part1', 'task_name_part2'], axis=1, inplace=True)
            pass
        else:
            assert each_num_under_bar <= 1, "each_num_under_bar should be less than 1"

        df_dict[each_task].drop(['basename', 'leftover'], axis=1, inplace=True)

    for each_task, each_df in df_dict.items():
        print(f"task: {each_task}, df shape: {each_df.shape}")
        each_df.to_csv(os.path.join(write_root_dir, each_task + '_img_list.csv'), index=False)


def data_sampler(root_dir, output_dir, sample_ratio, random_seed):
    """

    sample ratio는 0.1 = 10%, 0.01 = 1%을 따라서 작성한다.


    :param root_dir:
    :param sample_ratio:
    :return:
    """

    ### constant terms ###

    task_list = taskonomy_constants.TASKS_GROUP_NAMES_SUB
    scene_list = taskonomy_constants.BUILDINGS
    IMG = 'rgb'

    ### data load code ###

    df_dict = {}
    sampled_df_dict = {}  # sampled task를 저장하기 위한 dataframe dict
    task_img_num_dict = {}

    for each_task, each_num_under_bar in tqdm(task_list, total=len(task_list)):
        df_dict[each_task] = pd.read_csv(os.path.join(root_dir, each_task + '_img_list.csv'))
        task_img_num_dict[each_task] = len(df_dict[each_task])
        sampled_df_dict[each_task] = pd.DataFrame(
            columns=['dir', 'file', 'scene_name', 'idx', 'task_name'])  # skeleton dataframe build

    ### data statistics code ###
    ### sampling idx from rgb ###

    scene_img_num_dict = {}  # 이 걸 통해서 scene 별로 weighted sampling을 하려고 한다.
    scene_wise_target_idx_dict = {}

    # rgb가 기준이 되므로 이를 통해서 weight를 산출한다.
    for each_scene in scene_list:
        num_imgs = len(df_dict[IMG][df_dict[IMG]['scene_name'] == each_scene])
        scene_img_num_dict[each_scene] = num_imgs

        filtered_df = df_dict[IMG][df_dict[IMG]['scene_name'] == each_scene]
        sampled_df = filtered_df.sample(n=int(num_imgs * sample_ratio), random_state=random_seed)['idx']
        scene_wise_target_idx_dict[each_scene] = sampled_df

    ### sampling result write code for each task ###

    for each_scene, each_sampled_idx_list in tqdm(scene_wise_target_idx_dict.items()):
        # empty dataframe을 만들고, 추출하여 concatenate하는 방향으로 간다.
        # todo 별건 아닌데 병렬화 하면 빠를듯
        for each_task in task_list:
            # scene을 맞추고, scene에서의 idx를 맞추어야 한다.
            condition1 = df_dict[each_task]['scene_name'] == each_scene
            condition2 = df_dict[each_task]['idx'].isin(scene_wise_target_idx_dict[each_scene])
            filtered_df_part = df_dict[each_task][condition1 & condition2]
            sampled_df_dict[each_task] = pd.concat([sampled_df_dict[each_task], filtered_df_part], axis=0)

    for each_task, each_df in sampled_df_dict.items():
        print(f"task: {each_task}, df shape: {each_df.shape}")
        each_df.to_csv(os.path.join(output_dir, each_task + f'_{sample_ratio}_sampled_img_list.csv'), index=False)


@ray.remote(num_cpus=1)
class Counter(object):
    def __init__(self):
        self.n = 0

    def increment(self):
        self.n += 1

    def read(self):
        return self.n


@ray.remote
def resize_save_image_err_handle(image_path, output_dir, img_size, mode, counter):
    img_base_name = os.path.basename(image_path)
    img_write_path = os.path.join(output_dir, img_base_name)
    try:
        img = Image.open(image_path)
        img = img.resize(img_size, mode)

        img.save(img_write_path)

        counter.increment.remote()
        print(f"resized {img_base_name} written to {img_write_path}")
        return None
    except Exception as e:
        counter.increment.remote()
        print(e)
        return image_path


@ray.remote
def save_image(image_path, output_dir, counter):
    try:
        new_image_base_name = os.path.basename(image_path)
        new_image_write_path = os.path.join(output_dir, os.path.basename(image_path))
        shutil.copy(image_path, new_image_write_path)

        print(f"copying {new_image_base_name} to {new_image_write_path}")
        counter.increment.remote()
        return None

    except Exception as e:
        print(f"Error occured in {image_path}")
        print(e)
        counter.increment.remote()  # while 구문을 끝내주기 위해서는 무조건 필요
        return image_path


def parallel_sampled_image_writer(meta_info_dir, image_root_dir, output_dir, sampled_ratio):
    """
    여기서는 위에서 정의된 dataframe에 따라서 command를 하도록 만든 code이다.


    :return:
    """
    # IO intensive task에 대해서는 CPU coeff를 DISK IO 감당 가능 수치까지 늘린다. 128 IOPS
    context = ray.init(num_cpus=40, include_dashboard=True)

    task_list = taskonomy_constants.TASKS_GROUP_NAMES_SUB
    df_dict = {}
    for each_task in task_list:
        mode = Image.NEAREST if each_task == "segment_semantic" else Image.BILINEAR
        error_list = []
        os.makedirs(os.path.join(output_dir, each_task), exist_ok=True)
        df_dict[each_task] = pd.read_csv(
            os.path.join(meta_info_dir, each_task + f'_{sampled_ratio}_sampled_img_list.csv'))
        # df_dict[each_task] = pd.read_csv(os.path.join(meta_info_dir, each_task + f'_img_list.csv'))

        counter = Counter.remote()
        distributed_processing = [save_image.remote(os.path.join(image_root_dir, each_task, each_basename),
                                                    os.path.join(output_dir, each_task),
                                                    counter)
                                  for each_basename in df_dict[each_task]['file']]

        with tqdm(total=len(df_dict[each_task]), desc="Progress Indicator for each task") as pbar:
            while pbar.n < pbar.total:
                n = ray.get(counter.read.remote())
                pbar.n = n
                pbar.refresh()

        result = ray.get(distributed_processing)
        for each_result in result:
            if each_result is not None:
                error_list.append(each_result)

        with open(os.path.join(output_dir, f"error_img_{each_task}_list.txt"), "w") as csvfile:
            csvwriter = csv.writer(csvfile)
            for each_img_path in error_list:
                csvwriter.writerow([each_img_path])

    ray.shutdown()


def parallel_whole_image_writer(meta_info_dir, image_root_dir, output_dir, task_list, num_cpus):
    """
    여기서는 위에서 정의된 dataframe에 따라서 command를 하도록 만든 code이다.


    :return:
    """
    # IO intensive task에 대해서는 CPU coeff를 DISK IO 감당 가능 수치까지 늘린다. 128 IOPS
    context = ray.init(num_cpus=num_cpus, include_dashboard=True)

    df_dict = {}
    for each_task in task_list:
        mode = Image.NEAREST if each_task == "segment_semantic" else Image.BILINEAR
        error_list = []
        os.makedirs(os.path.join(output_dir, each_task), exist_ok=True)
        df_dict[each_task] = pd.read_csv(
            os.path.join(meta_info_dir, each_task + f'_img_list.csv'))
        # df_dict[each_task] = pd.read_csv(os.path.join(meta_info_dir, each_task + f'_img_list.csv'))

        counter = Counter.remote()
        distributed_processing = [save_image.remote(os.path.join(image_root_dir, each_task, each_basename),
                                                    os.path.join(output_dir, each_task),
                                                    counter)
                                  for each_basename in df_dict[each_task]['file']]

        with tqdm(total=len(df_dict[each_task]), desc="Progress Indicator for each task") as pbar:
            while pbar.n < pbar.total:
                n = ray.get(counter.read.remote())
                pbar.n = n
                pbar.refresh()

        result = ray.get(distributed_processing)
        for each_result in result:
            if each_result is not None:
                error_list.append(each_result)

        with open(os.path.join(output_dir, f"error_img_{each_task}_list.txt"), "w") as csvfile:
            csvwriter = csv.writer(csvfile)
            for each_img_path in error_list:
                csvwriter.writerow([each_img_path])

    ray.shutdown()


def parallel_whole_image_resize_writer(meta_info_dir, image_root_dir, output_dir, task_list, img_size, num_cpus):
    """
    여기서는 위에서 정의된 dataframe에 따라서 command를 하도록 만든 code이다.

    :return:
    """
    # IO intensive task에 대해서는 CPU coeff를 DISK IO 감당 가능 수치까지 늘린다. 128 IOPS
    context = ray.init(num_cpus=num_cpus, include_dashboard=True)

    df_dict = {}
    for each_task in task_list:
        mode = Image.NEAREST if each_task == "segment_semantic" else Image.BILINEAR
        error_list = []
        os.makedirs(os.path.join(output_dir, each_task), exist_ok=True)
        df_dict[each_task] = pd.read_csv(
            os.path.join(meta_info_dir, each_task + f'_img_list.csv'))
        # df_dict[each_task] = pd.read_csv(os.path.join(meta_info_dir, each_task + f'_img_list.csv'))

        counter = Counter.remote()
        distributed_processing = [
            resize_save_image_err_handle.remote(os.path.join(image_root_dir, each_task, each_basename),
                                                os.path.join(output_dir, each_task),
                                                img_size,
                                                mode,
                                                counter)
            for each_basename in df_dict[each_task]['file']]

        with tqdm(total=len(df_dict[each_task]), desc="Progress Indicator for each task") as pbar:
            while pbar.n < pbar.total:
                n = ray.get(counter.read.remote())
                pbar.n = n
                pbar.refresh()

        result = ray.get(distributed_processing)
        for each_result in result:
            if each_result is not None:
                error_list.append(each_result)

        with open(os.path.join(output_dir, f"error_img_{each_task}_list.txt"), "w") as csvfile:
            csvwriter = csv.writer(csvfile)
            for each_img_path in error_list:
                csvwriter.writerow([each_img_path])

    ray.shutdown()


if __name__ == '__main__':
    # TASKS_GROUP_NAMES_SUB = ["segment_semantic", "normal", "depth_euclidean", "depth_zbuffer", "edge_occlusion", "keypoints2d", "keypoints3d", "reshading", "principal_curvature", 'rgb']

    # img_path_list_processor(root_dir=txt_root_dir, write_root_dir=csv_root_dir)
    # data_sampler(root_dir=csv_root_dir, output_dir=sample_csv_root_dir, sample_ratio=SAMPLE_RATIO, random_seed=RANDOM_SEED)

    """
    ### parallel sample image writer ###
    SAMPLE_RATIO = 0.1
    RANDOM_SEED = 42
    sample_csv_root_dir = f'/dataset/Taskonomy_tiny/meta_info/processed_data_path_{SAMPLE_RATIO}'
    image_root_dir = r'/dataset/Taskonomy_tiny/vtm_dataset_10_bak_resize_trial2'
    sampled_img_root_dir = f'/dataset_ssd/vtm_dataset_10_resized_sample_{SAMPLE_RATIO}_ssd'
    os.makedirs(sampled_img_root_dir, exist_ok=True)
    parallel_sampled_image_writer(meta_info_dir=sample_csv_root_dir,
                                  image_root_dir=image_root_dir,
                                  output_dir=sampled_img_root_dir,
                                  sampled_ratio=SAMPLE_RATIO)
    """

    """
    ### parallel whole image writer ###
    meta_csv_root_dir = r'/dataset/Taskonomy_tiny/meta_info/processed_data_path'
    image_root_dir = r'/dataset/Taskonomy_tiny/vtm_dataset_10_bak'
    # output_root_dir = r'/dataset/Taskonomy_tiny/vtm_dataset_10_bak_resize_trial2'
    output_root_dir = r'/dataset_ssd/vtm_dataset_10_resized_ssd'
    task_list = ['segment_semantic']
    num_cpus = 128

    parallel_whole_image_writer(meta_info_dir=meta_csv_root_dir,
                                image_root_dir=image_root_dir,
                                output_dir=output_root_dir,
                                task_list=task_list,
                                num_cpus=num_cpus)
    """
    ### parallel whole image resize writer ###
    meta_csv_root_dir = r'/dataset/Taskonomy_tiny/meta_info/processed_data_path'
    image_root_dir = r'/dataset/Taskonomy_tiny/vtm_dataset_10_bak'
    # output_root_dir = r'/dataset/Taskonomy_tiny/vtm_dataset_10_bak_resize_trial2'
    output_root_dir = r'/dataset_ssd/vtm_dataset_10_resized_ssd'
    task_list = ["segment_semantic", "normal", "depth_euclidean",
                 "depth_zbuffer", "edge_occlusion", "keypoints2d",
                 "keypoints3d", "reshading", "principal_curvature", 'rgb']
    num_cpus = 128
    img_size = (256, 256)

    parallel_whole_image_resize_writer(meta_info_dir=meta_csv_root_dir,
                                       image_root_dir=image_root_dir,
                                       output_dir=output_root_dir,
                                       task_list=task_list,
                                       img_size=img_size,
                                       num_cpus=num_cpus)
