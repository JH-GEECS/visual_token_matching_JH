import pandas as pd
import numpy as np
import ray
import os
import shutil
from tqdm import tqdm
import cv2
import glob

import csv

import os

os.environ["RAY_IGNORE_UNHANDLED_ERRORS"] = "0"

import argparse


@ray.remote(num_cpus=1)
class Counter(object):
    def __init__(self):
        self.n = 0

    def increment(self):
        self.n += 1

    def read(self):
        return self.n


def two_csv_comparison(path_csv1, path_csv2):
    df_1 = pd.read_csv(path_csv1)  # smaller one
    df_2 = pd.read_csv(path_csv2)  # bigger one
    df_2['exist'] = 0

    for index, row in df_1.iterrows():
        if row['id'] in df_2['id'].values:
            row_idxs = np.where(df_2['id'] == row['id'])[0]
            df_2.loc[row_idxs, 'exist'] = 1
    df_2.to_csv(path_csv2, index=False)


@ray.remote
def resize_save_image(image_path, output_dir, img_size, counter):
    img_base_name = os.path.basename(image_path)
    img_write_path = os.path.join(output_dir, img_base_name)

    img = cv2.imread(image_path)
    img = cv2.resize(img, img_size)

    cv2.imwrite(img_write_path, img)

    counter.increment.remote()
    print(f"resized {img_base_name} written to {img_write_path}")


@ray.remote
def rename_save_image(image_path, output_dir, scene_name, counter):
    new_image_base_name = scene_name + '_' + os.path.basename(image_path)
    new_image_write_path = os.path.join(output_dir, new_image_base_name)
    shutil.copy(image_path, new_image_write_path)

    counter.increment.remote()
    print(f"copying {new_image_base_name} to {new_image_write_path}")


@ray.remote
def resize_save_image_err_handle(image_path, output_dir, img_size, counter):
    img_base_name = os.path.basename(image_path)
    img_write_path = os.path.join(output_dir, img_base_name)
    try:
        img = cv2.imread(image_path)
        img = cv2.resize(img, img_size)

        cv2.imwrite(img_write_path, img)

        counter.increment.remote()
        print(f"resized {img_base_name} written to {img_write_path}")
        return None
    except Exception as e:
        print(e)
        counter.increment.remote()
        return image_path


def parallel_rename_save_image(image_dir, output_root_dir, scene_split):
    task_split = ["segment_semantic", "normal", "depth_euclidean", "depth_zbuffer", "edge_occlusion",
                  "keypoints2d", "keypoints3d", "reshading", "principal_curvature", "rgb"]
    context = ray.init(num_cpus=40, include_dashboard=True)

    # 각각의 task에 대해서도 상당히 방대한 데이터가 있으므로 이런 이중 for문이 cost차지 비중은 낮다.
    for scene_name in scene_split:  # 각각의 scene에 대하여
        scene_dirs = os.path.join(image_dir, scene_name)
        for task_name in task_split:  # 각 scene의 각각의 task에 대하여
            each_scene_dir = os.path.join(scene_dirs, task_name)
            image_list = os.listdir(each_scene_dir)  # 각 task의 image들에 대하여

            # 이 지점에서 각각의 new directory에 대하여 실제 directory를 os.mkdir로 만들어줘야 함
            output_dir = os.path.join(output_root_dir, task_name)  # VTM 자료구조는 scene은 버리기 때문이다.
            os.makedirs(output_dir, exist_ok=True)

            counter = Counter.remote()

            distributed_processing = []
            for each_img_name in image_list:
                distributed_processing.append(
                    resize_save_image_err_handle.remote(
                        os.path.join(each_scene_dir, each_img_name), output_dir, scene_name, counter))

            with tqdm(total=len(image_list), desc="Progress Indicator for each scene, task") as pbar:
                while pbar.n < pbar.total:
                    n = ray.get(counter.read.remote())
                    pbar.n = n
                    pbar.refresh()

            ray.get(distributed_processing)  # 여기서는 result를 받을 필요가 없어서 이렇게 처리함
    ray.shutdown()


def parallel_resize_save_image(image_root_dir, output_root_dir, img_resize_obj, args):
    os.makedirs(output_root_dir, exist_ok=True)

    task_split = ["segment_semantic", "normal", "depth_euclidean", "depth_zbuffer", "edge_occlusion",
                  "keypoints2d", "keypoints3d", "reshading", "principal_curvature", "rgb"]

    task_split_done = [
        "segment_semantic", "normal", "depth_euclidean", "edge_occlusion", "keypoints2d"
    ]

    task_split = [args.target_task]
    # "depth_zbuffer" 오류로 삭제하고 다시해주기

    error_img_list = []
    context = ray.init(num_cpus=40, include_dashboard=True)

    # 하단 code 재 작성 필요

    os.makedirs(output_root_dir, exist_ok=True)
    print("start processing")
    for task_name in task_split:  # 각 scene의 각각의 task에 대하여
        each_task_dir = os.path.join(image_root_dir, task_name)
        img_pattern = os.path.join(each_task_dir, "*.png")
        image_list = glob.glob(img_pattern)  # 각 task의 image들에 대하여
        print(f'number of images in {task_name} : {len(image_list)}')

        # 이 지점에서 각각의 new directory에 대하여 실제 directory를 os.mkdir로 만들어줘야 함
        output_dir = os.path.join(output_root_dir, task_name)  # VTM 자료구조는 scene은 버리기 때문이다.
        os.makedirs(output_dir, exist_ok=True)

        counter = Counter.remote()

        distributed_processing = []
        print(f"start processing {task_name}")
        for each_img_path in image_list:
            distributed_processing.append(
                resize_save_image_err_handle.remote(each_img_path, output_dir, img_resize_obj, counter))

        with tqdm(total=len(image_list), desc="Progress Indicator for each task") as pbar:
            while pbar.n < pbar.total:
                n = ray.get(counter.read.remote())
                pbar.n = n
                pbar.refresh()

        result = ray.get(distributed_processing)  # 다시 확인하니 error가 꽤 발생해서, 이 부분에서 manual하게 받을 필요가 있다.
        error_img_list.append(result)

    with open(os.path.join(output_root_dir, f"error_img_list_{task_split[0]}.txt"), "w") as csvfile:
        csvwriter = csv.writer(csvfile)
        for each_img_path in error_img_list:
            csvwriter.writerow([each_img_path])

    ray.shutdown()

def args_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--target_task', type=str, help="target task name")
    # target_task = "segment_semantic"

    return parser.parse_args()


if __name__ == '__main__':
    """
    ### code for parallel_rename_save_image ### 
    
    image_dir = r'/data/dataset/universial_vision_dense'
    output_dir = r'/dataset/Taskonomy_tiny/vtm_dataset_10_bak'
    scene_split = ['woodbine']

    parallel_rename_save_image(image_dir, output_dir, scene_split)
    """

    args = args_parse()

    # image_dir = r'Z:\dataset\universial_vision_dense\woodbine'
    image_dir = r'/dataset/Taskonomy_tiny/vtm_dataset_10_bak'
    # output_dir = r'Z:\dataset\universial_vision_dense\woodbine_resize_test'
    output_dir = r'/dataset/Taskonomy_tiny/vtm_dataset_10_bak_resize_trial2'
    img_resize_obj = (256, 256)
    parallel_resize_save_image(image_dir, output_dir, img_resize_obj, args)
