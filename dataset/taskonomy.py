import os
import random
import numpy as np
import PIL
import pandas as pd
from PIL import Image
from einops import rearrange, repeat

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .taskonomy_constants import SEMSEG_CLASSES, SEMSEG_CLASS_RANGE, TASKS_GROUP_DICT, TASKS, BUILDINGS
from .augmentation import RandomHorizontalFlip, FILTERING_AUGMENTATIONS, RandomCompose, Mixup
from .utils import crop_arrays, SobelEdgeDetector



class TaskonomyBaseDataset(Dataset):
    def __init__(self, root_dir, buildings, tasks, base_size=(256, 256), img_size=(224, 224), seed=None, precision='fp32'):
        super().__init__()

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
        
        self.data_root = root_dir 
        self.buildings = sorted(buildings)
        
        self.tasks = tasks
        self.subtasks = []
        for task in tasks:
            if task in TASKS_GROUP_DICT:
                self.subtasks += TASKS_GROUP_DICT[task]
            else:
                self.subtasks += [task]
        
        self.support_classes = SEMSEG_CLASSES
                
        self.base_size = base_size
        self.img_size = img_size
        self.precision = precision

        # add code and csv information for fast loading of img paths
        self.meta_info_path = 'dataset/meta_info'
        self.rgb_file_df = pd.read_csv(os.path.join(self.meta_info_path, 'rgb_img_list.csv'))
        self.rgb_path_list = self.rgb_file_df['file'].sort_values().to_list()

        # 여기에서 os.listdir operation을 쓰는데 disk overhead가 매우 클것이다.
        # todo 이부분 무조건 수정이 필요하다., problem exists in here
        self.img_paths = self.rgb_file_df[self.rgb_file_df['scene_name'].isin(self.buildings)]['file'].sort_values().to_list()
        # os.listdir(os.path.join(self.data_root, 'rgb'))  ## previously

        # 이 부분은 pure python code여서 느리긴 한데, disk io만큼은 아니고 한번 init되면 끝이므로 괜찮다.
        self.path_dict = {building: [i for i, img_path in enumerate(self.img_paths)
                                     if img_path.split('_')[0] == building]
                          for building in self.buildings}
        # register euclidean depth and occlusion edge statistics, sobel edge detectors, and class dictionary

        self.depth_quantiles = torch.load(os.path.join(self.meta_info_path, 'depth_quantiles.pth'))
        self.edge_params = torch.load(os.path.join(self.meta_info_path, 'edge_params.pth'))
        self.sobel_detectors = [SobelEdgeDetector(kernel_size=k, sigma=s) for k, s in self.edge_params['params']]
        self.edge_thresholds = torch.load(os.path.join(self.meta_info_path, 'edge_thresholds.pth'))
        
    ## try except는 성능에 크게 영향을 미치지 않는다.
    def load_img(self, img_path):
        img_path = os.path.join(self.data_root, 'rgb', img_path)
        try:
            # open image file
            img = Image.open(img_path)
            # img = img.resize((256, 256), PIL.Image.BILINEAR)  # image preprocessed 512 -> 256
            img = np.asarray(img)
            # 여기 bilinear interpolation 적용해야함
            
            # type conversion
            img = img.astype('float32') / 255
            
            # shape conversion
            img = np.transpose(img, (2, 0, 1))
            
            success = True

        except PIL.UnidentifiedImageError:
            print(f'PIL Error on {img_path}')
            img = -np.ones((3, 256, 256)).astype('float32')
            # 이 부분에서 image error인 경우에 있어서, 그래도 batchify를 위해서 256,256으로 만들어 주는 것이 맞을 것 같다.
            # img = -np.ones(3, 80, 80).astype('float32')
            success = False

        except OSError:
            print(f'OSError on {img_path}')
            img = -np.ones((3, 256, 256)).astype('float32')
            success = False

        return img, success

    def load_label(self, task, img_path):
        task_root = os.path.join(self.data_root, task)
        if task == 'segment_semantic':
            label_name = img_path.replace('rgb', 'segmentsemantic')
        else:
            label_name = img_path.replace('rgb', task)
        label_path = os.path.join(task_root, label_name)

        num_of_sub_task = len(TASKS_GROUP_DICT[task])

        try:
            # open label file
            label = Image.open(label_path)
            # label = label.resize((256, 256), PIL.Image.BILINEAR)  # image preprocessed 512 -> 256
            label = np.asarray(label)
            # 여기 bilinear interpolation 적용해야함

            # type conversion
            if label.dtype == 'uint8':
                label = label.astype('float32') / 255
            else:
                label = label.astype('float32')

            # shape conversion
            # 1 channel task를 의미할 것이다.
            if label.ndim == 2:
                label = label[np.newaxis, ...]
            # many channel task를 의미할 것이다.
            elif label.ndim == 3:
                label = np.transpose(label, (2, 0, 1))

            success = True

        except Exception as e:
            print(f'Error on {label_path}')
            label = -np.ones((num_of_sub_task, 256, 256)).astype('float32')
            success = False

        except PIL.UnidentifiedImageError as e:
            print(f'PIL Error on {label_path}')
            label = -np.ones((num_of_sub_task, 256, 256)).astype('float32')
            success = False

        return label, success
        
    def load_task(self, task, img_path):
        if task == 'segment_semantic':
            label, success = self.load_label(task, img_path)
            label = (255*label).astype("long")
            label[label == 0] = 1
            label = label - 1
            mask = np.ones_like(label)
            
        elif task == 'normal':
            label, success = self.load_label(task, img_path)
            label = np.clip(label, 0, 1)
            
            mask = np.ones_like(label)
            
        elif task in ['depth_euclidean', 'depth_zbuffer']:
            label, success = self.load_label(task, img_path)
            label = np.log((1 + label)) / np.log(2 ** 16)
            
            depth_label, success = self.load_label('depth_euclidean', img_path)
            mask = (depth_label < 64500)
            
        # 해당 task는 수행하지 않겠다는 것으로 보인다.
        elif task == 'edge_texture':
            label = mask = None
            success = False
            
        elif task == 'edge_occlusion':
            label, success = self.load_label(task, img_path)
            label = label / (2 ** 16)
            
            depth_label, success = self.load_label('depth_euclidean', img_path)
            mask = (depth_label < 64500)
            
        elif task == 'keypoints2d':
            label, success = self.load_label(task, img_path)
            label = label / (2 ** 16)
            label = np.clip(label, 0, 0.005) / 0.005
            
            mask = np.ones_like(label)
            
        elif task == 'keypoints3d':
            label, success = self.load_label(task, img_path)
            label = label / (2 ** 16)
            
            depth_label, success = self.load_label('depth_euclidean', img_path)
            mask = (depth_label < 64500)
            
        elif task == 'reshading':
            label, success = self.load_label(task, img_path)
            label = label[:1]
            label = np.clip(label, 0, 1)
            
            mask = np.ones_like(label)
            
        elif task == 'principal_curvature':
            label, success = self.load_label(task, img_path)
            label = label[:2]
            label = np.clip(label, 0, 1)
            
            depth_label, success = self.load_label('depth_euclidean', img_path)
            mask = (depth_label < 64500)
            
        else:
            raise ValueError(task)
            
        return label, mask, success
    
    def preprocess_segment_semantic(self, labels, channels, drop_background=True):
        # regard non-support classes as background
        for c in SEMSEG_CLASS_RANGE:
            if c not in channels:
                labels = np.where(labels == c,
                                  np.zeros_like(labels),
                                  labels)

        # re-label support classes
        for i, c in enumerate(sorted(channels)):
            labels = np.where(labels == c,
                              (i + 1)*np.ones_like(labels),
                              labels)

        # one-hot encoding
        labels = torch.from_numpy(labels).long().squeeze(1)
        labels = F.one_hot(labels, len(channels) + 1).permute(0, 3, 1, 2).float()
        if drop_background:
            labels = labels[:, 1:]
        masks = torch.ones_like(labels)
        
        return labels, masks
    
    def preprocess_depth(self, labels, masks, channels, task):
        labels = torch.from_numpy(labels).float()
        masks = torch.from_numpy(masks).float()

        labels_th = []
        for c in channels:
            assert c < len(self.depth_quantiles[task]) - 1

            # get boundary values for the depth segment
            t_min = self.depth_quantiles[task][c]
            if task == 'depth_euclidean':
                t_max = self.depth_quantiles[task][c+1]
            else:
                t_max = self.depth_quantiles[task][5]

            # thresholding and re-normalizing
            labels_ = torch.where(masks.bool(), labels, t_min*torch.ones_like(labels))
            labels_ = torch.clip(labels_, t_min, t_max)
            labels_ = (labels_ - t_min) / (t_max - t_min)
            labels_th.append(labels_)

        labels = torch.cat(labels_th, 1)
        masks = masks.expand_as(labels)
        
        return labels, masks
    
    def preprocess_edge_texture(self, imgs, channels):
        labels = []
        # detect sobel edge with different set of pre-defined parameters
        for c in channels:
            labels_ = self.sobel_detectors[c].detect(imgs)
            labels.append(labels_)
        labels = torch.cat(labels, 1)

        # thresholding and re-normalizing
        labels = torch.clip(labels, 0, self.edge_params['threshold'])
        labels = labels / self.edge_params['threshold']

        masks = torch.ones_like(labels)
        
        return labels, masks
    
    def preprocess_edge_occlusion(self, labels, masks, channels):
        labels = torch.from_numpy(labels).float()
        masks = torch.from_numpy(masks).float()

        labels_th = []
        labels = torch.where(masks.bool(), labels, torch.zeros_like(labels))
        for c in channels:
            assert c < len(self.edge_thresholds)
            t_max = self.edge_thresholds[c]

            # thresholding and re-normalizing
            labels_ = torch.clip(labels, 0, t_max)
            labels_ = labels_ / t_max
            labels_th.append(labels_)

        labels = torch.cat(labels_th, 1)
        masks = masks.expand_as(labels)
        
        return labels, masks
    
    def preprocess_default(self, labels, masks, channels):
        labels = torch.from_numpy(labels).float()

        if masks is not None:
            masks = torch.from_numpy(masks).float().expand_as(labels)
        else:
            masks = torch.ones_like(labels)
            
        labels = labels[:, channels]
        masks = masks[:, channels]
            
        return labels, masks

    # images processing 단에서 compose로 짜지 않아서 상당한 병목이 예상된다.
    def preprocess_batch(self, task, imgs, labels, masks, channels=None, drop_background=True):
        imgs = torch.from_numpy(imgs).float()

        # process all channels if not given
        if channels is None:
            if task == 'segment_semantic':
                channels = SEMSEG_CLASSES
            elif task in TASKS_GROUP_DICT:
                channels = range(len(TASKS_GROUP_DICT[task]))
            else:
                raise ValueError(task)
            
        # task-specific preprocessing
        if task == 'segment_semantic':
            labels, masks = self.preprocess_segment_semantic(labels, channels, drop_background)

        # depth z_buffer에서 무슨 문제가 있는가?
        elif task in ['depth_euclidean', 'depth_zbuffer']:
            labels, masks = self.preprocess_depth(labels, masks, channels, task)
        
        elif task == 'edge_texture':
            labels, masks = self.preprocess_edge_texture(imgs, channels)

        elif task == 'edge_occlusion':
            labels, masks = self.preprocess_edge_occlusion(labels, masks, channels)
                
        else:
            labels, masks = self.preprocess_default(labels, masks, channels)

        # ensure label values to be in [0, 1]
        labels = labels.clip(0, 1)
        
        # precision conversion
        if self.precision == 'fp16':
            imgs = imgs.half()
            labels = labels.half()
            masks = masks.half()
        elif self.precision == 'bf16':
            imgs = imgs.bfloat16()
            labels = labels.bfloat16()
            masks = masks.bfloat16()

        return imgs, labels, masks

        
class TaskonomyHybridDataset(TaskonomyBaseDataset):
    def __init__(self, root_dir, buildings, tasks, shot, tasks_per_batch, domains_per_batch,
                 image_augmentation, unary_augmentation, binary_augmentation, mixed_augmentation, dset_size=-1, **kwargs):
        super().__init__(root_dir, buildings, tasks, **kwargs)
        
        assert shot > 0
        self.shot = shot
        #  각 task에 대하여 얼마 만큼의 sample을 보여 줄 것인가?
        self.tasks_per_batch = tasks_per_batch
        # 각각의 batch에 얼마만큼의 task를 넣을 것인가?
        self.domains_per_batch = domains_per_batch
        # domain per batch가 2로 지정되어 있는데 이게 뭘까?
        # 각각의 batch에 대해서 train으로 지정된 building 중에서 2개를 고른다는 의미였다.
        self.dset_size = dset_size
        # 20000 * 8 = 160000 16만장인데 이게 의미하는 바가 무엇인가?

        if image_augmentation:
            self.image_augmentation = RandomHorizontalFlip()
        else:
            self.image_augmentation = None
        
        if unary_augmentation:
            self.unary_augmentation = RandomCompose(
                [augmentation(**kwargs) for augmentation, kwargs in FILTERING_AUGMENTATIONS.values()],
                p=0.8,
            )
        else:
            self.unary_augmentation = None

        if binary_augmentation is not None:
            self.binary_augmentation = Mixup()
        else:
            self.binary_augmentation = None

        self.mixed_augmentation = mixed_augmentation
        
    def __len__(self):
        if self.dset_size > 0:
            return self.dset_size
            # val_iter: 20000 * 8 = 160000을 dataset size로 선정했다는 점이
            # 잘 이해가 가지 않는다.
        else:
            return len(self.img_paths) // self.shot
    
    # batch가 model의 한 iter에 해당하는 것인가?
    # task를 어떻게 정의하는 가?
    # 뭔가 여기서 문제가 자주 생기네
    def sample_batch(self, task, channel, path_idxs=None):
        # sample data paths
        if path_idxs is None:
            # sample buildings for support and query
            buildings = np.random.choice(self.buildings, 2*self.domains_per_batch, replace=False)
            # 그런데 여기에서 왜 2를 곱하는 건가?, 하여간 4개의 building만을 선발함을 알 수 있다.

            # sample image path indices in each building
            path_idxs = np.array([], dtype=np.int64)
            for building in buildings:
                # 특정 building에서, shot / domain_per_batch를 통해서
                # 4 shot이라면 4개의 예제를 의미하는 것일 텐데, building의 수로 나눈다.

                # 한 building당 4개를 sampling함
                path_idxs = np.concatenate((path_idxs,
                                            np.random.choice(self.path_dict[building], 
                                                             self.shot // self.domains_per_batch,
                                                             replace=False)))
        # 첫 번째에 대해서 다 된다.
        # load images and labels
        imgs = []
        labels = []
        masks = []
        for path_idx in path_idxs:
            # index image path
            # 여기 조금 문제 발생
            img_path = self.img_paths[path_idx]

            # load image, label, and mask
            # success 이 부분을 잘 이용하는 수밖에 없을 것 같다.
            img, success_img = self.load_img(img_path)
            # task의 구성을 어떻게 하는 가를 살펴봐야 한다.
            label, mask, success_label = self.load_task(task, img_path)
            if not success_img or not success_label:
                mask = np.zeros_like(label)

            imgs.append(img)
            labels.append(label)
            # mask의 역활은 해당 label을 쓸 수 있다를 의미한다.
            masks.append(mask)
            
        # form a batch
        imgs = np.stack(imgs)
        labels = np.stack(labels) if labels[0] is not None else None
        masks = np.stack(masks) if masks[0] is not None else None
        
        # preprocess and make numpy arrays to torch tensors
        # 요 부분이 전처리 code
        # 여기에서 channel을 어떻게 쓰는지 가 문제이다.
        imgs, labels, masks = self.preprocess_batch(task, imgs, labels, masks, [channel])
        
        return imgs, labels, masks, path_idxs
    
    def sample_tasks(self):
        # sample subtasks
        replace = len(self.subtasks) < self.tasks_per_batch
        subtasks = np.random.choice(self.subtasks, self.tasks_per_batch, replace=replace)
        
        # parse task and channel from the subtasks
        tasks = []
        channels = []
        for subtask in subtasks:
            # subtask format: "{task}_{channel}"
            task = '_'.join(subtask.split('_')[:-1])
            channel = int(subtask.split('_')[-1])
            
            tasks.append(task)
            channels.append(channel)
            
        return tasks, channels
    
    # 각각의 batch를 가져오는 역활을 하는 것을 보인다.
    # todo 거의다 왔다. task 구성이랑 optimized params만 잘 살펴보자.
    def __getitem__(self, idx):
        # sample tasks
        tasks, channels = self.sample_tasks()
        if self.binary_augmentation is not None:
            tasks_aux, channels_aux = self.sample_tasks()
            
        X = []
        Y = []
        M = []
        t_idx = []
            
        path_idxs = None  # generated at the first task and then shared for the remaining tasks
        # todo, analysis how data is loaded?
        # 각각의 batch(4shot 학습, 4shot validation)에 대하여 5개의 tasks, model이 5개의 차원에 대한 output
        for i in range(self.tasks_per_batch):
            # sample a batch of images, labels, and masks for each task
            X_, Y_, M_, path_idxs = self.sample_batch(tasks[i], channels[i], path_idxs)
                
            # apply image augmentation
            if self.image_augmentation is not None:
                (X_, Y_, M_), image_aug = self.image_augmentation(X_, Y_, M_, get_augs=True)
            else:
                image_aug = lambda x: x
            
            # apply unary task augmentation
            if self.unary_augmentation is not None:
                Y_, M_ = self.unary_augmentation(Y_, M_)
                
            # apply binary task augmentation
            if self.binary_augmentation is not None:
                _, Y_aux_, M_aux_, _, = self.sample_batch(tasks_aux[i], channels_aux[i], path_idxs)
                if self.mixed_augmentation and self.image_augmentation is not None:
                    (Y_aux_, M_aux_) = self.image_augmentation(Y_, M_)
                else:
                    Y_aux_ = image_aug(Y_aux_)
                    M_aux_ = image_aug(M_aux_)
                Y_, M_ = self.binary_augmentation(Y_, Y_aux_, M_, M_aux_)
            
            X.append(X_)
            Y.append(Y_)
            M.append(M_)
            
            t_idx.append(TASKS.index(f'{tasks[i]}_{channels[i]}'))
        
        # form a global batch
        # 이 부분에서 문제이다. 가장 큰 점은 256 x 256 preprocessing이 끝나면 well done이라는 점이나,
        # 저자의 code 사상으로는 512 x 512 image는 자동으로 resize transform이 되었어야 하는 것이다.

        X = torch.stack(X)
        Y = torch.stack(Y)  # 이 지점에서 error가 발생하고 있다.
        M = torch.stack(M)

        # random-crop arrays
        X, Y, M = crop_arrays(X, Y, M,
                              base_size=self.base_size,
                              img_size=self.img_size,
                              random=True)
            
        # task and task-group index
        t_idx = torch.tensor(t_idx)
        
        return X, Y, M, t_idx
    

class TaskonomyContinuousDataset(TaskonomyBaseDataset):
    def __init__(self, root_dir, buildings, task, channel_idx=-1, dset_size=-1, image_augmentation=False, **kwargs):
        super().__init__(root_dir, buildings, [task], **kwargs)
        
        self.task = task
        self.channel_idx = channel_idx
        self.dset_size = dset_size
        self.n_channels = len(TASKS_GROUP_DICT[task])
        
        if image_augmentation:
            self.image_augmentation = RandomHorizontalFlip()
        else:
            self.image_augmentation = None
    
    def __len__(self):
        if self.dset_size > 0:
            return self.dset_size
        else:
            return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx % len(self.img_paths)]

        # load image, label, and mask
        # success 이 부분을 잘 이용하는 수밖에 없을 것 같다.
        img, success_img = self.load_img(img_path)
        label, mask, success_label = self.load_task(self.task, img_path)
        if not success_img or not success_label:
            mask = np.zeros_like(label)

        # preprocess labels
        imgs, labels, masks = self.preprocess_batch(self.task,
                                                    img[None],
                                                    None if label is None else label[None],
                                                    None if mask is None else mask[None],
                                                    channels=([self.channel_idx] if self.channel_idx >= 0 else None),
                                                    drop_background=False)
        
        
        X, Y, M = imgs[0], labels[0], masks[0]
        if self.image_augmentation is not None:
            X, Y, M = self.image_augmentation(X, Y, M)
        
        # crop arrays
        X, Y, M = crop_arrays(X, Y, M,
                              base_size=self.base_size,
                              img_size=self.img_size,
                              random=True)
        
        return X, Y, M

    
class TaskonomySegmentationDataset(TaskonomyBaseDataset):
    def __init__(self, root_dir, buildings, semseg_class, dset_size=-1, **kwargs):
        super().__init__(root_dir, buildings, ['segment_semantic'], **kwargs)

        self.semseg_class = semseg_class        
        self.img_paths = sorted(self.rgb_path_list)  # use global path dictionary
        self.class_dict = torch.load(os.path.join(self.meta_info_path, 'class_dict.pth'))

        self.n_channels = 1

        perm_path = os.path.join(self.meta_info_path, 'idxs_perm_all.pth')
        if os.path.exists(perm_path):
            idxs_perm = torch.load(perm_path)
        else:
            idxs_perm = {}
            for c in SEMSEG_CLASSES:
                n_imgs = 0
                for building in BUILDINGS:
                    n_imgs += len(self.class_dict[building][c])
                idxs_perm[c] = torch.randperm(n_imgs)
            torch.save(idxs_perm, perm_path)
        
        # collect all images of class c
        class_idxs = []
        for building in BUILDINGS:
            class_idxs += self.class_dict[building][self.semseg_class]
            
        # permute the image indices
        class_idxs = (torch.tensor(class_idxs)[idxs_perm[self.semseg_class]]).numpy()
        
        # filter images in given buildings
        self.class_idxs = [class_idx for class_idx in class_idxs if self.img_paths[class_idx].split('_')[0] in buildings]
        
        self.dset_size = dset_size
    
    def __len__(self):
        if self.dset_size > 0:
            return self.dset_size
        else:
            return len(self.class_idxs)
            
    def __getitem__(self, idx):
        path_idx = self.class_idxs[idx % len(self.class_idxs)]
        img_path = self.img_paths[path_idx]

        # load image, label, and mask
        img, success_img = self.load_img(img_path)
        label, mask, success_label = self.load_task('segment_semantic', img_path)
        if not success_img or not success_label:
            mask = np.zeros_like(mask)

        # preprocess labels
        imgs, labels, masks = self.preprocess_batch('segment_semantic',
                                                    img[None],
                                                    None if label is None else label[None],
                                                    None if mask is None else mask[None],
                                                    channels=[self.semseg_class])
        
        X, Y, M = imgs[0], labels[0], masks[0]
        
        # crop arrays
        X, Y, M = crop_arrays(X, Y, M,
                              base_size=self.base_size,
                              img_size=self.img_size,
                              random=True)
        
        return X, Y, M


class TaskonomyFinetuneDataset(TaskonomyBaseDataset):
    def __init__(self, root_dir, buildings, task, support_idx, channel_idx, shot,
                 dset_size=1, image_augmentation=False, fix_seed=False, shuffle_idxs=True, **kwargs):
        super().__init__(root_dir, buildings, [task], **kwargs)
        
        self.task = task
        self.support_idx = support_idx
        self.shot = shot
        self.dset_size = dset_size
        self.channel_idx = channel_idx
        self.fix_seed = fix_seed
        self.shuffle_idxs = shuffle_idxs
        self.offset = support_idx*shot
        
        if image_augmentation:
            self.image_augmentation = RandomHorizontalFlip()
        else:
            self.image_augmentation = None
            
        if task == 'segment_semantic':
            self.img_paths = sorted(self.rgb_path_list)
            self.class_dict = torch.load(os.path.join(self.meta_info_path, 'class_dict.pth'))

            class_idxs = []
            for building in buildings:
                class_idxs += self.class_dict[building][self.channel_idx]
            self.class_idxs = class_idxs

            perm_path = os.path.join(self.meta_info_path, f'class_perm_finetune_{self.channel_idx}.pth')
            if not os.path.exists(perm_path):
                class_perm = torch.randperm(len(self.class_idxs))
                torch.save(class_perm, perm_path)
            else:
                class_perm = torch.load(perm_path)
            self.class_perm = class_perm

        else:
            perm_path = os.path.join(self.meta_info_path, 'idxs_perm_finetune.pth')
            if not os.path.exists(perm_path):
                idxs_perm = torch.randperm(len(self.img_paths))
                torch.save(idxs_perm, perm_path)
            else:
                idxs_perm = torch.load(perm_path)
            self.idxs_perm = idxs_perm
    
    def __len__(self):
        return self.dset_size
    
    def __getitem__(self, idx):
        idxs = [(idx % self.shot) + self.offset + i for i in range(self.shot)]
        random.shuffle(idxs)
        imgs = []
        labels = []
        masks = []
        for idx_ in idxs:
            if self.task == 'segment_semantic':
                if self.shuffle_idxs:
                    idx_ = self.class_perm[idx_ % len(self.class_idxs)]
                path_idx = self.class_idxs[idx_]
                img_path = self.img_paths[path_idx]
            else:
                if self.shuffle_idxs:
                    idx_ = self.idxs_perm[idx_ % len(self.img_paths)]
                img_path = self.img_paths[idx_]

            # load image, label, and mask
            img, success_img = self.load_img(img_path)
            label, mask, success_label = self.load_task(self.task, img_path)
            if not success_img or not success_label:
                mask = np.zeros_like(label)

            imgs.append(img)
            labels.append(label)
            masks.append(mask)

        imgs = np.stack(imgs)
        labels = np.stack(labels) if labels[0] is not None else None
        masks = np.stack(masks) if masks[0] is not None else None

        # preprocess labels
        imgs, labels, masks = self.preprocess_batch(self.task, imgs, labels, masks,
                                                    channels=([self.channel_idx] if self.channel_idx >= 0 else None),
                                                    drop_background=True)
        
        X = repeat(imgs, 'N C H W -> T N C H W', T=labels.size(1))
        Y = rearrange(labels, 'N T H W -> T N 1 H W')
        M = rearrange(masks, 'N T H W -> T N 1 H W')
        t_idx = torch.arange(len(Y))

        if self.image_augmentation is not None and not self.fix_seed:
            X, Y, M = self.image_augmentation(X, Y, M)
        
        # crop arrays
        X, Y, M = crop_arrays(X, Y, M,
                              base_size=self.base_size,
                              img_size=self.img_size,
                              random=(not self.fix_seed))
        
        return X, Y, M, t_idx
