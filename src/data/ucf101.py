# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
""" UCF101 dataset. TODO: uploade the dataset about 6.5G to huawei cloud obs gallery. """

import os
import random
from typing import Optional, Callable
from itertools import chain
from src.data.meta import ParseDataset
from src.data import transforms
from src.data.video_dataset import VideoDataset
from src.utils.class_factory import ClassFactory, ModuleType

__all__ = ["UCF101", "ParseUCF101"]


@ClassFactory.register(ModuleType.DATASET)
class UCF101(VideoDataset):
    """
     Args:
        path (string): Root directory of the Mnist dataset or inference image.
        split (str): The dataset split supports "train", "test" or "infer". Default: "train".
        transform (callable, optional): A function transform that takes in a video. Default:None.
        target_transform (callable, optional): A function transform that takes in a label. Default: None.
        seq(int): The number of frames of captured video. Default: 16.
        seq_mode(str): The way of capture video frames,"part") or "discrete" fetch. Default: "part".
        align(boolean): The video contains multiple actions. Default: False.
        batch_size (int): Batch size of dataset. Default:16.
        repeat_num (int): The repeat num of dataset. Default:1.
        shuffle (bool, optional): Whether or not to perform shuffle on the dataset. Default:None.
        num_parallel_workers (int): Number of subprocess used to fetch the dataset in parallel.Default: 1.
        num_shards (int, optional): Number of shards that the dataset will be divided into. Default: None.
        shard_id (int, optional): The shard ID within num_shards. Default: None.
        download (bool) : Whether to download the dataset. Default: False.

    Examples:
        >>> from mindvision.msvideo.dataset.ucf101 import UCF101
        >>> dataset = UCF101("./data/")
        >>> dataset = dataset.run()

    About UCF101 dataset:

        The UCF101 dataset consists of 13320 video in 101 classes.

        Here is the original UCF101 dataset structure.
        You can unzip the dataset files into the following directory structure and read them by MindSpore Vision's API.

        .
        |-ucf101                                     // contains 101 file folder
          |-- ApplyEyeMakeup           r             // contains 145 videos
          |   |-- v_ApplyEyeMakeup_g01_c01.avi     // video file
          |   |-- v_ApplyEyeMakeup_g01_c02.avi     // video file
          |    ...
          |-- ApplyLipstick                         // contains 114 image files
          |   |-- v_ApplyLipstick_g01_c01.avi      // video file
          |   |-- v_ApplyLipstick_g01_c02.avi      // video file
          |    ...
          |-- ucfTrainTestlist                         // contains 114 image files
          |   |-- classInd.txt      // Category file.
          |   |-- testlist01.txt      // split file
          |   |-- trainlist01.txt      // split file
          |    ...
          ...
    """

    def __init__(self,
                 path: str,
                 split: str = "train",
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 seq: int = 16,
                 seq_mode: str = "part",
                 align: bool = False,
                 batch_size: int = 16,
                 repeat_num: int = 1,
                 shuffle: Optional[bool] = None,
                 num_parallel_workers: int = 1,
                 num_shards: Optional[bool] = None,
                 shard_id: Optional[bool] = None,
                 download: bool = False
                 ):
        load_data = ParseUCF101(os.path.join(path, split)).parse_dataset
        super(UCF101, self).__init__(path=path,
                                     split=split,
                                     load_data=load_data,
                                     transform=transform,
                                     target_transform=target_transform,
                                     seq=seq,
                                     seq_mode=seq_mode,
                                     align=align,
                                     batch_size=batch_size,
                                     repeat_num=repeat_num,
                                     shuffle=shuffle,
                                     num_parallel_workers=num_parallel_workers,
                                     num_shards=num_shards,
                                     shard_id=shard_id,
                                     download=download)

    @property
    def index2label(self):
        """Get the mapping of indexes and labels."""
        parse_ucf101 = ParseUCF101(self.path)
        mapping, _ = parse_ucf101.load_cls(self.path)
        return mapping

    def download_dataset(self):
        """Download the UCF101 data if it doesn't exist already."""
        raise ValueError("UCF101 dataset download is not supported.")

    def default_transform(self):
        """Set the default transform for UCF101 dataset."""
        size = (224, 224)
        order = (3, 0, 1, 2)
        trans = [
            transforms.VideoResize(size),
            transforms.VideoReOrder(order),
        ]

        return trans


class ParseUCF101(ParseDataset):
    """
    Parse UCF101 dataset.
    """
    urlpath = "https://www.crcv.ucf.edu/data/UCF101.php#Results_on_UCF101"

    def parse_dataset(self, *args):
        """Traverse the UCF101 dataset file to get the path and label."""
        parse_ucf101 = ParseUCF101(self.path)
        split = os.path.split(parse_ucf101.path)[-1]
        base_path = os.path.dirname(parse_ucf101.path)
        video_label, video_path = [], []
        id2cls, cls2id = self.load_cls(base_path)
        if split in ("train", "test"):
            split_file = os.path.join(base_path, "ucfTrainTestlist", f"{split}list01.txt")
            with open(split_file, "r")as f:
                rows = f.readlines()
                for row in rows:
                    if split == "train":
                        info = row.split()[0]
                    if split == "test":
                        info = row.split()[0]
                    video_path.append(os.path.join(base_path, info))
                    cls = info.split('/')[0]
                    video_label.append(cls2id[cls])
            return video_path, video_label
        for cls in id2cls:
            cls_base_path = os.path.join(base_path, cls)
            video_list = os.listdir(cls_base_path)
            for file_path in video_list:
                video_path.append(os.path.join(cls_base_path, file_path))
                video_label.append(cls2id[cls])
        return video_path, video_label

    def load_cls(self, base_file):
        """Parse category file."""
        cls_file = os.path.join(base_file, "ucfTrainTestlist", "classInd.txt")
        id2cls = []
        cls2id = {}
        with open(cls_file, "r")as f:
            rows = f.readlines()
            for row in rows:
                index = int(row.split(' ')[0])
                cls = row.split(' ')[-1].strip()
                id2cls.append(cls)
                cls2id.setdefault(cls, index - 1)
        return id2cls, cls2id

    def create_task(self, task_num, n, k, q):
        """Create task list in few shot learning."""
        video_path, video_label = self.parse_dataset()
        sta_prefix = [0]
        temp = 0
        for index, label in enumerate(video_label):
            if label != temp:
                sta_prefix.append(index)
                temp = label
        cls_num = len(sta_prefix)
        task_list = []
        label_per_task = []
        num = 0
        while num < task_num:
            num += 1
            cls_list = random.sample(range(cls_num), n)
            support = []
            query = []
            label = []
            for index, cls in enumerate(cls_list):
                left = sta_prefix[cls]
                if cls + 1 == cls_num:
                    right = len(video_label)
                else:
                    right = sta_prefix[cls + 1]
                videos = random.sample(video_path[left:right], (k + q))
                support.append(videos[:k])
                query.append(videos[k:])
                label.append([index] * q)
            support = list(chain.from_iterable(support))
            query = list(chain.from_iterable(query))
            task = [support, query]
            task = list(chain.from_iterable(task))
            task_list.append(task)
            label = list(chain.from_iterable(label))
            label_per_task.append(label)
        return task_list, label_per_task

    def modify_struct(self):
        """If there is no category subdirectory in the folder, modify the file structure."""
        video_list = os.listdir(self.path)
        for video in video_list:
            cls_name = video.split("_")[1]
            input_file = os.path.join(self.path, video)
            output_file = os.path.join(self.path, cls_name)
            if not os.path.exists(output_file):
                os.mkdir(output_file)
            command = f"mv {input_file} {output_file}"
            os.system(command)
