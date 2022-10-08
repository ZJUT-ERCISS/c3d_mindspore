# Contents

## [C3D Description](#contents)

C3D model is widely used for 3D vision task. The construct of C3D network is similar to the common 2D ConvNets, the main difference is that C3D use 3D operations like Conv3D while 2D ConvNets are anentirely 2D architecture. To know more information about C3D network, you can read the original paper Learning Spatiotemporal Features with 3D Convolutional Networks.

## [Model Architecture](#contents)

C3D net has 8 convolution, 5 max-pooling, and 2 fully connected layers, followed by a softmax output layer. All 3D convolution kernels are 3 × 3 × 3 with stride 1 in both spatial and temporal dimensions. The 3D pooling layers are denoted from pool1 to pool5. All pooling kernels are 2 × 2 × 2, except for pool1 is 1 × 2 × 2. Each fully connected layer has 4096 output units.

## [Dataset](#contents)

Dataset used: [UCF101](https://www.crcv.ucf.edu/data/UCF101.php)

- Description: UCF101 is an action recognition data set of realistic action videos, collected from YouTube, having 101 action categories. This data set is an extension of UCF50 data set which has 50 action categories.

- Dataset size：13320 videos
    - Note：Use the official Train/Test Splits([UCF101TrainTestSplits](https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip)).
- Data Content Structure

```text
.
└─ucf101                                    // contains 101 file folder
  |-- ApplyEyeMakeup                        // contains 145 videos
  |   |-- v_ApplyEyeMakeup_g01_c01.avi      // video file
  |   |-- v_ApplyEyeMakeup_g01_c02.avi      // video file
  |    ...
  |-- ApplyLipstick                         // contains 114 image files
  |   |-- v_ApplyLipstick_g01_c01.avi       // video file
  |   |-- v_ApplyLipstick_g01_c02.avi       // video file
  |    ...
  |-- ucfTrainTestlist                      // contains category files
  |   |-- classInd.txt                      // Category file.
  |   |-- testlist01.txt                    // split file
  |   |-- trainlist01.txt                   // split file
  ...
```

## [Environment Requirements](#contents)

- Hardware
    - Prepare hardware environment with GPU(Nvidia).
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

## [Quick Start](#contents)

- Requirements Installation
Use the following commands to install dependencies:
```shell
pip install -r requirements.txt
```

After installing MindSpore via the official website, you can start training and evaluation as follows:

- Download pretrained model from [c3d.ckpt](https://zjuteducn-my.sharepoint.com/:u:/g/personal/201906010313_zjut_edu_cn/EbVF6SuKthpGj046abA37jkBkfkhzLm36F8NJmH2Do3jhg?e=xh32kW)


Refer to [src/config/c3d_ucf101.yaml](src/config/c3d_ucf101.yaml). We support some parameter configurations for quick start.

Please run the following command in root directory of `c3d_mindspore`. 

### Run on GPU

#### run with example files (standalone)
```bash
# run training example
python src/example/c3d_ucf101_train.py \
        --data_url [DATASET_PATH] \
        --epoch_size [NUM_OF_EPOCH] \
        --batch_size [BATCH_SIZE]
# run distributed training example
mpirun -n [NUM_OF_DEVICES] python src/example/c3d_ucf101_train.py \         
                            --data_url [DATASET_PATH] \
                            --epoch_size [NUM_OF_EPOCH] \
                            --batch_size [BATCH_SIZE]
# run evaluation example
python src/example/c3d_ucf101_eval.py \
        --data_url [DATASET_PATH] \
        --pretrained_path [CKPT_PATH] \
        --batch_size [BATCH_SIZE]
```
For example, 

- Run training example
```bash
python src/example/c3d_ucf101_train.py \
        --data_url /usr/dataset/ucf101 \
        --pretrained True \
        --pretrained_path ./c3d_pretrained.ckpt \
        --epoch_size 50 \
        --batch_size 8
```
- Run evaluation example
```bash
python src/example/c3d_ucf101_train.py \
        --data_url /usr/dataset/ucf101 \
        --pretrained_path ./c3d_ucf101.ckpt \
        --batch_size 16
```
Details of parameters can be referred to the [c3d_ucf101_train.py](src/example/c3d_ucf101_train.py) and [c3d_ucf101_eval.py](src/example/c3d_ucf101_eval.py).

#### run with scripts and config file

```bash
cd scripts
# run training example
bash run_standalone_train_gpu.sh [CONFIG_PATH]
# run distributed training example
bash run_distribute_train_gpu.sh [CONFIG_PATH] [NUM_DEVICES]
# run evaluation example
bash run_standalone_eval_gpu.sh [CONFIG_PATH]
```
Details of parameters can be referred to the yaml file.

For example 
Run training example
```bash
bash run_standalone_train_gpu.sh src/config/c3d_ucf101.yaml
```
Run distribute training example
```bash
bash run_distribute_train_gpu.sh src/config/c3d_ucf101.yaml 2
```
It is recommended to run with scripts or general [train.py](train.py), [eval.py](eval.py), because it can use yaml file to save and adjust parameters conveniently.




## [Performance](#contents)

#### Evaluation Performance

- C3D for UCF101

| Parameters          | GPU                                                       |
| -------------       |--------------------------------------  |
| Model Version       | C3D                                                       |
| Resource            | Nvidia 3090Ti                                             |
| uploaded Date       | 09/06/2022 (month/day/year)                               |
| MindSpore Version   | 1.6.1                                                     |
| Dataset             | UCF101                                                    |
| Training Parameters | epoch = 50,  batch_size = 8                               |
| Optimizer           | SGD                                                       |
| Loss Function       | Max_SoftmaxCrossEntropyWithLogits                         |
| Speed               | 1pc:237.128ms/step                                        |
| Top_1               | 1pc:75.3%                                                 |
| Total time          | 1pc:4hours                                                |
| Parameters (M)      | 78


## [ModelZoo Homepage](#contents)

Please check the official [ModelZoo](https://gitee.com/mindspore/models) for more models.


## [Citation](#contents)


If you find this project useful in your research, please consider citing:

```BibTeX
@article{C3D,
  title={Learning spatiotemporal features with 3d convolutional networks},
  author={Tran, Du and Bourdev, Lubomir and Fergus, Rob and Torresani, Lorenzo and Paluri, Manohar},
  booktitle={Proceedings of the IEEE international conference on computer vision},
  pages={4489--4497},
  year={2015}
}
```

```BibTeX
@misc{MindSpore Vision 2022, 
  title={{MindSpore Vision}:MindSpore Vision Toolbox and Benchmark}, 
  author={MindSpore Vision Contributors}, 
  howpublished = {\url{https://gitee.com/mindspore/vision}}, 
  year={2022}
}
```

