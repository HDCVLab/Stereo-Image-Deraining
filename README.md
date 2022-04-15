# Beyond monocular deraining: Parallel stereo deraining network via semantic prior （ECCV2020, IJCV2022）


## Prerequisites
- Python >= 3.6  
- [Pytorch](https://pytorch.org/) >= 1.0  
- Torchvision >= 0.2.2  
- Pillow >= 5.1.0  
- Numpy >= 1.14.3
- Scipy >= 1.1.0

## Introduction
- ```train.py``` is the codes for training the StereoDerainNet.
- ```test.py``` is the codes for testing the StereoDerainNet.
- ```train_data.py``` and ```val_data.py``` are used to load the training and validation/testing datasets.
- ```model.py``` defines the model of StereoDerainNet.
- ```utils.py``` contains all corresponding utilities.


## Quick Start

### 1. Testing
Ready the K12, K15 or cityscape dataset
- Please ensure the data structure is as below.

```
├── K12(K15)
   └── test
       ├── image_2_3_norain
       ├── image_3_2_norain
       ├── image_2_3_rain50
       └── image_3_2_rain50

```

```
├── rain_cityscape_val_gt
   └── citynames
   └── ...

├── rain_cityscape_val
   └── citynames
   └── ...
```
The StereoDerainNet pre-trained model can be found (https://pan.baidu.com/s/1mE4ouS_76pwJg6KHyOd33g
) (password : kmqt).

The Semantic_seg pre-trained model can be found (https://pan.baidu.com/s/1VrGk0A-RT54-Twp66YZ_mA
) (password : 2333).

You can evaluate the model by running the command below after download tow pre-trained models.
- For Stereo model :
```bash
$ python test_K12.py -semantic -single_stereo 

$ python test_K15.py -semantic -single_stereo 
```
- For Monocular model :
```bash
$ python test_K12.py -semantic -single_single

$ python test_K15.py -semantic -single_single

$ python test_cityscape.py -semantic -single_single 
```

### 2. Training

Ready the K12, K15 or cityscape dataset
- Please ensure the data structure is as below.

```
├── K12(K15)
   └── train
       ├── image_2_3_norain
       ├── image_3_2_norain
       ├── image_2_3_rain50
       └── image_3_2_rain50

```

```
├── rain_cityscape_gt
   └── citynames
   └── ...

├── rain_cityscape
   └── citynames
   └──...
```
After set the dataset directory in ```train.py```,\
You can train the model by running the command below 
- For Stereo model :
```bash
$ python train_K12.py -semantic -single_stereo 

$ python train_K15.py -semantic -single_stereo 
```
- For Monocular model :
```bash
$ python train_K12.py -semantic -single_single

$ python train_K15.py -semantic -single_single

$ python train_cityscape.py -semantic -single_single
```


## Training detail
#### Stereo :
   - K12 batch_size=2 epoch=55 learn_rate=0.0002
   - K15 batch_size=2 epoch=50 learn_rate=0.0002

#### Monocular :
   - K12 batch_size=4 epoch=55 learn_rate=0.0002
   - K15 batch_size=4 epoch=50 learn_rate=0.0002
   - CityScape batch_size=4 epoch=200(100 for coarse + 100 for both) learn_rate=0.0002



##  Our datasets
 - Our training and testing datasets can be found (https://pan.baidu.com/s/1T2UplwARbLS5apIQiAnEXg
) (password : zzkd).


##  Our result
 - Our derain result in three datasets can be found (https://pan.baidu.com/s/1BV2-TPL5GiTlDbxjSR0qyg
) (password : yb4y).



## Citation
If you think this work is useful for your research, please cite the following paper.

```
@inproceedings{zhang2020beyond,
  title={Beyond Monocular Deraining: Stereo Image Deraining via Semantic Understanding},
  author={Zhang, Kaihao and Luo, Wenhan and Ren, Wenqi and Wang, Jingwen and Zhao, Fang and Ma, Lin and Li, Hongdong},
  booktitle={European Conference on Computer Vision},
  pages={71--89},
  year={2020},
  organization={Springer}
}

@article{zhang2022beyond,
  title={Beyond monocular deraining: Parallel stereo deraining network via semantic prior},
  author={Zhang, Kaihao and Luo, Wenhan and Yu, Yanjiang and Ren, Wenqi and Zhao, Fang and Li, Changsheng and Ma, Lin and Liu, Wei and Li, Hongdong},
  journal={International Journal of Computer Vision},
  year={2022}
}
```



