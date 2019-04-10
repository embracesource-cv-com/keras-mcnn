# MCNN_in_Keras
keras实现的人群数量估计网络["Single Image Crowd Counting via Multi Column Convolutional Neural Network"](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Zhang_Single-Image_Crowd_Counting_CVPR_2016_paper.pdf)，持续改进中...

## 安装
1. Clone
    ```shell
    git clone https://github.com/ybcc2015/MCNN_in_Keras.git
    ```

2. 安装依赖库
    ```shell
    cd MCNN_in_Keras
    pip install -r requirements.txt
    ```

## 数据配置
1. 下载ShanghaiTech数据集:    
    [Dropbox](https://www.dropbox.com/s/fipgjqxl7uj8hd5/ShanghaiTech.zip?dl=0)
    or [百度云盘](http://pan.baidu.com/s/1nuAYslz)

2. 创建数据存放目录
    ```shell
    mkdir ./data/original/shanghaitech/
    ```

3. 将```part_A_final```和```part_B_final```存放到./data/original/shanghaitech/目录下

4. 生成测试集的ground truth文件
    ```shell
    cd data_preparation
    python create_gt_test_set_shtech.py [A or B]  # Part_A or Part_B
    ```
    生成好的ground-truth文件会保存在./data/original/shanghaitech/part_【A or B】_final/test_data/ground_truth_csv目录下
    
5. 生成训练集和验证集
    ```shell
    cd data_preparation
    python create_training_set_shtech.py [A or B]  # Part_A or Part_B
    ```
    生成好的数据保存在./data/formatted_trainval_【A or B】目录下

>2~5步均在工程根目录下操作

## 训练
```shell
python train.py [A or B]  # Part_A or Part_B
```
训练好的模型保存在./trained_models目录下

## 测试
```shell
python test.py [A or B]  # Part_A or Part_B
```
测试结果保存在./output_【A or B】目录下

## 结果

    |        |  MAE  |   MSE  |
    ---------------------------
    | Part_A |  todo |  todo  |
    ---------------------------
    | Part_B |  33.7 |  58.9  |

**Part_B**   
原图：  
![原图](./examples/IMG_148.jpg)   
Ground truth & Estimate：  
![GT](./examples/heatmap_gt_IMG_148.png "Ground Truth")&nbsp;![Estimate](./examples/heatmap_IMG_148.png "Estimate")

## todo
训练Part_A部分，由于Part_A部分的图片大小不一样，需要修改训练代码。  