# keras-mcnn
keras复现人群数量估计网络["Single Image Crowd Counting via Multi Column Convolutional Neural Network"](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Zhang_Single-Image_Crowd_Counting_CVPR_2016_paper.pdf)  
>参考pytorch版:https://github.com/svishwa/crowdcount-mcnn

## 安装
1. Clone
    ```shell
    git clone https://github.com/embracesource-cv-com/keras-mcnn.git
    ```

2. 安装依赖库
    ```shell
    cd keras-mcnn
    pip install -r requirements.txt
    ```

## 数据配置
1. 下载ShanghaiTech数据集:    
    [Dropbox](https://www.dropbox.com/s/fipgjqxl7uj8hd5/ShanghaiTech.zip?dl=0)
    or [百度云盘](http://pan.baidu.com/s/1nuAYslz)

2. 创建数据存放目录$ORIGIN_DATA_PATH
    ```shell
    mkdir /opt/dataset/crowd_counting/shanghaitech/original
    ```

3. 将```part_A_final```和```part_B_final```存放到$ORIGIN_DATA_PATH目录下

4. 生成测试集的ground truth文件
    ```shell
    python create_gt_test_set_shtech.py [A or B]  # Part_A or Part_B
    ```
    生成好的ground-truth文件将会保存在$TEST_GT_PATH/test_data/ground_truth_csv目录下

5. 生成训练集和验证集
    ```shell
    python create_training_set_shtech.py [A or B]
    ```
    生成好的数据保存将会在$TRAIN_PATH、$TRAIN_GT_PATH、$VAL_PATH、$VAL_GT_PATH目录下

6. 生成热力图  
    如果你想生成测试集的ground truth热力图：

    ```shell
    python create_heatmaps.py [A or B]
    ```
    

>2~6步均在工程根目录下操作

## 测试

a)下载训练模型

[mcnn-A.160.h5](https://drive.google.com/open?id=1szCKlFLmkz7TL1axcX8jDTazq8YOW_QP) 、[mcnn-B.035.h5](https://drive.google.com/open?id=1cWGXLYR2lVllbU8JodV88gpc42BWSjyG)



b) 如下命令分别测试A和B

```shell
python test.py --dataset A --weight_path /tmp/mcnn-A.160.h5 --output_dir /tmp/mcnn_A
python test.py --dataset B --weight_path /tmp/mcnn-B.035.h5 --output_dir /tmp/mcnn_B
```


## 训练
如果你想自己训练模型，很简单：
```shell
python train.py [A or B]
```


## 结果

    |        |  MAE   |  MSE   |
    ----------------------------
    | Part_A |  127.88 |  194.19 |
    ----------------------------
    | Part_B |  30.71  |  46.81  |

### 改进点

​	由于GT密度图每个像素点的值都很小(A数据集平均为0.02,B数据集平均为0.002)，这样小的值不利于网络优化，因此对GT做了**标准化**(减去均值,然后除方差);预测时将预测的值先乘方差，再加上均值，就是最终的预测值。这个改进对最终的结果提升明显，**使用标准化后**，A数据集的**MAE**为**127.88**，**没有使用标准化**时只有**154.7**。

## 样例

**Part_A**   
原图：  
![原图](./examples/IMG_2.jpg)   
Ground Truth (1111) & Estimate (1256)：  
![GT](./examples/heatmap_gt_IMG_2.png "Ground Truth")&nbsp;![Estimate](./examples/heatmap_IMG_2.png "Estimate")

**Part_B**   
原图：  
![原图](./examples/IMG_148.jpg)   
Ground Truth (252) & Estimate(242)：  
![GT](./examples/heatmap_gt_IMG_148.png "Ground Truth")&nbsp;![Estimate](./examples/heatmap_IMG_148.png "Estimate")
