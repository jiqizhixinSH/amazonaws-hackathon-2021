## SoccerHacker

![SoccerHacker](https://raw.githubusercontent.com/vaew/amazonaws-hackathon-2021/main/AI%E4%B8%BA%E4%BD%93%E8%82%B2%E8%A1%8C%E4%B8%9A%E5%B8%A6%E6%9D%A5%E7%9A%84%E6%96%B0%E5%85%83%E7%B4%A0%E3%80%81%E6%96%B0%E6%80%9D%E6%83%B3%E5%92%8C%E6%96%B0%E7%8E%A9%E6%B3%95%E3%80%82/%E6%9C%80%E4%BD%B3%E5%B0%84%E9%97%A8%E7%90%83%E5%91%98%E9%A2%84%E6%B5%8B%E5%B0%8F%E5%8A%A9%E6%89%8B-SoccerHacker/resources/SoccerHacker-v1.0-blue.svg)  ![build](https://img.shields.io/badge/Build-passing-green.svg)  ![Tests](https://raw.githubusercontent.com/vaew/amazonaws-hackathon-2021/main/AI%E4%B8%BA%E4%BD%93%E8%82%B2%E8%A1%8C%E4%B8%9A%E5%B8%A6%E6%9D%A5%E7%9A%84%E6%96%B0%E5%85%83%E7%B4%A0%E3%80%81%E6%96%B0%E6%80%9D%E6%83%B3%E5%92%8C%E6%96%B0%E7%8E%A9%E6%B3%95%E3%80%82/%E6%9C%80%E4%BD%B3%E5%B0%84%E9%97%A8%E7%90%83%E5%91%98%E9%A2%84%E6%B5%8B%E5%B0%8F%E5%8A%A9%E6%89%8B-SoccerHacker/resources/Tests-passing-green.svg)

<a href="https://test-ybuz.notebook.cn-northwest-1.sagemaker.com.cn/notebooks/SoccerHacker/test.ipynb"><img src="https://cn-northwest-1.console.amazonaws.cn/favicon.ico" alt="Open In AWS"></a>**Click this icon  to  open in AWS SageMaker studio!!!**

This repo is intended to build a program based on YOLO-V3 which enables player tracking and goal prediction. This project is deployed in AWS Sagemaker studio, which is a modification of the core code of YOLO-V3, and can be run on AWS with one click without installing other dependencies. Come and try it out!!! :rofl:

In recent years, as target detection algorithms have been focused on by many scholars, lightweight target detection has become possible. This project is a small turitorial target detection project based on the algorithm idea of  [YOLO v3](https://github.com/xiaochus/YOLOv3), which can achieve real-time tracking of players and infer the most likely goal scorer. This project is based on AWS Sagemaker studio and aims to use AI to give new impetus to sports events.

![](https://raw.githubusercontent.com/wizyoung/YOLOv3_TensorFlow/master/data/demo_data/results/messi.jpg)

## 0. Requirement

Python Version: 3

Packages:

- tensorflow >= 1.8.0 (theoretically any version that supports tf.data is ok)
- opencv-python
- tqdm

## 1. Quick start

* Run it in AWS Sagemaker Studio: click here!!! <a href="https://test-ybuz.notebook.cn-northwest-1.sagemaker.com.cn/notebooks/SoccerHacker/test.ipynb"><img src="https://cn-northwest-1.console.amazonaws.cn/favicon.ico" alt="Open In AWS"></a>
* Run it locally

a) Firstly, you need to download the pre-training weights file for the model from [**here**](https://pjreddie.com/media/files/yolov3.weights)

b) Then, put the weigth file in `\test\`, open the demo notebook: test.ipynb

c) Run it!!!

## 2. Demo Result

You can get the follwing result for players tracking and the best shooter predicting:

![](https://raw.githubusercontent.com/vaew/amazonaws-hackathon-2021/main/AI%E4%B8%BA%E4%BD%93%E8%82%B2%E8%A1%8C%E4%B8%9A%E5%B8%A6%E6%9D%A5%E7%9A%84%E6%96%B0%E5%85%83%E7%B4%A0%E3%80%81%E6%96%B0%E6%80%9D%E6%83%B3%E5%92%8C%E6%96%B0%E7%8E%A9%E6%B3%95%E3%80%82/%E6%9C%80%E4%BD%B3%E5%B0%84%E9%97%A8%E7%90%83%E5%91%98%E9%A2%84%E6%B5%8B%E5%B0%8F%E5%8A%A9%E6%89%8B-SoccerHacker/resources/result1.png)

![](https://raw.githubusercontent.com/vaew/amazonaws-hackathon-2021/main/AI%E4%B8%BA%E4%BD%93%E8%82%B2%E8%A1%8C%E4%B8%9A%E5%B8%A6%E6%9D%A5%E7%9A%84%E6%96%B0%E5%85%83%E7%B4%A0%E3%80%81%E6%96%B0%E6%80%9D%E6%83%B3%E5%92%8C%E6%96%B0%E7%8E%A9%E6%B3%95%E3%80%82/%E6%9C%80%E4%BD%B3%E5%B0%84%E9%97%A8%E7%90%83%E5%91%98%E9%A2%84%E6%B5%8B%E5%B0%8F%E5%8A%A9%E6%89%8B-SoccerHacker/resources/result2.png)

## 3.Inference speed

How fast is the inference speed? With images scaled to 416*416:

| Backbone              | GPU      | Time(ms) |
| --------------------- | -------- | -------- |
| Darknet-53 (paper)    | Titan X  | 29       |
| Darknet-53 (my impl.) | Titan XP | ~23      |

## 4. Model architecture

For better understanding of the model architecture, you can refer to the following picture. 

![](https://raw.githubusercontent.com/wizyoung/YOLOv3_TensorFlow/master/docs/yolo_v3_architecture.png)

## Reference

```
@article{YOLOv3,  
  title={YOLOv3: An Incremental Improvement},  
  author={J Redmon, A Farhadi },
  year={2018}
```

