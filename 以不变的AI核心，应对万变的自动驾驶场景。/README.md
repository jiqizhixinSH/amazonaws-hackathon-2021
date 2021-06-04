# 赛题二：以不变的AI核心，应对万变的自动驾驶场景。
随着环境感知、多传感器融合、智能决策、控制与执行系统、高精度地图与定位等核心技术的快速发展与成熟，自动驾驶汽车已经从实验室走向公开道路实地测试及商业化示范的阶段。在出行服务、干线物流、封闭园区物流、公交巴士、智慧地铁、末端配送、 市政环卫、港口码头、矿山开采、零售、自动泊车等诸多应用场景中，有着人工智能加持的自动驾驶技术都已经有了广泛应用。在自动驾驶应用越来越普及的现在，还有哪些场景可以应用自动驾驶呢？请以「以不变的AI核心，应对万变的自动驾驶场景」为主题，利用人工智能技术完成一款产品（软件或硬件）的开发。

## 1. 作品介绍：基于HRC LSTM的轨迹预测

随着自动驾驶的快速发展，如何理解自动驾驶周围目标的行为成为自动驾驶系统中的重要一环。轨迹预测任务旨在根据目标(如行人、车辆等交通参与者)当前或者历史轨迹与环境信息，对该目标未来的行驶轨迹进行预测。轨迹预测结果是自动驾驶系统进行提前决策的重要信息之一。
本项目基于LSTM进行改进，结合Dagger结构提出了新的减少累计误差的训练方法HRC LSTM，从而实现更加准确的轨迹预测。通过对NGSIM I-80的数十万条数据的学习，验证集上的误差为0.064，换道行为全部成功预测，本人已发表于交通领域顶刊Transportation Research Part C。

## 2. 作品截图：

- 轨迹预测问题：

  只给定第一段时间序列的真实输入，循环预测剩余时间步本车的位置

<img src="https://github.com/tjzxh/amazonaws-hackathon-2021/blob/main/%E4%BB%A5%E4%B8%8D%E5%8F%98%E7%9A%84AI%E6%A0%B8%E5%BF%83%EF%BC%8C%E5%BA%94%E5%AF%B9%E4%B8%87%E5%8F%98%E7%9A%84%E8%87%AA%E5%8A%A8%E9%A9%BE%E9%A9%B6%E5%9C%BA%E6%99%AF%E3%80%82/%E5%9F%BA%E4%BA%8EHRC%20LSTM%E7%9A%84%E8%BD%A8%E8%BF%B9%E9%A2%84%E6%B5%8B%20-%20TJZXH/pic/1.png" alt="image-20210518181448122" style="zoom:50%;" />

- HRC LSTM模型框架：

  对预测结果进行纵向约束，并作为输入再次训练LSTM网络

<img src="https://github.com/tjzxh/amazonaws-hackathon-2021/blob/main/%E4%BB%A5%E4%B8%8D%E5%8F%98%E7%9A%84AI%E6%A0%B8%E5%BF%83%EF%BC%8C%E5%BA%94%E5%AF%B9%E4%B8%87%E5%8F%98%E7%9A%84%E8%87%AA%E5%8A%A8%E9%A9%BE%E9%A9%B6%E5%9C%BA%E6%99%AF%E3%80%82/%E5%9F%BA%E4%BA%8EHRC%20LSTM%E7%9A%84%E8%BD%A8%E8%BF%B9%E9%A2%84%E6%B5%8B%20-%20TJZXH/pic/2.png" alt="image-20210518181402739" style="zoom: 80%;" />

- 模型结构：

  超参数设置如下图

<img src="https://github.com/tjzxh/amazonaws-hackathon-2021/blob/main/%E4%BB%A5%E4%B8%8D%E5%8F%98%E7%9A%84AI%E6%A0%B8%E5%BF%83%EF%BC%8C%E5%BA%94%E5%AF%B9%E4%B8%87%E5%8F%98%E7%9A%84%E8%87%AA%E5%8A%A8%E9%A9%BE%E9%A9%B6%E5%9C%BA%E6%99%AF%E3%80%82/%E5%9F%BA%E4%BA%8EHRC%20LSTM%E7%9A%84%E8%BD%A8%E8%BF%B9%E9%A2%84%E6%B5%8B%20-%20TJZXH/pic/3.png" alt="image-20210518181243727" style="zoom: 40%;" />

- 模型结果：

  拟合度高

  <img src="https://github.com/tjzxh/amazonaws-hackathon-2021/blob/main/%E4%BB%A5%E4%B8%8D%E5%8F%98%E7%9A%84AI%E6%A0%B8%E5%BF%83%EF%BC%8C%E5%BA%94%E5%AF%B9%E4%B8%87%E5%8F%98%E7%9A%84%E8%87%AA%E5%8A%A8%E9%A9%BE%E9%A9%B6%E5%9C%BA%E6%99%AF%E3%80%82/%E5%9F%BA%E4%BA%8EHRC%20LSTM%E7%9A%84%E8%BD%A8%E8%BF%B9%E9%A2%84%E6%B5%8B%20-%20TJZXH/pic/4.png" alt="image-20210518181847714" style="zoom: 80%;" />

## 3. 使用指南

见相应HTML图表

直接在浏览器中打开，点击play即可

## 4. 团队介绍

名称：TJZXH

联系方式：tjzxh@tongji.edu.cn

## 5. 使用到的亚马逊云科技服务内容

[Amazon SageMaker](https://console.amazonaws.cn/sagemaker/home?region=cn-north-1#/landing)
