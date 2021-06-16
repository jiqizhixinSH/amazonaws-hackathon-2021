#### 作品介绍
- 本作品的设计初衷来源于人们去其他国家旅游或者出差时对于异国语言交流的需求，作品主要实现多种语言的输入及对应的翻译输出。
    - 本作品考虑到实际使用时的便捷形式，除了支持文本输入以外，还支持语音输入；支持文本输出和语音输出。
    - 通过本产品的使用，在生活中为使用不同语言的人构建了交流的桥梁，减少沟通障碍，在一定程度上提升了差旅的体验感。另外，本产品也可作为语言学习工具，让用户能够比较方便地学习和锻炼简单语句的翻译。
- 作品开发过程中使用到的亚马逊云科技服务包括：
    - Amazon S3用于数据存取
    - Amazon Transcribe用于输入语音到本文的转换
    - Amazon Translate用于不同语言翻译
    - Amazon Polly用于本文到输出语音的转换

#### Demo展示

![使用界面](https://github.com/newbieyxy/amazonaws-hackathon-2021/blob/main/%E5%96%84%E7%94%A8AI%EF%BC%8C%E5%88%9B%E9%80%A0%E7%BE%8E%E5%A5%BD%E7%94%9F%E6%B4%BB%E3%80%82/%E5%A4%9A%E8%AF%AD%E8%A8%80%E4%BA%A4%E6%B5%81%E6%99%BA%E8%83%BD%E5%8A%A9%E6%89%8B%20--%20Xuyun%20Yang/source/home.png)

![选择目标语言](https://github.com/newbieyxy/amazonaws-hackathon-2021/blob/main/%E5%96%84%E7%94%A8AI%EF%BC%8C%E5%88%9B%E9%80%A0%E7%BE%8E%E5%A5%BD%E7%94%9F%E6%B4%BB%E3%80%82/%E5%A4%9A%E8%AF%AD%E8%A8%80%E4%BA%A4%E6%B5%81%E6%99%BA%E8%83%BD%E5%8A%A9%E6%89%8B%20--%20Xuyun%20Yang/source/target_spinner.png)

![文本输入测试](https://github.com/newbieyxy/amazonaws-hackathon-2021/blob/main/%E5%96%84%E7%94%A8AI%EF%BC%8C%E5%88%9B%E9%80%A0%E7%BE%8E%E5%A5%BD%E7%94%9F%E6%B4%BB%E3%80%82/%E5%A4%9A%E8%AF%AD%E8%A8%80%E4%BA%A4%E6%B5%81%E6%99%BA%E8%83%BD%E5%8A%A9%E6%89%8B%20--%20Xuyun%20Yang/source/text_input.png)

#### 安装

```
pip install requirements.txt
```

#### Demo运行
```
python main.py
```

#### 联系方式
846232979@qq.com
