# Meeting Stickers Demo

AWS Hackathon Online 2021 ｜ 
Organizer: Heart of Machine

## Project Name

AI Meeting Stickers: [DEMO](https://share.streamlit.io/vanessa920/aws-comp-nlp/main/UI.py) | [Presentation](https://github.com/vanessa920/amazonaws-hackathon-2021/blob/main/%E5%96%84%E7%94%A8AI%EF%BC%8C%E5%88%9B%E9%80%A0%E7%BE%8E%E5%A5%BD%E7%94%9F%E6%B4%BB%E3%80%82/AI%20Meeting%20Stickers%20-%20AI%E4%BC%9A%E8%AE%AE%E4%BE%BF%E7%AD%BE%E6%9D%A1%20-%20%E7%82%92%E6%A0%97%E5%AD%90%E5%B0%8F%E9%98%9F/img/AI%20Meeting%20Stickers.pdf)

## Project Team

+ Vanessa Hu - https://www.linkedin.com/in/vanessahu/
+ Sheng Xiong Ding (Roland) - https://www.linkedin.com/in/roland-ding-403a5b1a/

## Background

In work and life, the distribution and sorting of meeting minutes is a thankless task. For those who want to know the content of the meeting, it is difficult to quickly obtain the direct content of their interest by reading the long list of meeting minutes. We use the government's public meeting records as a training data set, and we also hope to help the government improve the transparency and accessibility of government affairs, and promote public taxpayers to participate in government affairs supervision and municipal construction. Usually, the content of the government's municipal meeting is published online in PDF format, and it is difficult for people to search and consult according to the specific content. We perform structural segmentation and text preprocessing of the conference content, convert the text data into real-valued vectors that can be directly used by machine learning algorithms, and use natural language processing algorithms for Topic Modeling, adding word embedding (Word2Vec) technology Perform classification feature processing, so that the public can search and subscribe to the topics of interest in the municipal meeting across the timeline, like flipping through the notes, without having to read the complete meeting record to obtain the interesting segmented content. The future vision of the work is to promote the work within the enterprise, to efficiently classify, retrieve and subscribe to the specific content of each department meeting, so as to save the participation time of some participants, and establish an automated AI system to improve the efficiency of meetings and communication.

## Goal

The goal is parse a large dataset of public meetings (i.e. City Council, School Board, Planning Commission) and surface critical insights to everyday community members. This may involve imagine recognition, natural language processing, and sentiment analysis. Meeting minutes are often stored as PDFs so we need help running image recognition on the PDFs. 

An example use case: we want to analyze the structure of each meeting and serialize the meeting structure so we can pass it to other software applications. 
Another example use case: we want to analyze the meeting contents so we can tag meetings. A user may want to subscribe to meetings that talk about housing so we need to tag meetings that talk about housing in the agenda."

The goal is to build out the NLP capabilities in processing text documents to accurately and succinctly capture relevant information on key words.

[Code for San Jose Project List](https://docs.google.com/spreadsheets/d/15nBWVyG4nFTOFKP4u1tOgFxH9xwAF8uaZG47ABm7HQ4/edit#gid=545916388)

## Data

San Jose City Council Meeting Minutes Source: [Legistar](https://sanjose.legistar.com/Calendar.aspx)

## Features

1. Subject keyword query function
2. Cross-timeline query and retrieval
3. A quick tour of conference topics
4. User personalized settings
5. Budget related query retrieval

***

# AI会议便签条
APP DEMO 网站呈现：[AI Meeting Stickers](https://share.streamlit.io/vanessa920/aws-comp-nlp/main/UI.py)
作品 PPT: [Presentation](https://github.com/vanessa920/amazonaws-hackathon-2021/blob/main/%E5%96%84%E7%94%A8AI%EF%BC%8C%E5%88%9B%E9%80%A0%E7%BE%8E%E5%A5%BD%E7%94%9F%E6%B4%BB%E3%80%82/AI%20Meeting%20Stickers%20-%20AI%E4%BC%9A%E8%AE%AE%E4%BE%BF%E7%AD%BE%E6%9D%A1%20-%20%E7%82%92%E6%A0%97%E5%AD%90%E5%B0%8F%E9%98%9F/img/AI%20Meeting%20Stickers.pdf)

## 团队成员

+ Vanessa Hu [LinkedIn](https://www.linkedin.com/in/vanessahu/) | [Github](https://github.com/vanessa920)
+ Sheng Xiong Ding (Roland) [LinkedIn](https://www.linkedin.com/in/roland-ding-403a5b1a/) | [Github](https://github.com/shengxio)

## 作品介绍

在工作和生活中，会议记录的分发和整理是一件费力不讨好的事情，对于想要了解具体开会内容的人，阅读连篇累牍的会议记录很难快速获取自身感兴趣的摘要内容。我们利用政府公开的会议记录作为训练数据集，也寄希望于能帮助政府提高政务公开的透明度和可查询性，推动公众纳税人参与政务监督与市政建设。 通常政府的市政会议内容会以PDF的格式进行线上公开，人们很难根据具体内容进行检索和查阅。我们对会议内容进行结构细分和文本预处理，把文本数据转换成便于机器学习算法直接使用的实值向量，并用自然语言处理的算法进行主题模型(Topic Modeling)，加入词嵌入（Word2Vec）技术进行分类特征处理，从而让公众可以对市政会议里感兴趣的主题进行跨时间线的检索与订阅，像翻阅便签条一样，无须阅读完整的会议记录而获取感兴趣的细分内容。作品未来的愿景是推广到企业内部，对于各部门会议的具体内容进行片断式的高效分类、检索和订阅，从而节省部分与会者的参与时间，以建立自动化的AI系统来提高开会和沟通的效率。

## 数据来源

美国加州圣何塞市2020年市政会议记录
- 共计45 份会议记录，842页，1648个段落，11万字
- 数据来源：[Legistar](https://sanjose.legistar.com/Calendar.aspx)

## 所使用到的AWS服务

Amazon S3, AWS Textract, AWS Lambda, AWS Key Management Service, IAM, Amazon Simple Storage Service, Amazon Elastic File System, AmazonCloudWatch, Amazon Simple Notification Service, AWS Data Transfer

## 安装运行

- 运行环境

gensim==3.8.3
spacy==3.0.6
pandas==1.2.4
scikit-learn==0.23.2
altair==4.1.0
boto3==1.17.62
en_core_web_sm==3.0.0
streamlit==0.81.1
wordcloud==1.8.1.post3+g0b3b942
tika==1.24
spacy-streamlit==1.0.0
numpy==1.20.2

- 运行指令

streamlit run UI.py


## 作品截图

- 界面一：网站首页 - 应用场景介绍，以及全部会议内容的关键词词云。

![网站首页](https://github.com/vanessa920/amazonaws-hackathon-2021/blob/main/%E5%96%84%E7%94%A8AI%EF%BC%8C%E5%88%9B%E9%80%A0%E7%BE%8E%E5%A5%BD%E7%94%9F%E6%B4%BB%E3%80%82/AI%20Meeting%20Stickers%20-%20AI%E4%BC%9A%E8%AE%AE%E4%BE%BF%E7%AD%BE%E6%9D%A1%20-%20%E7%82%92%E6%A0%97%E5%AD%90%E5%B0%8F%E9%98%9F/img/home_page.png)

- 界面二：会议便签条 - 根据关键词设定和会议日期范围，精准检索相关的细分内容，并查看是否提及预算以及其他关键会议内容。
![会议便签条](https://github.com/vanessa920/amazonaws-hackathon-2021/blob/main/%E5%96%84%E7%94%A8AI%EF%BC%8C%E5%88%9B%E9%80%A0%E7%BE%8E%E5%A5%BD%E7%94%9F%E6%B4%BB%E3%80%82/AI%20Meeting%20Stickers%20-%20AI%E4%BC%9A%E8%AE%AE%E4%BE%BF%E7%AD%BE%E6%9D%A1%20-%20%E7%82%92%E6%A0%97%E5%AD%90%E5%B0%8F%E9%98%9F/img/sticker_page.png)

- 界面三：原会议记录 - 全年的会议记录原文，以及各个会议的关键词词云。
![原会议记录](https://github.com/vanessa920/amazonaws-hackathon-2021/blob/main/%E5%96%84%E7%94%A8AI%EF%BC%8C%E5%88%9B%E9%80%A0%E7%BE%8E%E5%A5%BD%E7%94%9F%E6%B4%BB%E3%80%82/AI%20Meeting%20Stickers%20-%20AI%E4%BC%9A%E8%AE%AE%E4%BE%BF%E7%AD%BE%E6%9D%A1%20-%20%E7%82%92%E6%A0%97%E5%AD%90%E5%B0%8F%E9%98%9F/img/original_page.png)

- 界面四：上传功能 - 支持上传最新的会议记录，并重新训练模型，生成更多的会议便签条。
![上传会议记录](https://github.com/vanessa920/amazonaws-hackathon-2021/blob/main/%E5%96%84%E7%94%A8AI%EF%BC%8C%E5%88%9B%E9%80%A0%E7%BE%8E%E5%A5%BD%E7%94%9F%E6%B4%BB%E3%80%82/AI%20Meeting%20Stickers%20-%20AI%E4%BC%9A%E8%AE%AE%E4%BE%BF%E7%AD%BE%E6%9D%A1%20-%20%E7%82%92%E6%A0%97%E5%AD%90%E5%B0%8F%E9%98%9F/img/upload_page.png)




