# -*- coding: utf-8 -*-
# filename: receive.py

import xml.etree.ElementTree as ET


def parse_xml(web_data):
    if len(web_data) == 0:
        return None
    xmlData = ET.fromstring(web_data)
    msg_type = xmlData.find('MsgType').text
    if msg_type == 'event':
        print('msg_type is event')
        event_type = xmlData.find('Event').text
        if event_type == 'CLICK':
            return Click(xmlData)
    elif msg_type == 'text':
        print('msg_type is text')
        return TextMsg(xmlData)
    elif msg_type == 'image':
        print('msg_type is image')
        return ImageMsg(xmlData)
    else:
        pass


class Msg(object):
    def __init__(self, xmlData):
        self.ToUserName = xmlData.find('ToUserName').text
        self.FromUserName = xmlData.find('FromUserName').text
        self.CreateTime = xmlData.find('CreateTime').text
        self.MsgType = xmlData.find('MsgType').text
        self.MsgId = xmlData.find('MsgId').text


def subscribe(message):
    return "您好！欢迎关注“XLearn学会学习”！\n \
    我是一个分享高效学习，交流健康生活和专注自我提升的智能助手，可以叫我“小李”哦[愉快]。\n \
    我专注于三个方面：学习提升，心灵成长和素食健康。\n \
    欢迎和我聊天[愉快]\n \
    回复食材名字或者运动项目可以得知相应的热量\n \
    让我们一起成长，每天充满正能量！[太阳]"


def unsubscribe(message):
    return "谢谢您一路相伴！"


class TextMsg(Msg):
    def __init__(self, xmlData):
        Msg.__init__(self, xmlData)
        self.Content = xmlData.find('Content').text.encode("utf-8")


class ImageMsg(Msg):
    def __init__(self, xmlData):
        Msg.__init__(self, xmlData)
        self.PicUrl = xmlData.find('PicUrl').text
        self.MediaId = xmlData.find('MediaId').text


class EventMsg(object):
    def __init__(self, xmlData):
        self.ToUserName = xmlData.find('ToUserName').text
        self.FromUserName = xmlData.find('FromUserName').text
        self.CreateTime = xmlData.find('CreateTime').text
        self.MsgType = xmlData.find('MsgType').text
        self.Event = xmlData.find('Event').text


class Click(EventMsg):
    def __init__(self, xmlData):
        EventMsg.__init__(self, xmlData)
        self.Eventkey = xmlData.find('EventKey').text
