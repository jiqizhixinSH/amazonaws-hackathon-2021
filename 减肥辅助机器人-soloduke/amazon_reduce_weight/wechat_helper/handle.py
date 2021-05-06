# -*- coding: utf-8 -*-
# filename: handle.py
import hashlib

import web

from config import app_token
from wechat_helper import reply, receive
from handler_function import helper

# from poem.config import Config
# from poem.generate_poem import PoetryModel
# model = PoetryModel(Config)
# print('model loaded')


class Handle(object):

    def GET(self):
        try:
            data = web.input()
            if len(data) == 0:
                return u"hi, 这里是一个新的世界"

            # 首次绑定公众号时需要对签名进行验证
            signature = data.signature
            timestamp = data.timestamp
            nonce = data.nonce
            echostr = data.echostr
            token = app_token  # 请按照公众平台官网\基本配置中信息填写

            my_list = [token, timestamp, nonce]
            my_list.sort()
            sha1 = hashlib.sha1()
            map(sha1.update, my_list)
            hashcode = sha1.hexdigest()
            print("handle/GET func: hashcode, signature: ", hashcode, signature)  # 打印后台日志
            if hashcode == signature:
                return echostr
            else:
                return ""
        except Exception as e:
            print(e)
            return e

    def POST(self):
        try:
            webData = web.data()
            print("Handle Post webdata is \n", webData)  # 打印后台日志
            recMsg = receive.parse_xml(webData)

            if isinstance(recMsg, receive.Msg):
                toUser = recMsg.FromUserName
                fromUser = recMsg.ToUserName
                if recMsg.MsgType == 'text':
                    content = recMsg.Content
                    #replyMsg = reply.TextMsg(toUser, fromUser, content, model=lambda x: str(x))
                    replyMsg = reply.TextMsg(toUser, fromUser, content, model=helper)
                    return replyMsg.send()
                if recMsg.MsgType == 'image':
                    mediaId = recMsg.MediaId
                    replyMsg = reply.ImageMsg(toUser, fromUser, mediaId)
                    return replyMsg.send()
                else:
                    return reply.Msg().send()

            if isinstance(recMsg, receive.EventMsg):
                toUser = recMsg.FromUserName
                fromUser = recMsg.ToUserName
                if recMsg.Event == 'CLICK':
                    print('It is a CLICK event')
                    content = u'功能正在开发中，敬请期待..'.encode('utf-8')
                    replyMsg = reply.TextMsg(toUser, fromUser, content)
                    return replyMsg.send()

            print("暂且不处理")
            return reply.Msg().send()

        except Exception as e:
            print(e)
            return e
