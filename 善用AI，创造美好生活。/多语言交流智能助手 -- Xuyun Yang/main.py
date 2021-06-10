# -*- coding: utf-8 -*-
# https://blog.csdn.net/cloveses/article/details/80385986
import kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.properties import ObjectProperty
from kivy.uix.label import Label
from kivy.factory import Factory
from kivy.core.text import LabelBase
import time

# from utils import *
from utils import Agent

TRANSCRIBE_RES = ''

class MyForm(BoxLayout):  # 此处类定义虽然为空，但会将my.kv的GUI定义的相关“程序”引入，即相当于在此定义
    text_input = ObjectProperty()  # 在类中添加text_input属性，对应kv文件中用于外部引用的名称，最终指向对应id的GUI部件
    label_output = ObjectProperty()  # 在类中添加label_output属性，对应kv文件中用于外部引用的名称，最终指向对应id的GUI部件
    agent = Agent()
    translate_res = None

    # 加载字体资源（使用中文）
    kivy.resources.resource_add_path("./fonts")
    font_zh = kivy.resources.resource_find("msyh.ttc")
    # 通过labelBase
    LabelBase.register("msyh_labelBase", "msyh.ttc")
    kivy.core.text.Label.register("msyh_label", "msyh.ttc")
    
    def button_act(self, action=None):
        print('Start ...')
        if action is None:
            self.translate_res = self.text_input.text  # 获取text_input所指向GUI部件的text值，
        elif action == 'Translate':
            t = self.target_lang.text
            if t == 'Target':
                self.translate_res = self.text_input.text
            else:
                self.translate_res = self.agent.translate(text=self.text_input.text, source='auto', target=t)
        elif action == 'Audio':
            text = self.agent.transcribe()
            t = self.target_lang.text
            if t == 'Target':
                t = 'en'  # default
            self.translate_res = self.agent.translate(text=text, source='auto', target=t)
        elif action == 'Speech':
            # 语音输出翻译结果
            try:
                assert self.translate_res is not None
                self.agent.speech(self.translate_res)
                print('Finish speeching ...')
            except:
                print('No speeching content.')
                pass
        else:
            raise NotImplementedError

        print(self.translate_res)  # 打印结果到控制台
        if self.translate_res is not None:
            if action == 'Speech':
                pass
            else:
                # 显示翻译结果到UI界面
                self.label_output.text = self.translate_res
                print('Finish translating ...')
            pass

        return
    
    def clean_label(self,):
        # 清除label文本
        self.label_output.text = ""  # (Waiting ...)
        self.translate_res = None
        return


class AICommApp(App):  # 类名AICommApp 在运行时正好自动载入对应的aicomm.kv文件
    pass


if __name__ == '__main__':
    AICommApp().run()
