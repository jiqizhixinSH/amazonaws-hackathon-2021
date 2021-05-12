# class LoginScreen(GridLayout):
#     def __init__(self, **kwargs):
#         super(LoginScreen, self).__init__(**kwargs)
#         self.cols = 2
#         self.username = TextInput(multiline=False)
#         self.add_widget(self.username)
#         self.add_widget(Label(text='Search'))
#         self.add_widget(Label(text='bookTitle1'))
#         self.add_widget(Label(text='bookTitle2'))
#         self.add_widget(Label(text='bookTitle3'))
#         self.add_widget(Label(text='bookTitle4'))
#
#
# class TestApp(App):
#     def build(self):
#         return LoginScreen()
#
# TestApp().run()

# https://blog.csdn.net/cloveses/article/details/80385986
import kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.properties import ObjectProperty
from kivy.uix.label import Label
from kivy.factory import Factory
from kivy.core.text import LabelBase

# from utils import *
from utils import Agent

TRANSCRIBE_RES = ''

class MyForm(BoxLayout):  # 此处类定义虽然为空，但会将my.kv的GUI定义的相关“程序”引入，即相当于在此定义
    text_input = ObjectProperty()  # 在类中添加text_input属性，对应kv文件中用于外部引用的名称，最终指向对应id的GUI部件
    agent = Agent()
    translate_res = ''

    # 加载字体资源（使用中文）
    kivy.resources.resource_add_path("./fonts")
    font_zh = kivy.resources.resource_find("msyh.ttc")
    # 通过labelBase
    LabelBase.register("msyh_labelBase", "msyh.ttc")
    kivy.core.text.Label.register("msyh_label", "msyh.ttc")

    def button_act(self, action=None):
        if action is None:
            res = self.text_input.text # 获取text_input所指向GUI部件的text值，
        elif action == 'Translate_zh2en':
            res = self.agent.translate(text=self.text_input.text, source='auto', target='en')
        elif action == 'Translate_en2zh':
            res = self.agent.translate(text=self.text_input.text, source='auto', target='zh')
        elif action == 'Audio':
            text = self.agent.transcribe()
            # Optional: display on the UI
            res = self.agent.translate(text=text, source='auto', target='en')
        else:
            raise NotImplementedError

        print(res)  # 打印结果到控制台
        # 显示翻译结果到UI界面
        self.label_output.text = res
#        TRANSCRIBE_RES = res
#        cur_wdgt = Factory.TranslateOutput()
#        self.add_widget(cur_wdgt)
        return
    
    def clean_label(self,):
        # 清除label文本
        self.label_output.text = ""
        return

    # # 转换界面方法1
    # def chg_widget(self):
    #     self.clear_widgets()
    #     self.add_widget(Label(text='location'))  # 添加程序生成的Widget

    # 转换界面方法2
    def chg_widget(self, toScreen):
        self.clear_widgets()
        if toScreen == 'screenB':
            self.add_widget(Label(text='location'))
        elif toScreen == 'Search':
            cur_wdgt = Factory.MyFormSearch()
            self.add_widget(cur_wdgt)  # 添加kv文件中定义的Widget
        elif toScreen == 'Book1':
            cur_wdgt = Factory.MyFormBook1()
            self.add_widget(cur_wdgt)  # 添加kv文件中定义的Widget
        elif toScreen == 'Book2':
            cur_wdgt = Factory.MyFormBook2()
            self.add_widget(cur_wdgt)  # 添加kv文件中定义的Widget
        else:
            raise Exception

    def return_home(self):
        return


class MySubForm(BoxLayout):
    translate_res = 'Hello'


class MyApp(App):  # 类名MyApp 在运行时正好自动载入对应的my.kv文件
    pass


if __name__ == '__main__':
    MyApp().run()