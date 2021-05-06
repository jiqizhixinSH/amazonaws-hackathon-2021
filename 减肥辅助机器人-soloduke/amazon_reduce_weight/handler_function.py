# !/usr/bin/python
# -*- coding:UTF-8 -*-
# -----------------------------------------------------------------------#
# File Name: handler_function
# Author: Junyi Li
# Mail: 4ljy@163.com
# Created Time: 2021-05-03
# Description:
# -----------------------------------------------------------------------#
from amazon_api.tranlate import translate_helper
from calorie_helper.calculator import cal_resting_metal, loss_calorie_per_day
from calorie_helper.calorie_flow import main
from nlp_helper import re_helper


def qa_function(input_sentence):
    parse_input = re_helper(input_sentence)
    print(parse_input)
    if not parse_input:
        return ""
    if len(parse_input) < 5:
        resting_metal = cal_resting_metal(int(parse_input[1]), int(parse_input[0]), int(parse_input[2]), parse_input[3])
        return f"每天静息代谢是{resting_metal}大卡"
    else:
        calorie_per_day, weight_per_day = loss_calorie_per_day(int(parse_input[1]), int(parse_input[0]), int(parse_input[2]),
                                               parse_input[3],
                                               int(parse_input[4]),
                                               int(parse_input[5]))
        return f"每天消耗的卡路里是{calorie_per_day}大卡, {weight_per_day}kg"


def helper(input_sentence):
    """
    对输入的所有string sentence进行处理

    :param input_sentence: string
    :return: string
    """
    #print(type(input_sentence))
    #print(len(input_sentence))
    input_sentence = input_sentence.strip()
    if "问询" in input_sentence:
        zh_res = main(input_sentence)
    elif "咨询" in input_sentence:
        zh_res = qa_function(input_sentence)
    else:
        zh_res = "你的请求暂不支持解析"
    if zh_res:
        en_res = translate_helper(zh_res)
    else:
        en_res = ""
    res = f"中文结果: {zh_res}\n" \
        f"English result is: {en_res}"
    print(res)
    return res


if __name__ == '__main__':
    #res = helper("问询：鸡蛋的热量是多少")
    #print(res)
    #res = helper("咨询：水电费水电费")
    #print(res)
    q = "咨询：我的体重是100kg，身高是178，年龄是27，性别是男，目标体重是70kg，想要在30天内达到目标体重"
    res = helper(q)
    print(res)
    q = "咨询：我的体重是100kg，身高是178，年龄是27，性别是男，静息代谢率是多少"
    res = helper(q)
    print(res)

