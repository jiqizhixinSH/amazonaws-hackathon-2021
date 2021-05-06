# !/usr/bin/python
# -*- coding:UTF-8 -*-
# -----------------------------------------------------------------------#
# File Name: calorie_flow
# Author: Junyi Li
# Mail: 4ljy@163.com
# Created Time: 2021-05-03
# Description:
# -----------------------------------------------------------------------#

import json
from collections import defaultdict

from knlp import ner

FOOD_CALORIE = None
WORK_OUT_DATA = None
WORK_OUT_DATA_PATH = "data/workout.json"
FOOD_CALORIE_PATH = "data/food_calorie.json"
with open(WORK_OUT_DATA_PATH) as f:
    WORK_OUT_DATA = json.load(f)
with open(FOOD_CALORIE_PATH) as f:
    FOOD_CALORIE = json.load(f)


def parse_sentence(sentence):
    """
    parse sentence
    使用NER技术解析获取到所有的Name entity

    :param sentence: str
    :return: dict, {}
    """
    ner_result = ner(sentence)
    tmp_dict = defaultdict(list)
    for word, tag in ner_result:
        if tag != "n" and word not in WORK_OUT_DATA and word not in FOOD_CALORIE:
            continue
        tmp_dict["food"].append(word)
        tmp_dict["work_out"].append(word)

    return tmp_dict


def construct_result(result_dict):
    """
    构建返回语句

    :param result_dict:
    :return:
    """
    tmp_str = ""

    for food_name, food_calorie in result_dict["food_calorie"].items():
        tmp_str += f"{food_name}的卡路里是：{food_calorie} \n"

    for work_out_name, work_out_energy in result_dict["work_out_data"].items():
        tmp_str += f"{work_out_name}的卡路里是：{work_out_energy} \n"
    return tmp_str


def main(input_sentence):
    """
    main function to return result

    :param input_sentence: str
    :return:
    """
    parse_result_dict = parse_sentence(input_sentence)
    food_words = parse_result_dict["food"]
    work_out_words = parse_result_dict["work_out"]

    output_result = {"food_calorie": {}, "work_out_data": {}}
    for food_word in food_words:
        if food_word in FOOD_CALORIE:
            output_result["food_calorie"][food_word] = FOOD_CALORIE[food_word]["能量"]
    for work_out_word in work_out_words:
        if work_out_word in WORK_OUT_DATA:
            output_result["work_out_data"][work_out_word] = WORK_OUT_DATA[work_out_word]

    return_result = construct_result(output_result)
    return return_result


if __name__ == '__main__':
    res = main("鸡蛋的卡路里是多，慢跑消耗多少，快跑消耗多少")
    print(res)
