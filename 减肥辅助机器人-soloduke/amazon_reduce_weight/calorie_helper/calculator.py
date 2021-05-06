# !/usr/bin/python
# -*- coding:UTF-8 -*-
# -----------------------------------------------------------------------#
# File Name: calculator
# Author: Junyi Li
# Mail: 4ljy@163.com
# Created Time: 2021-05-03
# Description:
# -----------------------------------------------------------------------#

"""热量计算公式
单位：千卡
可以使用基础代谢率作为瘦身热量
一千克的脂肪=7700千卡的热量
常见食物热量
常见运动耗能
示例饮食搭配表

通过用户输入身高体重，计算出静息代谢
通过用户输入目标体重和现在的体重，计算出每天应该消耗多少热量，摄入多少热量


"""


def cal_resting_metal(height, weight, age, gender, rate=1.2):
    """
    男性: BMR = 66 + (13.7× 体重(kg))+(5×身高(cm)) - (6.8×年龄(岁))
    女性: BMR = 65.5 + (9.6×体重(kg))+(1.8×身高(cm))-(4.7×年龄(岁))

    :param height:
    :param weight:
    :param age:
    :param gender:
    :param rate:
    :return:
    """
    if gender == "男" or "M":
        res = 66 + (13.7 * weight) + (5 * height) - (6.8 * age) * rate
    else:
        res = 65.5 + (9.6 * weight) + (1.8 * height) - (4.7 * age) * rate
    return res


def loss_calorie_per_day(height, weight, age, gender, goal_weight, days=30, rate=1.2):
    """
    一千克的脂肪=7700千卡的热量

    :param height:
    :param weight:
    :param age:
    :param gender:
    :param goal_weight:
    :param day:
    :param rate:
    :return:
    """
    base = cal_resting_metal(height, weight, age, gender, rate=1.2)
    distance = goal_weight * 7700 - base
    loss_calorie_each_day = distance / days
    loss_weight_each_day = loss_calorie_each_day / 7700
    return loss_calorie_each_day, loss_weight_each_day

if __name__ == '__main__':
    res = cal_resting_metal(178, 100, 27, "M")
    print(res)
