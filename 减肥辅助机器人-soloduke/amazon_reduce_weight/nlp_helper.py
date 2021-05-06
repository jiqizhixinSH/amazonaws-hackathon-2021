# !/usr/bin/python
# -*- coding:UTF-8 -*-
# -----------------------------------------------------------------------#
# File Name: nlp_helper
# Author: Junyi Li
# Mail: 4ljy@163.com
# Created Time: 2021-05-03
# Description:
# -----------------------------------------------------------------------#
import re

cal_resting_metal_demo = "咨询：我的体重是100kg，身高是178，年龄是27，性别是男，目标体重是70kg，想要在100天内达到目标体重"
loss_calorie_per_day_demo = "咨询：我的体重是100kg，身高是178，年龄是27，性别是男，静息代谢率是多少"


def re_helper(input_sentence):
    if "静息代谢" not in input_sentence:
        cal_resting_metal = r"咨询：我的体重是(.*)kg，身高是(.*?)，年龄是(.*?)，性别是(.*?)，目标体重是(.*?)kg，想要在(.*?)天内达到目标体重"
        match_obj = re.match(cal_resting_metal, input_sentence)
        output = []
        if not match_obj:
            return output
        for idx in range(1, 7):
            output.append(match_obj.group(idx))
    else:
        loss_calorie_per_day = r"咨询：我的体重是(.*)kg，身高是(.*)，年龄是(.*)，性别是(.*)，静息代谢率是多少"
        match_obj = re.match(loss_calorie_per_day, input_sentence)
        output = []
        if not match_obj:
            return output
        for idx in range(1, 5):
            output.append(match_obj.group(idx))
    return output


if __name__ == '__main__':
    res = re_helper(cal_resting_metal_demo)
    print(res)
