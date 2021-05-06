# !/usr/bin/python
# -*- coding:UTF-8 -*-
# -----------------------------------------------------------------------#
# File Name: tranlate
# Author: Junyi Li
# Mail: 4ljy@163.com
# Created Time: 2021-05-03
# Description:
# -----------------------------------------------------------------------#
import boto3


def translate_service(input_sentence="美女你好啊", src_lang="zh", tgt_lan="en", region_name="us-west-2"):
    translate = boto3.client(service_name='translate', region_name=region_name, use_ssl=True)

    result = translate.translate_text(Text=input_sentence,
                                      SourceLanguageCode=src_lang, TargetLanguageCode=tgt_lan)
    return result


def translate_helper(input_sentence="美女你好啊", src_lang="zh", tgt_lan="en", region_name="us-west-2"):
    result = translate_service(input_sentence=input_sentence, src_lang=src_lang, tgt_lan=tgt_lan,
                               region_name=region_name)
    return result.get('TranslatedText')


if __name__ == '__main__':
    result = translate_service(input_sentence="今天是一个测试")
    print('TranslatedText: ' + result.get('TranslatedText'))
    print('SourceLanguageCode: ' + result.get('SourceLanguageCode'))
    print('TargetLanguageCode: ' + result.get('TargetLanguageCode'))
