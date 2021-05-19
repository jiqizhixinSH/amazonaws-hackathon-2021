# -- coding: utf-8 --
"""
Copyright: Qinzi Now, Tencent Cloud.
"""

import json
from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
from tencentcloud.nlp.v20190408 import nlp_client, models

import sys
import codecs
import base64
from utils.config_helper import get_info
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

s_id, s_key = get_info()
cred = credential.Credential(s_id, s_key)


def sen_embedding(input_sentence):
    try:
        httpProfile = HttpProfile()
        httpProfile.endpoint = "nlp.tencentcloudapi.com"

        clientProfile = ClientProfile()
        clientProfile.httpProfile = httpProfile
        client = nlp_client.NlpClient(cred, "ap-beijing", clientProfile)

        req = models.SentenceEmbeddingRequest()
        message_bytes = input_sentence.encode('utf-8') # 中文用utf-8 encode
        params = {
            "Content": str(base64.b64encode(message_bytes))[1:]
        }
        req.from_json_string(json.dumps(params))

        resp = client.SentenceEmbedding(req)
        print(resp.to_json_string())

    except TencentCloudSDKException as err:
        print(err)


def run():
    sen_embedding("为什么海水不能直接喝？")


if __name__ == "__main__":
    run()
