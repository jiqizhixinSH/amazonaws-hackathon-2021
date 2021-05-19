"""
Copyright: Qinzi Now, Tencent Cloud.
"""
import sys
sys.path.append("..")
import json
from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
from tencentcloud.tiia.v20190529 import tiia_client, models
import json

import base64
from utils.config_helper import get_info

s_id, s_key = get_info()
cred = credential.Credential(s_id, s_key)


def rec_main(image_url,
             camera=True,
             album=False):
    try:
        httpProfile = HttpProfile()
        httpProfile.endpoint = "tiia.tencentcloudapi.com"

        clientProfile = ClientProfile()
        clientProfile.httpProfile = httpProfile
        client = tiia_client.TiiaClient(cred, "ap-beijing", clientProfile)

        req = models.DetectLabelRequest()
        params = {
            "ImageUrl": image_url,
            "Scenes": ["WEB", "CAMERA", "ALBUM"]
        }
        req.from_json_string(json.dumps(params))

        resp = client.DetectLabel(req)

        x = resp.to_json_string()
        # parse x:
        y = json.loads(x)

        if camera:
            return y['Response']['CameraLabels'][0]['Name']
        elif album:
            return y['Response']['AlbumLabels'][0]['Name']
        return y['Response']['Labels'][0]['Name']

    except TencentCloudSDKException as err:
        print(err)


def run():
    # 这是一张猫猫的照片
    # 输入照片一定是国内可以访问的，google不可以
    test_cat = "https://ss3.bdstatic.com/70cFv8Sh_Q1YnxGkpoWK1HF6hhy/it/u=113789440,336919229&fm=26&gp=0.jpg"
    rec_main(test_cat)


if __name__ == "__main__":
    run()
