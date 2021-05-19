"""
Copyright: Qinzi Now, Tencent Cloud.
"""
import sys
sys.path.append("..")
from configparser import ConfigParser


def get_info():
    cfg = ConfigParser()
    path = '../resources/config.ini'
    cfg.read(path)
    s_id = cfg.get("apiconfig", "secret_id")
    s_key = cfg.get('apiconfig', "secret_key")
    return s_id, s_key


if __name__ == '__main__':
    print(get_info())