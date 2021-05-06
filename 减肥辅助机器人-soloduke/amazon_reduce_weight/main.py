# -*- coding: utf-8 -*-
# filename: main.py
import web
from wechat_helper.handle import Handle  # no qa

urls = (
    '/wx', 'Handle',
)

if __name__ == '__main__':
    app = web.application(urls, globals())
    app.run()
