"""
Copyright: Qinzi Now, Tencent Cloud

Reference: https://www.jianshu.com/p/ae2c8d042941
"""

# from flask import Flask, jsonify, request
# from scenery_rec.recognition import rec_main
# from utils.log import Log
#
# import json
#
# api_log = Log('api')
#
# app = Flask(__name__)  # 1.定义个一个Flask实例，__name__指定了程序主模块的名字，
# # Flask以此决定程序的根目录。即如果flask要从一个相对路径获取资源，根目录就是这个主
# # 模块所在的目录。如果你不清楚，直接写__name__就可以了。
# app.config['JSON_AS_ASCII'] = False  # 如果你返回的json数据数乱码，那么这个设定可以帮助你。
#
# class InputData(Form):
#     # 我们要对模型输入的数据，这个类定义了检查数据的方式。
#     # accept_content表示对输入数据中的 accept_content字段进行规范；StringField要求字
#     # 段必须是文本；validators参数指定检查器，可以有多个检查器，都放在[]里；DataRequired
#     # 意思是这个字段是必须的，其中的message表示如果没有这个输入这个数据，会提示什么文
#     # 字。
#     # accept_content = StringField(validators=[DataRequired(message='没有accept_content')])
#     # category_lv1 = StringField(validators=[DataRequired(message='没有输入category_lv1')])
#     # category_lv2 = StringField(validators=[DataRequired(message='没有输入category_lv2')])
#     # category_lv3 = StringField(validators=[DataRequired(message='没有输入category_lv3')])
#     pass
#
#
# # 下面这句这不是API脚本的内容，这个是用requests库调用api的方式。这句代码中：
# # requests.post表示执行post方法，对应的requests.get表示执行get方法；
# # http://127.0.0.1:5000/complaint就是我们指定的url，127.0.0.1是本机地址，是在本机测试
# # API时使用的，真正部署时，需要填服务器的API地址。5000是默认端口，也可以自己指
# # 定。complaint是自己的定义，也可以取其他名字；json = data_json中，json表示数据必须以
# # json格式输入，这是我们开发接口时限定的条件，也可以限定其他格式。data_json中就是模
# # 型需要的输入数据，为json格式。
# # requests.post('http://127.0.0.1:5000/complaint', json = data_json)
#
# @app.route('/complaint', methods=['post'])  # 这个装饰器定义了url的路径，并指定了调用方
# # 法必须是post
# def predition():  # 这是核心程序，当调用上述url时，就会执行这个函数。
#     api_log.info("get a request:referer{0} user_agent{1}".format(request.referrer, request.user_agent))  # 记录每一次请求者的信息
#     if not request.is_json:  # 判断输入是否json格式
#         return bad_request()  # 我们定义了一个bad_request函数，来处理输入数据不符合要求
#     # 时要怎么办。
#     else:
#         data = request.get_json()  # 获取json数据
#         try:
#             data = json.loads(data)  # 转成字典，这是我这边调用模型要求输入必须字典格式。
#         except:
#             return bad_request()  # 如果无法转成字典，说明请求中的数据不是json格式，不符合
#         # 要求。
#         # json解析成字典，再作为关键字参数传输进去，验证数据，这步是必须的，必须转成
#         # 字典。
#         input = InputData(**data)
#         if not input.validate():  # input.validate()执行验证，如果验证通过，返回True
#             # 如果没通过验证，input.errors返回错误信息，我们将错误信息返回给调用方，让他
#             # 们进行调整。
#             return jsonify(input.errors)
#         else:
#             # 运行模型，测试阶段，如果模型出错，会返回测试数据给调用方，方便他们根据测
#             # 试数据继续开发。这是我自己的个性化需求，因为我们的开发人员，需要获取响应数据，再
#             # 执行其他开发工作。他们不在乎返回是什么，只要有返回就行了。如果因为我们模型的问
#             # 题，导致无法返回数据，会影响他们的开发工作。所以我这边定义模型如果出错，返回测试
#             # 结果，同时我记录出错数据和错误信息，再进行debug。
#             try:
#                 data = input_column_to_eng(data)  # 将英文key转为中文，这是我的模型文件要求的
#         pre_result = predict(data)  # 执行模型运算
#         log.info("request data \n:{0}".format(data))  # 记录输出结果到日志文件
#         except:  # 测试期间，运行错误返回错误值，并记录错误原因，错误数据
#         log.error("=*20模型出错:", exc_info=True)
#         with open('cannot_predict.txt', 'a') as file:  # 记录出错数据
#             file.write(str(data) + '\n\n')
#         pre_result = {'测试数据': '测试结果'
#                       }
#         pre_result_code = result_to_code(pre_result)  # 转码输出结果，不用管
#         return jsonify(pre_result_code)  # 将模型结果，用json格式发送出去。
#
#
# # 一下装饰器，分别定义了各种接口调用错误的处理方式。要了解这些错误，需要去了解一
# # 下相关的web知识。
# @app.errorhandler(400)
# def bad_request(error=None):
#     message = {
#         'status': 400,
#         'message': 'Bad request:Please check your request, is it json type?'
#     }
#     resp = jsonify(message)
#     resp.status_code = 400
#     return resp
#
#
# @app.errorhandler(404)
# def not_found(e):
#     message = {
#         'status': 404,
#         'message': 'Notfound: please check your url'
#     }
#     resp = jsonify(message)
#     resp.status_code = 404
#     return resp
#
#
# @app.errorhandler(405)
# def Method_error(e):
#     message = {
#         'status': 405,
#         'message': 'Method not allow: please make sure your method is POST'
#     }
#     resp = jsonify(message)
#     resp.status_code = 405
#     return resp
#
#
# @app.errorhandler(500)
# def serve_error(e):
#     message = {
#         'status': 500,
#         'message': 'Internal serve error: Try again , or ask API developer for help.'
#     }
#     resp = jsonify(message)
#     resp.status_code = 500
#     return resp
