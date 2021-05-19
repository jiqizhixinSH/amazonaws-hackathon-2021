import requests

# years_exp = [{"Age": 22, "Sex": "male", "Embarked": "S"},
#              {"Age": 22, "Sex": "female", "Embarked": "C"},
#              {"Age": 80, "Sex": "female", "Embarked": "C"},
#              {"Age": 22, "Sex": "male", "Embarked": "S"},
#              {"Age": 22, "Sex": "female", "Embarked": "C"},
#              {"Age": 80, "Sex": "female", "Embarked": "C"},
#              {"Age": 22, "Sex": "male", "Embarked": "S"},
#              {"Age": 22, "Sex": "female", "Embarked": "C"},
#              {"Age": 80, "Sex": "female", "Embarked": "C"},
#              {"Age": 22, "Sex": "male", "Embarked": "S"},
#              {"Age": 22, "Sex": "female", "Embarked": "C"},
#              {"Age": 80, "Sex": "female", "Embarked": "C"},
#              {"Age": 22, "Sex": "male", "Embarked": "S"},
#              {"Age": 22, "Sex": "female", "Embarked": "C"},
#              {"Age": 80, "Sex": "female", "Embarked": "C"},
#              ]
years_exp = [{"topic": '餐桌前', 'url': "https://ss3.bdstatic.com/70cFv8Sh_Q1YnxGkpoWK1HF6hhy/it/u=113789440,336919229&fm=26&gp=0.jpg"}]
response = requests.post(url='http://127.0.0.1:8000/predict', json=years_exp)
print(response)
result = response.json()
print('model API返回结果：', result)