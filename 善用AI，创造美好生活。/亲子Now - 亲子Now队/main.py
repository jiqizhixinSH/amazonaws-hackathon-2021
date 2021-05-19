"""
Copyright: Qinzi Now, Tencent Cloud.
"""
# from embedding.bert_embedding import run as bert_embed
# from scenery_rec.recognition import rec_main as recognition_main


# step 1: cold start
def cold_start():
    pass


# step 2: content system: using tencent cloud to build content embedding
# def embedding_(corpus):
#     for data in corpus:
#         bert_embed(data)


# step 3: scenery recognition
# def scenery_recognition(input_pic):
#     return recognition_main(input_pic)


if __name__ == '__main__':
    import pandas as pd
    from recommender.baseline_rs import main as recommender_system
    json_ = [{"topic": '餐桌前', 'url': "https://ss3.bdstatic.com/70cFv8Sh_Q1YnxGkpoWK1HF6hhy/it/u=113789440,336919229&fm=26&gp=0.jpg"}]
    temp = pd.DataFrame(json_)

    model_columns = ['topic', 'url']
    query = temp.reindex(columns=model_columns, fill_value=0)
    result = []
    for index, row in query.iterrows():
        result.append(recommender_system(row['topic']))
    print(result)
