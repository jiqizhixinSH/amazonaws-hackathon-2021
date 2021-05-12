# -*- coding: utf-8 -*-

import boto3
import pyaudio
from pyaudio import PyAudio, paInt16
import numpy as np
from datetime import datetime
import wave


ACCESS_ID = 'AKIAQ467PMINGHX5FVLO'
ACCESS_KEY = '6gqRjYqqMLMv2R85WIY/TKgVQFj2MetEep0P7F5Z'


def get_resource():
    s3 = boto3.resource('s3',
         aws_access_key_id=ACCESS_ID,
         aws_secret_access_key= ACCESS_KEY)

    s3 = boto3.resource('s3')
    # Print out bucket names
    for bucket in s3.buckets.all():
        print(bucket.name)

    # boto3.set_stream_logger('botocore', level='DEBUG')
    return


def get_information():
    return


def get_voice():
    # pip install pyaudio
    # pip install numpy
    # http://people.csail.mit.edu/hubert/pyaudio/
    """PyAudio example: Record a few seconds of audio and save to a WAVE file."""

    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 8000
    RECORD_SECONDS = 10
    WAVE_OUTPUT_FILENAME = "output.wav"

    p = pyaudio.PyAudio()  # 实例化PyAudio ，它设置portaudio系统

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)  # 在所需设备上打开所需音频参数的流

    print("* recording")

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')  # 打开写入文件
    wf.setnchannels(CHANNELS)  # 设置通道
    wf.setsampwidth(p.get_sample_size(FORMAT))  # 设置带宽
    wf.setframerate(RATE)  # 设置频率

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)  # 读进一个buffer
        wf.writeframes(data)

    print("* done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf.close()  # 关闭写入文件


def create_bucket():
    """
    Create bucket and edit
    Ref: https://docs.aws.amazon.com/AmazonS3/latest/userguide/enable-event-notifications.html#SettingBucketNotifications-enable-events
    Ref: https://docs.aws.amazon.com/AmazonS3/latest/userguide/UsingBucket.html

    :return:
    """
    # # Following: create bucket in sagemaker jupyter notebook instance
    # s3 = boto3.resource('s3')
    # my_region = boto3.session.Session().region_name
    # print(my_region)
    # s3.create_bucket(Bucket=bucket_name, CreateBucketConfiguration={'LocationConstraint': my_region})
    # # Delete bucket
    # bucket_to_delete = boto3.resource('s3').Bucket(bucket_name)
    # bucket_to_delete.objects.all().delete()

    # Method 1
    bucket_name = 'yangxy202105071538'  # <--- CHANGE THIS VARIABLE TO A UNIQUE NAME FOR YOUR BUCKET
    client = boto3.client(
        's3',
        region_name='us-east-2',
        aws_access_key_id=ACCESS_ID,
        aws_secret_access_key=ACCESS_KEY
    )
    client.create_bucket(
        Bucket=bucket_name,
        CreateBucketConfiguration={
            'LocationConstraint': 'us-east-2',
        },
    )
    return


def load2bucket(filename):
    """
    Upload file to bucket
    :param filename:
    :return:
    """
    bucket_name = 'yangxy202105071538'  # <--- CHANGE THIS VARIABLE TO A UNIQUE NAME FOR YOUR BUCKET
    client = boto3.client(
        's3',
        region_name='us-east-2',
        aws_access_key_id=ACCESS_ID,
        aws_secret_access_key=ACCESS_KEY
    )

    client.upload_file(filename,
                       Bucket=bucket_name,
                       Key='output.wav'
                       )
    return


def delete_bucket():
    bucket_name = 'yangxy202105071538'  # <--- CHANGE THIS VARIABLE TO A UNIQUE NAME FOR YOUR BUCKET
    client = boto3.client(
        's3',
        region_name='us-east-2',
        aws_access_key_id=ACCESS_ID,
        aws_secret_access_key=ACCESS_KEY
    )
    client.delete_bucket(
        Bucket=bucket_name,
        )
    return


def get_object_url():
    bucket_name = 'yangxy202105071538'  # <--- CHANGE THIS VARIABLE TO A UNIQUE NAME FOR YOUR BUCKET
    object_name = 'output.wav'
    client = boto3.client(
        's3',
        region_name='us-east-2',
        aws_access_key_id=ACCESS_ID,
        aws_secret_access_key=ACCESS_KEY
    )
    # Check object in the bucket.
    objs = client.list_objects(Bucket=bucket_name,
                               MaxKeys=10,
                               )
    for k,v in objs.items():
        print(k,v)

    res = client.generate_presigned_url('get_object',
                                            Params={'Bucket': bucket_name,
                                                    'Key': object_name},
                                             )
    print(res)
    return res


def get_texture():
    """
    Recognize texture from pictures.
    Use Amazon Textract.
    :return:
    """
    return

def transcribe():
    """
    Transfer audio input to language output.
    Use Amazon Transcribe.
    Ref: https://docs.aws.amazon.com/zh_cn/transcribe/latest/dg/what-is-transcribe.html
    Ref: https://aws.amazon.com/cn/blogs/machine-learning/analyzing-contact-center-calls-part-1-use-amazon-transcribe-and-amazon-comprehend-to-analyze-customer-sentiment/
    """
    client = boto3.client('transcribe',
                          region_name='us-east-2',
                          aws_access_key_id=ACCESS_ID,
                          aws_secret_access_key=ACCESS_KEY
                          )
    # 获取bucket中对应obj的uri
    bucket_name = 'yangxy202105071538'  # <--- CHANGE THIS VARIABLE TO A UNIQUE NAME FOR YOUR BUCKET
    object_name = 'output.wav'
    uri = "s3://{}/{}".format(bucket_name, object_name)
    
    jobname = 'transcribe_0507_01'

    client.start_transcription_job(
        TranscriptionJobName=jobname,
        # MediaSampleRateHertz=1024,
        MediaFormat='wav',
        Media={
            'MediaFileUri': uri
        },
        OutputBucketName='yangxy202105071538',
        OutputKey='transcribe_0507.json',
        IdentifyLanguage=True,
        
        # LanguageCode='zh-CN',
        # LanguageOptions=['zh-CN', 'en-GB',],
    )
    
    return


def get_transcribe_res():
    """
    Ref: https://docs.aws.amazon.com/zh_cn/AmazonS3/latest/userguide/download-objects.html
    """
    client = boto3.client('transcribe',
                          region_name='us-east-2',
                          aws_access_key_id=ACCESS_ID,
                          aws_secret_access_key=ACCESS_KEY
                          )
    
    jobname = 'transcribe_0507_01'
    response = client.get_transcription_job(
        TranscriptionJobName=jobname
    )
    return response


def conversation():
    """
    AI chat-robot.
    Use Amazon Lex.
    """
    # client = boto3.client('lexv2-models')
    return


def translate():
    """
    Translate input of source language to target language.
    Use Amazon Translate.
    Ref: https://docs.aws.amazon.com/zh_cn/translate/latest/dg/what-is.html#what-is-languages
    """
    client = boto3.client('translate',
                          region_name='us-east-2',
                          aws_access_key_id=ACCESS_ID,
                          aws_secret_access_key=ACCESS_KEY
                          )
    response = client.translate_text(
        Text='I am a child',
        SourceLanguageCode='auto',
        TargetLanguageCode='zh'
        # TerminologyNames=['string',],
    ) # return a dict
    return response['TranslatedText']


if __name__ == '__main__':
    # conversation()
    # res = translate()
    # print(res)
    # get_voice()
    # create_bucket()
    # load2bucket("./output.wav")
    # get_object_url()
    # delete_bucket()
    # transcribe()
    res = get_transcribe_res()
    print(res)
