# -*- coding: utf-8 -*-

import numpy as np
import json

import boto3
import pyaudio
from pyaudio import PyAudio, paInt16
import wave
from playsound import playsound


ACCESS_ID = 'AKIAQ467PMINGHX5FVLO'
ACCESS_KEY = '6gqRjYqqMLMv2R85WIY/TKgVQFj2MetEep0P7F5Z'
BUCKET_NAME = 'aws-ai-communicate-demo-202105'  # <--- CHANGE THIS VARIABLE TO A UNIQUE NAME FOR YOUR BUCKET

AUDIO_INPUT = 'input.wav'
TRANSCRIBE_OUTPUT = 'transcribe_output.json'
TRANSCRIBE_JOBNAME = 'transcribe_2105'


class Agent(object):
    def __init__(self):
        super(Agent, self).__init__()
        self.ACCESS_ID = 'AKIAQ467PMINGHX5FVLO'
        self.ACCESS_KEY = '6gqRjYqqMLMv2R85WIY/TKgVQFj2MetEep0P7F5Z'
        self.BUCKET_NAME = 'awsaitranslatedemo202105'
        self.AUDIO_INPUT = 'input.wav'
        self.AUDIO_OUTPUT_PREFIX = 'speech'
        self.TRANSCRIBE_OUTPUT = 'transcribe_output.json'
        self.TRANSCRIBE_JOBNAME = 'transcribe_2105'

    # Query

    def get_resource(self,):
        s3 = boto3.resource('s3',
             aws_access_key_id=self.ACCESS_ID,
             aws_secret_access_key= self.ACCESS_KEY)

        s3 = boto3.resource('s3')
        # Print out bucket names
        for bucket in s3.buckets.all():
            print(bucket.name)

        # boto3.set_stream_logger('botocore', level='DEBUG')
        return

    # Resource

    def create_bucket(self,):
        """
        Create bucket and edit
        Ref: https://docs.aws.amazon.com/AmazonS3/latest/userguide/enable-event-notifications.html#SettingBucketNotifications-enable-events
        Ref: https://docs.aws.amazon.com/AmazonS3/latest/userguide/UsingBucket.html

        :return:
        """
        # Method 1
        client = boto3.client(
            's3',
            region_name='us-east-2',
            aws_access_key_id=self.ACCESS_ID,
            aws_secret_access_key=self.ACCESS_KEY
        )
        client.create_bucket(
            Bucket=self.BUCKET_NAME,
            CreateBucketConfiguration={
                'LocationConstraint': 'us-east-2',
            },
        )
        return

    def delete_bucket(self, bn):
        client = boto3.client(
            's3',
            region_name='us-east-2',
            aws_access_key_id=self.ACCESS_ID,
            aws_secret_access_key=self.ACCESS_KEY
        )
        client.delete_bucket(
            Bucket=bn,
        )
        return

    def load2bucket(self, filename):
        """
        Upload file to bucket
        :param filename:
        :return:
        """
        client = boto3.client(
            's3',
            region_name='us-east-2',
            aws_access_key_id=self.ACCESS_ID,
            aws_secret_access_key=self.ACCESS_KEY
        )

        client.upload_file(filename,
                           Bucket=self.BUCKET_NAME,
                           Key=filename
                           )
        return

    def get_bucket_file(self, key=TRANSCRIBE_OUTPUT, filename=TRANSCRIBE_OUTPUT):
        """
        Ref: https://docs.aws.amazon.com/zh_cn/AmazonS3/latest/userguide/download-objects.html
        """
        s3 = boto3.client(
            's3',
            region_name='us-east-2',
            aws_access_key_id=self.ACCESS_ID,
            aws_secret_access_key=self.ACCESS_KEY
        )

        s3.download_file(Bucket=self.BUCKET_NAME, Key=key, Filename=filename)
        with open(filename, 'r', encoding='utf-8') as f:
            res = json.loads(f.read())
        text = res['results']['transcripts'][0]['transcript']
        print(text)
        return text

    # Obtain information

    def get_voice(self,):
        # pip install pyaudio
        # pip install numpy
        # http://people.csail.mit.edu/hubert/pyaudio/
        """PyAudio example: Record a few seconds of audio and save to a WAVE file."""

        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 8000
        RECORD_SECONDS = 10
        # WAVE_OUTPUT_FILENAME = "input.wav"

        p = pyaudio.PyAudio()  # 实例化PyAudio ，它设置portaudio系统

        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)  # 在所需设备上打开所需音频参数的流

        print("* recording")

        wf = wave.open(self.AUDIO_INPUT, 'wb')  # 打开写入文件
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

    def get_texture(self,):
        """
        Recognize texture from pictures.
        Use Amazon Textract.
        :return:
        """
        return

    # Execute

    def transcribe_core(self, object_name='input.wav', jobname='transcribe_2105', output_key='transcribe_output.json'):
        """
        Transfer audio input to language output.
        Use Amazon Transcribe.
        Ref: https://docs.aws.amazon.com/zh_cn/transcribe/latest/dg/what-is-transcribe.html
        Ref: https://aws.amazon.com/cn/blogs/machine-learning/analyzing-contact-center-calls-part-1-use-amazon-transcribe-and-amazon-comprehend-to-analyze-customer-sentiment/
        """
        client = boto3.client('transcribe',
                              region_name='us-east-2',
                              aws_access_key_id=self.ACCESS_ID,
                              aws_secret_access_key=self.ACCESS_KEY
                              )
        # 获取bucket中对应obj的uri
        uri = "s3://{}/{}".format(self.BUCKET_NAME, object_name)
        try:
            client.start_transcription_job(
                TranscriptionJobName=jobname,
                # MediaSampleRateHertz=1024,
                MediaFormat='wav',
                Media={
                    'MediaFileUri': uri
                },
                OutputBucketName=self.BUCKET_NAME,
                OutputKey=output_key,
                IdentifyLanguage=True,
    
                # LanguageCode='zh-CN',
                # LanguageOptions=['zh-CN', 'en-GB',],
            )
        except:
            # There exists the same transcribe job
            self.delete_transcribe_job(self.TRANSCRIBE_JOBNAME)
            client.start_transcription_job(
                TranscriptionJobName=jobname,
                # MediaSampleRateHertz=1024,
                MediaFormat='wav',
                Media={
                    'MediaFileUri': uri
                },
                OutputBucketName=self.BUCKET_NAME,
                OutputKey=output_key,
                IdentifyLanguage=True,
    
                # LanguageCode='zh-CN',
                # LanguageOptions=['zh-CN', 'en-GB',],
            )

        # Check the transcribe job.
        # If completed, then delete the job.
        # print('Check transcribe job status ...')
        count = 0
        while True:
            response = client.get_transcription_job(
                TranscriptionJobName=jobname
            )
            if response['TranscriptionJob']['TranscriptionJobStatus'].lower() == 'completed' \
            or response['TranscriptionJob']['TranscriptionJobStatus'].lower() == 'failed':
                client.delete_transcription_job(
                    TranscriptionJobName=jobname
                )
                print('The transcribe job {} is finished. Now delete the job.'.format(jobname))
                break
            count += 1
            if count > 100:
                break

        return
    
    def delete_transcribe_job(self, jobname):
        client = boto3.client('transcribe',
                              region_name='us-east-2',
                              aws_access_key_id=self.ACCESS_ID,
                              aws_secret_access_key=self.ACCESS_KEY
                              )
        client.delete_transcription_job(
            TranscriptionJobName=jobname
        )
        print('Finish deleting the job: ', jobname)
        return

    def transcribe(self,):
        """
        Transcribe the recorded audio.
        :return:
        """
        self.get_voice()
        self.load2bucket(self.AUDIO_INPUT)
        self.transcribe_core(object_name=self.AUDIO_INPUT, jobname=self.TRANSCRIBE_JOBNAME, output_key=self.TRANSCRIBE_OUTPUT)
        text = self.get_bucket_file(key=self.TRANSCRIBE_OUTPUT, filename=self.TRANSCRIBE_OUTPUT)
        return text

    def translate(self, text='I am a child', source='auto', target='zh'):
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
            Text=text,
            SourceLanguageCode=source,
            TargetLanguageCode=target
            # TerminologyNames=['string',],
        )  # return a dict
        return response['TranslatedText']

    def speech(self, text):
        client = boto3.client('polly',
                              region_name='us-east-2',
                              aws_access_key_id=ACCESS_ID,
                              aws_secret_access_key=ACCESS_KEY
                              )
        try:
            response = client.start_speech_synthesis_task(
                OutputFormat='mp3',
                OutputS3BucketName=self.BUCKET_NAME,
                OutputS3KeyPrefix=self.AUDIO_OUTPUT_PREFIX,
                Text=text,
                VoiceId='Amy',
            )
        except:
            raise NotImplementedError
            
        task_id = response['SynthesisTask']['TaskId']
        audio_name = self.AUDIO_OUTPUT_PREFIX+'.'+task_id+'.mp3'
        
        # print('Check speech job status ...')
        count = 0
        while True:
            response = client.get_speech_synthesis_task(
                TaskId=task_id
            )
            if response['SynthesisTask']['TaskStatus'].lower() == 'completed' \
            or response['SynthesisTask']['TaskStatus'].lower() == 'failed':
                break
            count += 1
            if count > 100 or response['SynthesisTask']['TaskStatus'].lower() == 'failed':
                print('Fail to get speech.')
                break
        
        # Play the speech
        s3 = boto3.resource('s3',
                            region_name='us-east-2',
                            aws_access_key_id=ACCESS_ID,
                            aws_secret_access_key=ACCESS_KEY
                            )
        
        s3.Bucket(self.BUCKET_NAME).download_file(audio_name,
                                                  self.AUDIO_OUTPUT_PREFIX+'.mp3')

        playsound(self.AUDIO_OUTPUT_PREFIX+'.mp3')
        
        return

#    def conversation(self,):
#        """
#        AI chat-robot.
#        Use Amazon Lex.
#        """
#        # client = boto3.client('lexv2-models')
#        return


if __name__ == '__main__':
    agent = Agent()
    # agent.conversation()
    # res = agent.translate()
    # print(res)
    # agent.get_voice()
    # agent.create_bucket()
    # agent.load2bucket("./input.wav")
    # agent.delete_bucket(bucket_name)
    # agent.transcribe_core()
    # agent.get_transcribe_res()
    # agent.delete_transcribe_job(agent.TRANSCRIBE_JOBNAME)
    # agent.transcribe()
    text = 'Hello'
    agent.speech(text)