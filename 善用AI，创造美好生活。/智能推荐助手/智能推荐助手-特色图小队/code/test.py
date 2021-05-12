import boto3
import json
import cv2
import PySimpleGUI as sg

def glassRecommend(image,response):
    print('-----------recommend sunglasses and blackglasses-----')
    data = cv2.imread(image,1)
    height,width,channel = data.shape
    boudingbox = response['Instances'][0]
    boudingbox = boudingbox['BoundingBox']
    boxWidth = float(boudingbox['Width'])*width
    boxHeight = float(boudingbox['Height'])*height
    boxLeft = float(boudingbox['Left'])*width
    boxTop = float(boudingbox['Top'])*height
    boxWidth,boxHeight,boxLeft,boxTop = int(boxWidth),int(boxHeight),int(boxLeft),int(boxTop)
    cv2.rectangle(data, (boxLeft, boxTop), (boxLeft + boxWidth, boxTop + boxHeight), (0, 255, 0), 2)

    cv2.putText(data,'Do you like sunglasses?',(boxLeft + boxWidth, boxTop + boxHeight), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
    cv2.imshow('glass',data)
    cv2.waitKey(100)
    # if key == 32:
    #     sunglassesPic = cv2.imread('LOHO.jpg')
    #     cv2.imshow('recommend',sunglassesPic)
    #     cv2.waitKey()
    #cv2.destroyAllWindows()
    input= sg.popup_yes_no('recommend sunglasses?')
    print(input)
    if input=='Yes':
        sunglassesPic = cv2.imread('LOHO.jpg')
        cv2.imshow('recommend', sunglassesPic)
        cv2.waitKey(1000)

    #cv2.destroyAllWindows()


def carRecommend(image,boxWidth, boxHeight, boxLeft, boxTop):
    print('-----------recommend car-----')
    data = cv2.imread(image, 1)
    height, width, channel = data.shape
    # boudingbox = response['Person'][0]
    # boudingbox = boudingbox['BoundingBox']
    # boxWidth = float(boudingbox['Width']) * width
    # boxHeight = float(boudingbox['Height']) * height
    # boxLeft = float(boudingbox['Left']) * width
    # boxTop = float(boudingbox['Top']) * height
    # boxWidth, boxHeight, boxLeft, boxTop = int(boxWidth), int(boxHeight), int(boxLeft), int(boxTop)


    cv2.putText(data, 'Do you like car?', (boxLeft + boxWidth, boxTop + boxHeight), cv2.FONT_HERSHEY_PLAIN, 2.0,
                (0, 0, 255), 2)
    cv2.imshow('car', data)
    cv2.waitKey(100)
    input = sg.popup_yes_no('recommend car?')
    print(input)
    if input == 'Yes':
        sunglassesPic = cv2.imread('car.jpg')
        cv2.imshow('recommend', sunglassesPic)
        cv2.waitKey(1000)
    #return

def smileRecmmend(image,boxWidth, boxHeight, boxLeft, boxTop):
    print('-----------recommend followers-----')
    data = cv2.imread(image, 1)
    height, width, channel = data.shape
    cv2.putText(data, 'Do you like follwers?', (boxLeft + boxWidth, boxTop + boxHeight), cv2.FONT_HERSHEY_PLAIN, 2.0,
                (0, 0, 255), 2)
    cv2.imshow('smile', data)
    cv2.waitKey(100)
    input = sg.popup_yes_no('recommend followers?')
    print(input)
    if input == 'Yes':
        follwerData = cv2.imread('follwer.jpg')
        cv2.imshow('recommend', follwerData)
        cv2.waitKey(1000)
    #return



