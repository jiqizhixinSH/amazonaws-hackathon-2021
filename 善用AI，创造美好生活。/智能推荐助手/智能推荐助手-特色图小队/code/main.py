import boto3
import json
import cv2
import test

if __name__ == "__main__":

    imageFile = 'RGB_video_20181119_145801_00038.jpg'
    client = boto3.client('rekognition')

    with open(imageFile, 'rb') as image:
        response = client.detect_labels(Image={'Bytes': image.read()})

    # with open('save.json','r') as f:
    #     response = json.load(f)
    data = cv2.imread(imageFile, 1)
    height, width, channel = data.shape

    print('Detected labels in ' + imageFile)
    boxWidth, boxHeight, boxLeft, boxTop =0,0,0,0
    for label in response['Labels']:
        print(label['Name'] + ' : ' + str(label['Confidence']))
        if label['Name'] == 'Person' and  float(label['Confidence'])>80:
            boudingbox = label['Instances'][0]
            boudingbox = boudingbox['BoundingBox']
            boxWidth = float(boudingbox['Width']) * width
            boxHeight = float(boudingbox['Height']) * height
            boxLeft = float(boudingbox['Left']) * width
            boxTop = float(boudingbox['Top']) * height
            boxWidth, boxHeight, boxLeft, boxTop = int(boxWidth), int(boxHeight), int(boxLeft), int(boxTop)
            break

    if not boxWidth:
       raise AssertionError

    for label in response['Labels']:
        if label['Name'] == 'Glasses' and float(label['Confidence'])>80:
            #print('-----------recommend sunglasses and blackglasses-----')
            test.glassRecommend(imageFile,label)
            cv2.destroyAllWindows()
        if label['Name'] == 'Car' and  float(label['Confidence'])>80:
            #print('-----------recommend sunglasses and blackglasses-----')
            test.carRecommend(imageFile,boxWidth, boxHeight, boxLeft, boxTop )
            #cv2.destroyAllWindows()
        if label['Name'] == 'Smile' and  float(label['Confidence'])>80:
            test.smileRecmmend(imageFile,boxWidth, boxHeight, boxLeft, boxTop)

    print('Done...')

    with open('save.json','w') as f:
        json.dump(response,f,indent=4)