# coding: utf-8
import os
import sys
import subprocess
import time
import random
from PIL import Image
import cv2
from torchvision import transforms

def pull_screenshot(file_name):
    # Get screenshot
    process = subprocess.Popen('adb shell screencap -p', shell=True, stdout=subprocess.PIPE)
    screenshot = process.stdout.read()
    f = open(file_name, 'wb')
    f.write(screenshot)
    f.close()

def preprocess(image):
    # Crop and normalize image
    w, h = image.size
    top =  (h - w)/2

    image = image.crop((0,top,w,w+top))
    image = image.convert('RGB')
    image = image.resize((224,224), resample=Image.LANCZOS)

    normalize = transforms.Normalize(mean=[0.92206, 0.92206, 0.92206], std=[0.08426, 0.08426, 0.08426])
    transform = transforms.Compose([transforms.ToTensor(), normalize])

    image = transform(image)

    return image

def jump(press_time, swipe_x1, swipe_y1, swipe_x2, swipe_y2):
    press_time = int(press_time)
    cmd = 'adb shell input swipe {x1} {y1} {x2} {y2} {duration}'.format(
        x1=swipe_x1,
        y1=swipe_y1,
        x2=swipe_x2,
        y2=swipe_y2,
        duration=press_time
    )
    os.system(cmd)

def get_reward(file_name):
    # Compute action reward
    # Failed = 0, Other = Game Score
    pass

def restart(file_name):
    # Check game over and restart
    pass
