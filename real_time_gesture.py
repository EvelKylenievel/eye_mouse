#script to open camera with or without CUDA, then perform gesture detection on a live feed.

import cv2
import time
import os
import ctypes 
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np 
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from PIL import Image


def set_previous(prevs, thresh=2, command=0):
    #shift the prevs array, set the most recent action
    for i in range(1,thresh):
        prevs[thresh-i]=prevs[thresh-1-i]
    prevs[0]=command
    
def check_previous(prevs, thresh=2, command=0): 
    # returns true if the last consecutive (thresh) images are all equal
    if prevs[0]==0: return False # not a gesture is never a command!
    for i in range(thresh):
        if (command != prevs[i]):
            return False
    return True

def up():
    ctypes.windll.user32.keybd_event(keymap["up"], 0, 0, 0) # press
    ctypes.windll.user32.keybd_event(keymap["up"], 0, 0x0002, 0) # release
def down():
    ctypes.windll.user32.keybd_event(keymap["down"], 0, 0, 0) 
    ctypes.windll.user32.keybd_event(keymap["down"], 0, 0x0002, 0)
def left():
    ctypes.windll.user32.keybd_event(keymap["left"], 0, 0, 0) 
    ctypes.windll.user32.keybd_event(keymap["left"], 0, 0x0002, 0)
def right():
    ctypes.windll.user32.keybd_event(keymap["right"], 0, 0, 0)
    ctypes.windll.user32.keybd_event(keymap["right"], 0, 0x0002, 0)
def left_click(x,y):
    ctypes.windll.user32.mouse_event(keymap["left_click"], ctypes.c_long(x), ctypes.c_long(y), 0, 0)
    left_status = True
def left_release(x,y):
    ctypes.windll.user32.mouse_event(keymap["left_release"], ctypes.c_long(x), ctypes.c_long(y), 0, 0)
    left_status = False
def right_click(x,y):
    ctypes.windll.user32.mouse_event(keymap["right_click"], ctypes.c_long(x), ctypes.c_long(y), 0, 0)
    right_status = True
def right_release(x,y):
    ctypes.windll.user32.mouse_event(keymap["right_release"], ctypes.c_long(x), ctypes.c_long(y), 0, 0)
    right_status = False

def main():
    img_counter = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        cv2.imshow("webcam feed", frame)
        
        k = cv2.waitKey(1)
        if k%256 == 27:  # ESC pressed
            print("Escape hit, closing...")
            break
        else:
            img_counter += 1
            img_name = "opencv_frame_{count}.png".format(count = img_counter)
            if (img_counter % interval == 0):
                cv2.imwrite(img_name, frame)
                img_pil = Image.open(img_name)
                img = data_transforms(img_pil)
                input_batch = img.unsqueeze(0)
                input_batch = input_batch.to(device)
                output = model(input_batch)
                _, predictions = torch.max(output, 1)
                
                command = int(predictions[0])

                os.remove(img_name)
                if (check_previous(prevs, previous_detections_considered, command)):
                    #if THRESH consecutive gestures were detected in a row, send command\
                    #### SEND COMMAND HERE ####
                    
                    if mode == "arrow": #arrow key mode
                        print(cmds[command], "COMMAND SENT: ",command)
                        if command == 1:
                            down()
                        elif command ==2:
                            left()
                        elif command ==3:
                            right()
                        elif command ==4:
                            up()
                    elif mode == "mouse": #cursor mode
                        coords = ctypes.wintypes.POINT()
                        ctypes.windll.user32.GetCursorPos(ctypes.byref(coords))
                        scaledX = 65536 * coords.x // w + 1
                        scaledY = 65536 * coords.y // h + 1
                        
                        if command == 2:
                            print(cmds[command], "COMMAND SENT: ",command)
                            left_click(scaledX, scaledY)
                            if right_status:
                                print("release right click")
                                right_release(scaledX, scaledY)
                        elif command == 3:
                            print(cmds[command], "COMMAND SENT: ",command)
                            right_click(scaledX, scaledY)
                            if left_status:
                                print("release left click")
                                left_release(scaledX,scaledY)
                        else:
                            if right_status:
                                print("release right click")
                                right_release(scaledX, scaledY)
                            if left_status:
                                print("release left click")
                                left_release(scaledX,scaledY)

                    set_previous(prevs, previous_detections_considered, command)

                else:    
                    print(cmds[command])
                    set_previous(prevs, previous_detections_considered, command)
                    
    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    #optional arguments
    #For best performance, powerful PCs should lower the interval and increase the previous detections considered.
    description  = "usage: python real_time_gesture.py --interval=[int] --prev_considered=[int] --mode=['arrow' or 'mouse'] \nNOTE: Experimental only, not for practical usage"
    parser = argparse.ArgumentParser(prog="real_time_gesture.py",usage=description)
    opt_group = parser.add_argument_group("Options")
    opt_group.add_argument('--interval', dest='interval', action='store', default=12, type=int, help='how many frames pass before we analyze another one? (use lower number if using CUDA)')
    opt_group.add_argument('--prev_considered', dest='previous_detections_considered', action='store', default=2, type=int, help='how many consecutive times should we identify a gesture before sending a command?')
    opt_group.add_argument('--mode', dest='mode', action='store', default="arrow", type=str, help='arrow mode: gestures correlate to the arrow keys. mouse mode: gestures correlate to left and right click')
    
    args = parser.parse_args() 
    
    data_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    print("Booting Camera")
    print("'esc' to close")

    cam = cv2.VideoCapture(0)
    fps = cam.get(cv2.CAP_PROP_FPS)

    interval = args.interval
    interval = int(fps/interval)
    previous_detections_considered = args.previous_detections_considered -1 # -1 because we dont track the current action
    mode = args.mode
    if mode != "arrow" and mode != "mouse":
        mode = "arrow"
    model_path = "resnext50_9324.pth"
    model_path = os.path.join(os.getcwd(), "senior_des", "eye_mouse", model_path)
    
    print("Mode: ", mode)
    
    if torch.cuda.is_available():
        model = torch.load(model_path)
    else:
        model = torch.load(model_path, map_location='cpu')
    model.eval()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available(): print("Using CUDA gpu") 
    else: print("CUDA not available")
    cv2.namedWindow("webcam feed")
    
    if mode == "arrow":
        keymap = {
        'left': 0x25,
        'up': 0x26, 
        'right': 0x27, 
        'down': 0x28 
        }
        cmds = ["None", "Palm Flat", "Point Left", "Point Right", "Thumbs Up"]
    elif mode == "mouse":
        keymap = {
        'left_click': 0x0002,
        'left_release': 0x0004,
        'right_click': 0x008, 
        'right_release': 0x010 
        }
        cmds = ["None", "Palm Flat", "Point Left", "Point Right", "Thumbs Up"]
    
    w,h = (ctypes.windll.user32.GetSystemMetrics(0), ctypes.windll.user32.GetSystemMetrics(1)) # used with clicking
    left_status = False
    right_status = False
    prevs = [] #array of previous detections
    for i in range(previous_detections_considered):
        prevs.append(0)
    
    main()
    