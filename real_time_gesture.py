#script to open camera with or without CUDA, then perform gesture detection on a live feed.

import cv2
import os
import argparse
import pyautogui as pyag

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np 
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from PIL import Image

####  Synthesizing code from http://www.pygaze.org/documentation/

from pygaze import libtime
from pygaze.libscreen import Display, Screen
from pygaze.libinput import Keyboard
from pygaze.eyetracker import EyeTracker

import random
#####



def set_previous(prevs, thresh=2, command=0):
    #shift the prevs array, set the most recent action
    if len(prevs) == 0: return 
    for i in range(1,thresh):
        prevs[thresh-i]=prevs[thresh-1-i]
    prevs[0]=command
    
def check_previous(prevs, thresh=2, command=0): 
    # returns true if the last consecutive (thresh) images are all equal
    if len(prevs) == 0: return True # prev_considered=1
    if prevs[0]==0: return True # most recent is non-gesture, send that info
    for i in range(thresh):
        if (command != prevs[i]):
            return False
    return True

def main():
    img_counter = 0
    mouse_clicked =  False
    
    eyetracker.start_recording()
    trialstart = libtime.get_time()
    points = 0
    t0 = libtime.get_time()
    tstim = libtime.get_time()
    
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
                gazepos = eyetracker.sample()
                pyag.moveTo(gazepos)
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
                    
                    if mode == "arrow" and command>0: #arrow key mode
                        print(cmds[command], "COMMAND SENT: ",command)
                        pyag.press(keymap[cmds[command]])
                    elif mode == "mouse":
                        #x, y = pyag.position()
                        
                        if command == 4 and not mouse_clicked:
                            print("mouse pressed")
                            mouse_clicked = True
                            pyag.mouseDown()
                        elif command != 4:
                            if mouse_clicked: print("mouse released")
                            mouse_clicked = False
                            pyag.mouseUp()
                                
                    set_previous(prevs, previous_detections_considered, command)

                else:    
                    #print(cmds[command])
                    set_previous(prevs, previous_detections_considered, command)
                    
    trialend = libtime.get_time()
    eyetracker.stop_recording()
    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    #optional arguments
    description  = "usage: python real_time_gesture.py --interval=[int] --prev_considered=[int] --mode=['arrow' or 'mouse'] \nNOTE: Experimental only, not for practical usage"
    parser = argparse.ArgumentParser(prog="real_time_gesture.py",usage=description)
    opt_group = parser.add_argument_group("Options")
    opt_group.add_argument('--prev_considered', dest='previous_detections_considered', action='store', default=2, type=int, help='how many consecutive times should we identify a gesture before sending a command?')
    opt_group.add_argument('--interval', dest='interval', action='store', default=15, type=int, help='how many frames pass before we analyze another one? (use lower number if using CUDA)')
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

    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    #fps = cam.get(cv2.CAP_PROP_FPS)
    interval = args.interval
    #interval = int(fps/interval)
    previous_detections_considered = args.previous_detections_considered -1 # -1 because we dont track the current action
    prevs = [] #array of previous detections
    for i in range(previous_detections_considered):
        prevs.append(0)    
    mode = args.mode
    if mode != "arrow" and mode != "mouse": 
        mode = "arrow"
    print("Mode: ", mode)
    
    model_path = "resnext50_9324.pth"
    model_path = os.path.join(os.getcwd(), model_path)
    
    if torch.cuda.is_available():
        model = torch.load(model_path)
    else:
        model = torch.load(model_path, map_location='cpu')
    model.eval()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available(): print("Using CUDA gpu") 
    else: print("CUDA not available")
    cv2.namedWindow("webcam feed")

    keymap = { #for arrow mode only
    "Palm Flat": 'down',
    "Thumbs Up": 'up', 
    "Point Right": 'right', 
    "Point Left": 'left' 
    }
    cmds = ["None", "Palm Flat", "Point Left", "Point Right", "Thumbs Up"]
        
    # # # # #
    # eye tracking prep
    ####  similarly taken from http://www.pygaze.org/documentation/
    
    # create keyboard object
    keyboard = Keyboard()
    
    # display object
    disp = Display()
    
    # screen objects
    screen = Screen()
    blankscreen = Screen()
    hitscreen = Screen()
    hitscreen.clear(colour=(0,255,0))
    misscreen = Screen()
    misscreen.clear(colour=(255,0,0))
    
    # create eyelink objecy
    eyetracker = EyeTracker(disp)
    
    # eyelink calibration
    eyetracker.calibrate()
    
    
    main()
    
