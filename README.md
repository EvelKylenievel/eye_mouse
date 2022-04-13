# eye_mouse
Quick project combining a previous project that detects real time gestures with an open source eye tracker to make a rudimentary hands-free UI.  The open source eye tracking software can be found here: http://www.pygaze.org/documentation/
make sure you calibrate your webcam first by following the pygaze guide.

# webcam
The machine learning model is quite biased and sensitive, but you can supplement its performance.

There are 2 modes, arrow keys input and mouse input.  For arrow key mode, position the camera facing your torso so your hand is contrasting against your clothing. For mouse mode, point the camera over one of your shoulders as the only gesture it looks for is a "thumbs up".

# dependencies
python 3.7.9
download the pytorch model and real time gesture python script

necessary libraries:
pytorch torchvision numpy pyatogui open-cv pygaze psychopy

>python real_time_gesture.py --mode=mouse
