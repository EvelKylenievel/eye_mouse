#script to create a folder, then take pictures based on the gesture

import cv2

import os

print("This script will create a directory named whatever gesture you want and then open up a window showing your webcam, where it will take the number of screenshots you specified. To take pictures, press 'space', and press 'esc' to exit prematurely.")

foldername = input("Enter gesture name: ")

numpics = int(input("Enter number of pictures to take: "))

os.mkdir(foldername)


cam = cv2.VideoCapture(0)

cv2.namedWindow("webcam feed")

img_counter = 0

while img_counter < numpics:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("webcam feed", frame)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "opencv_frame_{count}_{gesture}.png".format(count = img_counter, gesture = foldername)
        save_name = os.path.join(foldername, img_name)
        cv2.imwrite(save_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1

cam.release()

cv2.destroyAllWindows()