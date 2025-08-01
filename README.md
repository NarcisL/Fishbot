# Fishbot
Metin2 fishbot python script made using AI

how to run:
open a terminal
navigate in the dataset folder
run this command:

yolo detect train data=data.yaml model=yolov8s.pt epochs=100 imgsz=640

!!IMPORTANT NOTE MY PC DUE TO THE LARGE AMMOUNT OF DATA IN THE DATASET THE TIME NEEDED TO COMPLETE 1 EPOCH IS ~17 minutes adjust the number of epochs accordingly to your system!!

before running the main.py script make sure you have all the requirments
if not
pip install pydirectinput
pip install pyautogui
pip install pillow
pip install opencv-python-headless
pip install ultralytics
pip install mss
pip install numpy


For this project a YOLO model has been trained using a set of ~1000 images detecting a moving fish.

Workflow:
make sure youre in the directory with the main.py
run the script
select the game client window
enjoy

How to stop:
go to the terminal in which the script is running and press ctrl + C

TO DO:
stop mechanism
improve model accuracy
adjust business logic

<img width="2250" height="1500" alt="BoxP_curve" src="https://github.com/user-attachments/assets/6b49358f-51b1-4e67-8791-30d1f2b98e48" />
<img width="2400" height="1200" alt="results" src="https://github.com/user-attachments/assets/19db29fb-64d2-4b02-83b0-26c1dc736f73" />

