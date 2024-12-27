import cv2
from ultralytics import YOLO
import os

# Define the font parameters
org = (20, 120)
font_face = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 4
font_color = (0, 0, 255)  # Red
font_thickness = 5
line_type = cv2.LINE_AA

# Load the YOLOv8 model
model = YOLO('runs/detect/CitrusTrees_yolov8x_custom/weights/best.pt')

# Open the video file
folder_path = "test_images"
folder = os.listdir("test_images")
for i in folder:
        img_path = f'{folder_path}'+"/"+f'{i}'
        results = model.predict(img_path,conf=0.1)
        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        Count = len(results[0])
        cv2.putText(annotated_frame, f"Tree Count:{Count}", org, font_face, font_scale, font_color, font_thickness, line_type)
        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        cv2.waitKey(5000)
cv2.destroyAllWindows()