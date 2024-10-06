#archivo para testear que las funciones de predictor funcionen correctamente, fase previa a implementación de endpoints
#para que funcione correctamente, borrar 'src.' de las dependencias de predictor
#volver a insertarlas una vez concluida las pruebas en este archivo. 
from ultralytics import YOLO
import numpy as np
import cv2

import sys
import os

# Get the current directory
curr_dir = os.path.dirname(os.path.abspath(__file__))


from predictor import annotate_segmentation, GunDetector
model_path='yolov8n-seg.pt'

detector=GunDetector()

model=YOLO(model_path)
#img_path='/Users/dylanjitton/Documents/topicos IA/1er parcial/1er parcial/topicos-ia-2024-1er-parcial/screen_shot_2017-06-23_at_5.53.28_pm.png'
img_path="/Users/dylanjitton/Documents/topicos IA/1er parcial/1er parcial/topicos-ia-2024-1er-parcial/gun1.jpg"
#img_path="/Users/dylanjitton/Documents/topicos IA/1er parcial/1er parcial/topicos-ia-2024-1er-parcial/gun2.jpg"
#img_path="/Users/dylanjitton/Documents/topicos IA/1er parcial/1er parcial/topicos-ia-2024-1er-parcial/gun6.jpg"

results=model(img_path)
image=cv2.imread(img_path)

# Ensure the image is correctly loaded
if image is None:
    raise ValueError("Image could not be loaded. Check the file path.")

# Segment people and detect guns
segmentation_result = detector.segment_people(image, threshold=0.5, max_distance=10)

# Annotate the image with segmentation results
annotated_image = annotate_segmentation(image, segmentation_result, draw_boxes=True)

# Save or display the result
output_image_path = "/Users/dylanjitton/Documents/topicos IA/1er parcial/1er parcial/topicos-ia-2024-1er-parcial/annotated_image.png"
cv2.imwrite(output_image_path, annotated_image)

# If you want to display the image directly:
cv2.imshow("Annotated Image", annotated_image)
cv2.waitKey(0)  # Press any key to close the image window
cv2.destroyAllWindows()


