#prueba con herramientas base, previo a implementación de funciones en predeictor.
import os
from ultralytics import YOLO
import numpy as np
import cv2

model_path = 'yolov8n-seg.pt'

# Cargar el modelo YOLO
model = YOLO(model_path)

# Cargar la imagen
img_path = '/Users/dylanjitton/Documents/topicos IA/1er parcial/1er parcial/topicos-ia-2024-1er-parcial/screen_shot_2017-06-23_at_5.53.28_pm.png'
image = cv2.imread(img_path)

# Obtener los resultados de la predicción
results = model(img_path)
predictions = results[0]

# Si hay máscaras de segmentación
if predictions.masks:
    # Obtener las máscaras de segmentación
    masks = predictions.masks.data.numpy()
    class_ids = predictions.boxes.cls.numpy()

    person_id=0
    # Definir los colores para las máscaras y las bounding boxes
    mask_color = (0, 255, 0)  # Verde para las máscaras
    bbox_color = (0, 0, 255)  # Rojo para las bounding boxes

    # Superponer las máscaras de segmentación en la imagen
    for i, (mask, class_id) in enumerate(zip(masks, class_ids)):
        if class_id == person_id:  # Solo procesar las máscaras con ID de persona
            # Ajustar el tamaño de la máscara al tamaño original de la imagen
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

            # Crear una máscara binaria donde se aplica la segmentación
            binary_mask = mask.astype(bool)
            
            # Aplicar el color de la máscara sobre la máscara binaria
            color_mask = np.zeros_like(image)
            color_mask[binary_mask] = mask_color

            # Superponer la máscara coloreada sobre la imagen original
            image = cv2.addWeighted(image, 1, color_mask, 0.5, 0)  # Ajustar la transparencia con el parámetro alpha

    # Obtener las bounding boxes y etiquetas
    boxes = predictions.boxes
    labels = predictions.boxes.cls.tolist()

    # Dibujar las bounding boxes sobre la imagen
    for i, box in enumerate(boxes.xyxy.tolist()):
        if labels[i] == person_id:  # Asegurarse de que solo dibujamos las bounding boxes de personas
            x1, y1, x2, y2 = [int(coord) for coord in box]  # Coordenadas de la bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), bbox_color, 2)  # Dibujar la caja
            label = predictions.names[labels[i]]
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bbox_color, 2)  # Etiqueta

    # Mostrar la imagen resultante con máscaras y bounding boxes
    cv2.imshow("Image with Person Segmentation Masks and Bounding Boxes", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    print("No se encontraron máscaras de segmentación.")