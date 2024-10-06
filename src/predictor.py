from ultralytics import YOLO
import numpy as np
import cv2
from shapely.geometry import Polygon, box
from src.models import Detection, PredictionType, Segmentation,PersonType
from src.config import get_settings

SETTINGS = get_settings()


def match_gun_bbox(segment: list[list[int]], bboxes: list[list[int]], max_distance: int = 10) -> list[int] | None:
    matched_box = None
    segment_polygon=Polygon(segment)
    #min_distance=float('inf')

    for bbox in bboxes:
        x1,y1,x2,y2=bbox
        box_polygon=box(x1,y1,x2,y2)

        distance=segment_polygon.distance(box_polygon)

        if distance<max_distance:
            matched_box=bbox
    ### ========================== ###
    ### SU IMPLEMENTACION AQUI     ###
    ### ========================== ###

    return matched_box


def annotate_detection(image_array: np.ndarray, detection: Detection) -> np.ndarray:
    ann_color = (0, 0, 255)
    annotated_img = image_array.copy()
    for label, conf, box in zip(detection.labels, detection.confidences, detection.boxes):
        x1, y1, x2, y2 = box
        cv2.rectangle(annotated_img, (x1, y1), (x2,y2), ann_color, 3)
        cv2.putText(
            annotated_img,
            f"{label}: {conf:.1f}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            ann_color,
            2,
        )
    return annotated_img


def annotate_segmentation(image_array: np.ndarray, segmentation: Segmentation, draw_boxes: bool = True) -> np.ndarray:
    ann_color_safe = (0, 255, 0)  # Green for "safe"
    ann_color_danger = (255, 0, 0)  # Red for "danger"

    # Draw segmentation polygons
    for polygon, label in zip(segmentation.polygons, segmentation.labels):
        if label == PersonType.safe:
            color = ann_color_safe
        else:
            color = ann_color_danger

        # Draw the segmentation as contours on the image
        polygon_np = np.array(polygon, np.int32)
        #cv2.polylines(image_array, [polygon_np], isClosed=True, color=color, thickness=1)
        mask = np.zeros_like(image_array, dtype=np.uint8)

        # Fill the polygon (i.e., segmentation mask) with the color
        cv2.fillPoly(mask, [polygon_np], color)

        # Overlay the mask onto the image with transparency (alpha)
        alpha = 0.5  # You can adjust this to make the mask more or less transparent
        image_array = cv2.addWeighted( image_array, 1, mask, alpha, 0)  # Adjust transparency here

    # Optionally draw bounding boxes
    if draw_boxes:
        for box,label in zip(segmentation.boxes,segmentation.labels):
            if label == PersonType.safe:
                color = ann_color_safe
            else:
                color = ann_color_danger
            x1, y1, x2, y2 = box
            cv2.rectangle(image_array, (x1, y1), (x2, y2), color, 2)  # Blue boxes for detection
            cv2.putText(
            image_array,
            f"{label.value}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2,
        )

        ### ========================== ###
        ### SU IMPLEMENTACION AQUI     ###
        ### ========================== ###
    return image_array



class GunDetector:
    def __init__(self) -> None:
        print(f"loading od model: {SETTINGS.od_model_path}")
        self.od_model = YOLO(SETTINGS.od_model_path)
        print(f"loading seg model: {SETTINGS.seg_model_path}")
        self.seg_model = YOLO(SETTINGS.seg_model_path)

    def detect_guns(self, image_array: np.ndarray, threshold: float = 0.5):
        results = self.od_model(image_array, conf=threshold)[0]
        labels = results.boxes.cls.tolist()
        indexes = [
            i for i in range(len(labels)) if labels[i] in [3, 4]
        ]  # 0 = "person"
        boxes = [
            [int(v) for v in box]
            for i, box in enumerate(results.boxes.xyxy.tolist())
            if i in indexes
        ]
        confidences = [
            c for i, c in enumerate(results.boxes.conf.tolist()) if i in indexes
        ]
        labels_txt = [
            results.names[labels[i]] for i in indexes
        ]
        print("detect guns, labels txt: ",labels_txt)
        return Detection(
            pred_type=PredictionType.object_detection,
            n_detections=len(boxes),
            boxes=boxes,
            labels=labels_txt,
            confidences=confidences,
        )
    
    def segment_people(self, image_array: np.ndarray, threshold: float = 0.5, max_distance: int = 10):

        polygons=[]
        boxes=[]
        labels=[]
        n_detections=0

        # Detectar armas
        gun_detection = self.detect_guns(image_array, threshold=threshold)
        gun_bboxes = gun_detection.boxes
        
        #segmentation
        results=self.seg_model(image_array,conf=threshold)
        predictions=results[0]
        
        if predictions.masks:
            # Retrieve the segmentation masks
            masks = predictions.masks.data.numpy()
            class_ids=predictions.boxes.cls.numpy()

            person_id=0
           
            # Iterate over each mask and extract the points
            for i, (mask, class_id) in enumerate(zip(masks, class_ids)):
                if class_id == person_id:
                    # Convertir la máscara al tamaño de la imagen original
                    mask = cv2.resize(mask, (image_array.shape[1], image_array.shape[0]), interpolation=cv2.INTER_NEAREST)
                    
                    # Crear la binary mask, para reducir la cantidad de puntos a dibujar en el mask, a algo manejable para enviar en el JSON
                    binary_mask = (mask > 0.5).astype(np.uint8)
                    
                    # Encontrar los contornos en la binary mask
                    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    # Iterar sobre cada contorno y simplificar
                    for contour in contours:
                        epsilon = 0.01 * cv2.arcLength(contour, True)
                        approx = cv2.approxPolyDP(contour, epsilon, True)

                        # Obtener puntos simplificados como una lista de (x, y)
                        points = [(int(point[0][0]), int(point[0][1])) for point in approx]
                        
                     # Verificar si alguno de los puntos coincide con una detección de arma
                    gun_bbox = match_gun_bbox(points, gun_bboxes, max_distance=max_distance)
                    
                    if gun_bbox:
                        labels.append(PersonType.danger)
                    else:
                        labels.append(PersonType.safe)

                    n_detections += 1
                    polygons.append(points)
        else:
            print("No segmentation masks found.")

        for pol in polygons:
            print('polygons: ',len(pol))
        #print('polugons: ',polygons)
        print('n detections: ',n_detections)
        # Convert YOLO boxes to list of lists of integers
        boxes = [
            box[0] for box, class_id in zip([box.xyxy.cpu().numpy().astype(int).tolist() for box in predictions.boxes], class_ids)
            if class_id == person_id
        ]
        print("boxes: ",boxes)
        # Create labels using class IDs
        #labels = ['person' if class_id == person_id else 'unknown' for class_id in class_ids]
        print("labels: ",labels)
        
        ### ========================== ###
        ### SU IMPLEMENTACION AQUI     ###
        ### ========================== ###

        return Segmentation(
            pred_type=PredictionType.segmentation,
            n_detections=n_detections,#cuantas personas detecta
            polygons=polygons,
            boxes=boxes,
            labels=labels
        )
