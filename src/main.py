import io
import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException, status, Depends
from fastapi.responses import Response, JSONResponse
import numpy as np
from functools import cache
from PIL import Image, UnidentifiedImageError
from src.predictor import GunDetector, Detection, Segmentation, annotate_detection, annotate_segmentation
from src.config import get_settings
from src.models import Gun, Person, PixelLocation

SETTINGS = get_settings()

app = FastAPI(title=SETTINGS.api_name, version=SETTINGS.revision)


@cache
def get_gun_detector() -> GunDetector:
    print("Creating model...")
    return GunDetector()


def detect_uploadfile(detector: GunDetector, file, threshold) -> tuple[Detection, np.ndarray]:
    img_stream = io.BytesIO(file.file.read())
    if file.content_type.split("/")[0] != "image":
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Not an image"
        )
    # convertir a una imagen de Pillow
    try:
        img_obj = Image.open(img_stream)
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Image format not suported"
        )
    # crear array de numpy
    img_array = np.array(img_obj)
    return detector.detect_guns(img_array, threshold), img_array

def detectPeople_uploadfile(detector: GunDetector, file, threshold) -> tuple[Detection, np.ndarray]:
    img_stream = io.BytesIO(file.file.read())
    if file.content_type.split("/")[0] != "image":
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Not an image"
        )
    # convertir a una imagen de Pillow
    try:
        img_obj = Image.open(img_stream)
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Image format not suported"
        )
    # crear array de numpy
    img_array = np.array(img_obj)
    return detector.segment_people(img_array, threshold), img_array

def polygon_area(points):
    n = len(points)  
    area = 0.0

    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]  # Siguiente punto (con % n para que el último conecte al primero)
        area += x1 * y2 - y1 * x2

    # Dividir entre 2 y tomar el valor absoluto
    return abs(area) / 2.0

@app.get("/model_info")
def get_model_info(detector: GunDetector = Depends(get_gun_detector)):
    return {
        "model_name": "Gun detector",
        "gun_detector_model": detector.od_model.model.__class__.__name__,
        "semantic_segmentation_model": detector.seg_model.model.__class__.__name__,
        "input_type": "image",
    }


@app.post("/detect_guns")
def detect_guns(
    threshold: float = 0.5,
    file: UploadFile = File(...),
    detector: GunDetector = Depends(get_gun_detector),
) -> Detection:
    results, _ = detect_uploadfile(detector, file, threshold)

    return results


@app.post("/annotate_guns")
def annotate_guns(
    threshold: float = 0.5,
    file: UploadFile = File(...),
    detector: GunDetector = Depends(get_gun_detector),
) -> Response:
    detection, img = detect_uploadfile(detector, file, threshold)
    annotated_img = annotate_detection(img, detection)

    img_pil = Image.fromarray(annotated_img)
    image_stream = io.BytesIO()
    img_pil.save(image_stream, format="JPEG")
    image_stream.seek(0)
    return Response(content=image_stream.read(), media_type="image/jpeg")

@app.post("/detect_people")
def detect_people(
    threshold: float = 0.5,
    file: UploadFile = File(...),
    detector: GunDetector = Depends(get_gun_detector),
)-> Segmentation:
    results,_=detectPeople_uploadfile(detector,file,threshold)
    #print(results)
    return results
    # response_data = {
    #     "n_detections": results.n_detections,
    #     "boxes": results.boxes,
    #     "labels": results.labels,
    #     "polygons": results.polygons#[0][:10]  # Limited results for demonstration purposes
    # }
    
    # return JSONResponse(content=response_data)

@app.post("/annotate_people")
def annotate_people(
    threshold: float = 0.5,
    annotate:bool = True,
    file: UploadFile = File(...),
    detector: GunDetector = Depends(get_gun_detector),
)-> Response:
    detection, img = detectPeople_uploadfile(detector, file, threshold)
    annotated_img = annotate_segmentation(img, detection,annotate)

    img_pil = Image.fromarray(annotated_img)
    image_stream = io.BytesIO()
    img_pil.save(image_stream, format="JPEG")
    image_stream.seek(0)
    return Response(content=image_stream.read(), media_type="image/jpeg")

@app.post("/detect")
def detection(
    threshold: float = 0.5,
    file: UploadFile = File(...),
    detector: GunDetector = Depends(get_gun_detector),
) -> JSONResponse:
    # Read file content once
    img_stream = io.BytesIO(file.file.read())
    if file.content_type.split("/")[0] != "image":
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Not an image"
        )

    try:
        img_obj = Image.open(img_stream)
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Image format not supported"
        )

    # Convert image to numpy array
    img_array = np.array(img_obj)

    # Run detections
    detection_result = detector.detect_guns(img_array, threshold)
    segmentation_result = detector.segment_people(img_array, threshold)

    detection={
        "n_detections": detection_result.n_detections,
        "boxes":detection_result.boxes,
        "labels":detection_result.labels,
        "confidences":detection_result.confidences
    }
    # Structure the response
    segmentation = {
        "n_detections": segmentation_result.n_detections,
        "polygons": segmentation_result.polygons,
        "boxes": segmentation_result.boxes,
        "labels": segmentation_result.labels
    }

    response_data = {
        "GunDetection": detection,
        "PeopleSegmentation": segmentation
    }
    return JSONResponse(content=response_data)

@app.post("/annotate")
def annotate(
    threshold: float = 0.5,
    annotate:bool = True,
    file: UploadFile = File(...),
    detector: GunDetector = Depends(get_gun_detector),
)-> Response:
    
    img_stream = io.BytesIO(file.file.read())
    if file.content_type.split("/")[0] != "image":
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Not an image"
        )

    try:
        img_obj = Image.open(img_stream)
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Image format not supported"
        )

    # Convert image to numpy array
    img = np.array(img_obj)

    # Run detections
    detection_result = detector.detect_guns(img, threshold)
    segmentation_result = detector.segment_people(img, threshold)

    annotated_img = annotate_segmentation(img, segmentation_result,annotate)

    annotated_img = annotate_detection(annotated_img, detection_result)

    img_pil = Image.fromarray(annotated_img)
    image_stream = io.BytesIO()
    img_pil.save(image_stream, format="JPEG")
    image_stream.seek(0)
    return Response(content=image_stream.read(), media_type="image/jpeg")

@app.post("/guns")
def guns(
    threshold: float = 0.5,
    file: UploadFile = File(...),
    detector: GunDetector = Depends(get_gun_detector),
) -> list[Gun]:

    guns=[]
    results, _ = detect_uploadfile(detector, file, threshold)
    for i in range (results.n_detections):
        print('pos',i)
        print('result boxes: ',results.boxes[i])
        print('result labels: ',results.labels[i])
        x1, y1, x2, y2 =results.boxes[i]
        pxl=PixelLocation(
            x=x2-x1,
            y=y2-y1
        )
        gun=Gun(
            gun_type=results.labels[i],
            location=pxl
        )
        guns.append(gun)
    return guns

@app.post("/people")
def people(
    threshold: float = 0.5,
    file: UploadFile = File(...),
    detector: GunDetector = Depends(get_gun_detector),
) -> list[Person]:

    people=[]
    results, _ = detectPeople_uploadfile(detector, file, threshold)
    for i in range (results.n_detections):
        print('pos',i)
        print('result boxes: ',results.boxes[i])
        print('result labels: ',results.labels[i])
        print('result points: ',results.polygons[i])
        area=polygon_area(results.polygons[i])
        print('area: ',area)
        x1, y1, x2, y2 =results.boxes[i]

        pxl=PixelLocation(
            x=x2-x1,
            y=y2-y1
        )
        person=Person(
            person_type=results.labels[i],
            location=pxl,
            area=int(area)
        )
        people.append(person)
    return people

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.app:app", port=8080, host="0.0.0.0")
