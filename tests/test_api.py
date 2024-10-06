import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image
import numpy as np
from src.main import app

client = TestClient(app)

# Helper function to create an image in memory
def create_test_image():
    image_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    image_pil = Image.fromarray(image_array)
    img_byte_arr = io.BytesIO()
    image_pil.save(img_byte_arr, format="JPEG")
    img_byte_arr.seek(0)
    return img_byte_arr

# Test model info endpoint
def test_model_info():
    response = client.get("/model_info")
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert "input_type" in data

# Test /detect_guns endpoint
def test_detect_guns():
    img_byte_arr = create_test_image()
    files = {'file': ("test_image.jpg", img_byte_arr, "image/jpeg")}
    response = client.post("/detect_guns", files=files, data={'threshold': '0.5'})
    assert response.status_code == 200
    data = response.json()
    assert "n_detections" in data
    assert "boxes" in data
    assert "labels" in data
    assert "confidences" in data

# Test /annotate_guns endpoint
def test_annotate_guns():
    img_byte_arr = create_test_image()
    files = {'file': ("test_image.jpg", img_byte_arr, "image/jpeg")}
    response = client.post("/annotate_guns", files=files, data={'threshold': '0.5'})
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"

# Test /detect_people endpoint
def test_detect_people():
    img_byte_arr = create_test_image()
    files = {'file': ("test_image.jpg", img_byte_arr, "image/jpeg")}
    response = client.post("/detect_people", files=files, data={'threshold': '0.5'})
    assert response.status_code == 200
    data = response.json()
    assert "n_detections" in data
    assert "polygons" in data
    assert "boxes" in data
    assert "labels" in data

# Test /annotate_people endpoint
def test_annotate_people():
    img_byte_arr = create_test_image()
    files = {'file': ("test_image.jpg", img_byte_arr, "image/jpeg")}
    response = client.post("/annotate_people", files=files, data={'threshold': '0.5', 'annotate': 'true'})
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"

# Test /detect endpoint
def test_detection():
    img_byte_arr = create_test_image()
    files = {'file': ("test_image.jpg", img_byte_arr, "image/jpeg")}
    response = client.post("/detect", files=files, data={'threshold': '0.5'})
    assert response.status_code == 200
    data = response.json()
    assert "GunDetection" in data
    assert "PeopleSegmentation" in data

# Test /annotate endpoint
def test_annotate():
    img_byte_arr = create_test_image()
    files = {'file': ("test_image.jpg", img_byte_arr, "image/jpeg")}
    response = client.post("/annotate", files=files, data={'threshold': '0.5', 'annotate': 'true'})
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"

# Test /guns endpoint
def test_guns():
    img_byte_arr = create_test_image()
    files = {'file': ("test_image.jpg", img_byte_arr, "image/jpeg")}
    response = client.post("/guns", files=files, data={'threshold': '0.5'})
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    if data:
        assert "gun_type" in data[0]
        assert "location" in data[0]

# Test /people endpoint
def test_people():
    img_byte_arr = create_test_image()
    files = {'file': ("test_image.jpg", img_byte_arr, "image/jpeg")}
    response = client.post("/people", files=files, data={'threshold': '0.5'})
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    if data:
        assert "person_type" in data[0]
        assert "location" in data[0]
        assert "area" in data[0]
