from shapely.geometry import Polygon

# Puntos del polígono (como en tu JSON)
points = [
    (280, 486),
    (282, 637),
    (422, 688),
    (430, 795),
    (1033, 795),
    (1135, 643),
    (1135, 556),
    (895, 383),
    (879, 198),
    (773, 107),
    (568, 180),
    (673, 424),
    (404, 521)
]

# Crear el objeto Polygon a partir de los puntos
polygon = Polygon(points)

# Calcular el área
area = polygon.area
print(f"El área del polígono es: {area} unidades cuadradas")

