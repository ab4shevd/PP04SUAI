import numpy as np

EARTH_RADIUS = 6371000  # метров


def latlon_to_mercator(lat, lon, ref_lat):
    """
    Конвертирует WGS84 в локальную систему координат (метры)
    """
    x = np.radians(lon) * EARTH_RADIUS * np.cos(np.radians(ref_lat))
    y = np.radians(lat) * EARTH_RADIUS
    return x, y


def calculate_transform_params(fragments):
    """
    Рассчитывает общие параметры трансформации для всех фрагментов
    """
    all_coords = np.vstack([corners for _, _, corners, _ in fragments])
    mean_lat = np.mean(all_coords[:, 0])

    # Находим границы холста
    x_coords, y_coords = [], []
    for _, _, corners, _ in fragments:
        for lat, lon in corners:
            x, y = latlon_to_mercator(lat, lon, mean_lat)
            x_coords.append(x)
            y_coords.append(y)

    return {
        'min_x': min(x_coords),
        'max_x': max(x_coords),
        'min_y': min(y_coords),
        'max_y': max(y_coords),
        'ref_lat': mean_lat
    }