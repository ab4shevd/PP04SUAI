#!/usr/bin/env python3
"""
Упрощенная склейка географических фрагментов карт
"""

import os
import re
import cv2
import numpy as np
import argparse
from tqdm import tqdm

# Константы
EARTH_RADIUS = 6371000  # метров


def parse_filename(filename):
    """Извлекает метаданные из имени файла"""
    pattern = r"area_(\d+)_([\d.]+)_([\d.]+)_([\d.]+)_([\d.]+)_([\d.]+)_([\d.]+)_([\d.]+)_([\d.]+)_([-\d.]+)\.png"
    match = re.match(pattern, filename)

    if not match:
        return None

    frame_id = int(match.group(1))
    azimuth = float(match.group(11))

    corners = np.array([
        [float(match.group(2)), float(match.group(3))],
        [float(match.group(4)), float(match.group(5))],
        [float(match.group(6)), float(match.group(7))],
        [float(match.group(8)), float(match.group(9))]
    ])

    return frame_id, corners, azimuth


def latlon_to_mercator(lat, lon, ref_lat):
    """Конвертирует координаты в метры"""
    x = np.radians(lon) * EARTH_RADIUS * np.cos(np.radians(ref_lat))
    y = np.radians(lat) * EARTH_RADIUS
    return x, y


def calculate_transform_params(fragments):
    """Рассчитывает параметры трансформации"""
    all_coords = np.vstack([corners for _, corners, _ in fragments])
    mean_lat = np.mean(all_coords[:, 0])

    x_coords, y_coords = [], []
    for _, corners, _ in fragments:
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


def main():
    """Основная функция"""
    parser = argparse.ArgumentParser(
        description='Склейка географических фрагментов карт',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Параметры с значениями по умолчанию
    parser.add_argument('-i', '--input', default='data/fragments',
                        help='Папка с фрагментами карт')
    parser.add_argument('-o', '--output', default='data/output/map.png',
                        help='Путь для сохранения результата')
    parser.add_argument('-r', '--resolution', type=float, default=0.1,
                        help='Разрешение в пикселях на метр')

    args = parser.parse_args()

    print("""
    #############################################
    #      Программа склейки фрагментов карт     #
    #        (упрощенная версия без плагинов)    #
    #############################################
    """)

    print(f"• Папка с фрагментами: {args.input}")
    print(f"• Результат: {args.output}")
    print(f"• Разрешение: {args.resolution} пикселей/метр")

    # Сбор фрагментов
    fragments = []
    print("\n🔍 Поиск фрагментов карт...")
    for fname in os.listdir(args.input):
        if fname.endswith('.png'):
            data = parse_filename(fname)
            if data:
                fragments.append((fname, *data))

    if not fragments:
        print("❌ Ошибка: Фрагменты не найдены!")
        return

    # Сортировка по номеру кадра
    fragments.sort(key=lambda x: x[1])
    print(f"• Найдено фрагментов: {len(fragments)}")

    # Расчет параметров трансформации
    print("🧮 Расчет параметров трансформации...")
    tp = calculate_transform_params(fragments)

    # Создание холста
    width_m = tp['max_x'] - tp['min_x']
    height_m = tp['max_y'] - tp['min_y']
    width_px = int(width_m * args.resolution) + 1000
    height_px = int(height_m * args.resolution) + 1000
    canvas = np.zeros((height_px, width_px, 4), dtype=np.uint8)

    print(f"• Размер холста: {width_px} x {height_px} пикселей")
    print("\n🚀 Начало склейки...")

    # Обработка фрагментов
    for fname, _, corners, azimuth in tqdm(fragments, desc="Склейка"):
        img_path = os.path.join(args.input, fname)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        if img is None:
            continue

        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

        # Поворот по азимуту
        if abs(azimuth) > 0.1:
            center = (img.shape[1] // 2, img.shape[0] // 2)
            M = cv2.getRotationMatrix2D(center, azimuth, 1.0)
            img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                                 borderMode=cv2.BORDER_TRANSPARENT)

        # Трансформация
        merc_corners = []
        for lat, lon in corners:
            x, y = latlon_to_mercator(lat, lon, tp['ref_lat'])
            px = int((x - tp['min_x']) * args.resolution)
            py = int((tp['max_y'] - y) * args.resolution)
            merc_corners.append([px, py])

        src_points = np.array([[0, 0], [img.shape[1], 0],
                               [img.shape[1], img.shape[0]],
                               [0, img.shape[0]]], dtype=np.float32)

        dst_points = np.array(merc_corners, dtype=np.float32)
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        warped = cv2.warpPerspective(img, M, (canvas.shape[1], canvas.shape[0]),
                                     flags=cv2.INTER_LANCZOS4,
                                     borderMode=cv2.BORDER_TRANSPARENT)

        # Смешивание
        alpha_f = warped[:, :, 3] / 255.0
        alpha_c = canvas[:, :, 3] / 255.0
        combined_alpha = alpha_f + alpha_c * (1 - alpha_f)

        for c in range(3):
            canvas[:, :, c] = (warped[:, :, c] * alpha_f) + \
                              (canvas[:, :, c] * alpha_c * (1 - alpha_f))
        canvas[:, :, 3] = combined_alpha * 255

    # Сохранение результата
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    cv2.imwrite(args.output, canvas)

    print(f"\n✅ Готово! Результат сохранен в: {args.output}")
    print(f"Размер изображения: {canvas.shape[1]}x{canvas.shape[0]} пикселей")


if __name__ == "__main__":
    main()