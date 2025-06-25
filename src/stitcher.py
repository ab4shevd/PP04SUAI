import os
import re
import cv2
import numpy as np
from .geo_utils import latlon_to_mercator, calculate_transform_params


def parse_filename(filename):
    """
    Извлекает метаданные из имени файла
    Формат: area_{frame}_{lat1}_{lon1}_{lat2}_{lon2}_{lat3}_{lon3}_{lat4}_{lon4}_{azimuth}.png
    """
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


class MapStitcher:
    def __init__(self, input_dir, output_path, resolution=0.1):
        self.input_dir = input_dir
        self.output_path = output_path
        self.resolution = resolution  # пикселей/метр
        self.fragments = []

    def load_fragments(self):
        """Загружает и парсит все фрагменты"""
        for fname in os.listdir(self.input_dir):
            if fname.endswith('.png'):
                data = parse_filename(fname)
                if data:
                    self.fragments.append((fname, *data))

        # Сортируем по номеру кадра
        self.fragments.sort(key=lambda x: x[1])

    def create_canvas(self, transform_params):
        """Создает пустое изображение-холст"""
        width_m = transform_params['max_x'] - transform_params['min_x']
        height_m = transform_params['max_y'] - transform_params['min_y']

        width_px = int(width_m * self.resolution) + 1000  # + буфер
        height_px = int(height_m * self.resolution) + 1000

        return np.zeros((height_px, width_px, 4), dtype=np.uint8)

    def _warp_fragment(self, img, corners, transform_params):
        """Трансформирует фрагмент для наложения"""
        merc_corners = []
        for lat, lon in corners:
            x, y = latlon_to_mercator(lat, lon, transform_params['ref_lat'])
            px = int((x - transform_params['min_x']) * self.resolution)
            py = int((transform_params['max_y'] - y) * self.resolution)
            merc_corners.append([px, py])

        src_points = np.array([
            [0, 0],
            [img.shape[1], 0],
            [img.shape[1], img.shape[0]],
            [0, img.shape[0]]
        ], dtype=np.float32)

        dst_points = np.array(merc_corners, dtype=np.float32)

        M = cv2.getPerspectiveTransform(src_points, dst_points)

        warped = cv2.warpPerspective(
            img, M, (self.canvas.shape[1], self.canvas.shape[0]),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_TRANSPARENT
        )

        return warped

    def _apply_rotation(self, img, azimuth):
        """Применяет поворот по азимуту"""
        if abs(azimuth) < 0.1:
            return img

        center = (img.shape[1] // 2, img.shape[0] // 2)
        M = cv2.getRotationMatrix2D(center, azimuth, 1.0)
        return cv2.warpAffine(
            img, M, (img.shape[1], img.shape[0]),
            borderMode=cv2.BORDER_TRANSPARENT
        )

    def _blend_images(self, canvas, fragment):
        """Смешивает изображения с учетом прозрачности"""
        alpha_f = fragment[:, :, 3] / 255.0
        alpha_c = canvas[:, :, 3] / 255.0

        combined_alpha = alpha_f + alpha_c * (1 - alpha_f)

        for c in range(3):
            canvas[:, :, c] = (fragment[:, :, c] * alpha_f) + \
                              (canvas[:, :, c] * alpha_c * (1 - alpha_f))

        canvas[:, :, 3] = combined_alpha * 255
        return canvas

    def stitch(self, progress_callback=None):
        """Основной процесс склейки"""
        self.load_fragments()
        if not self.fragments:
            print("Фрагменты не найдены!")
            return False

        print(f"Найдено {len(self.fragments)} фрагментов")

        tp = calculate_transform_params(self.fragments)
        self.canvas = self.create_canvas(tp)

        total = len(self.fragments)
        for idx, (fname, _, corners, azimuth) in enumerate(self.fragments):
            img_path = os.path.join(self.input_dir, fname)
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

            if img is None:
                print(f"Ошибка чтения: {img_path}")
                continue

            if img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

            img = self._apply_rotation(img, azimuth)
            warped = self._warp_fragment(img, corners, tp)
            self.canvas = self._blend_images(self.canvas, warped)

            if progress_callback:
                progress_callback(idx + 1, total)

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        cv2.imwrite(self.output_path, self.canvas)
        return True