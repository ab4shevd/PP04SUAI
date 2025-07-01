import os
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog, messagebox
from tqdm import tqdm

EARTH_RADIUS = 6378137  # Радиус Земли (WGS-84)


class MapStitcher:
    def __init__(self):
        self.fragments = []
        self.min_lat = None
        self.min_lon = None
        self.max_lat = None
        self.max_lon = None

    def parse_filename(self, filename):
        """Парсит имя файла и возвращает координаты углов и путь"""
        pattern = r"area_(\d+)_([\d.-]+)_([\d.-]+)_([\d.-]+)_([\d.-]+)_([\d.-]+)_([\d.-]+)_([\d.-]+)_([\d.-]+)_([\d.-]+)\.png"
        match = re.match(pattern, filename)
        if not match:
            return None

        # Извлекаем координаты 4 углов (8 значений)
        coords = [float(match.group(i)) for i in range(2, 10)]
        corners = np.array([
            [coords[0], coords[1]],  # top-left
            [coords[2], coords[3]],  # top-right
            [coords[4], coords[5]],  # bottom-right
            [coords[6], coords[7]]   # bottom-left
        ], dtype=np.float64)
        return corners, filename

    def latlon_to_mercator(self, lat, lon):
        """Перевод координат в проекцию Меркатора"""
        x = np.radians(lon) * EARTH_RADIUS
        y = np.log(np.tan(np.pi / 4 + np.radians(lat) / 2)) * EARTH_RADIUS
        return x, y

    def load_fragments(self, input_dir):
        """Загрузка и парсинг фрагментов карты"""
        self.fragments = []
        all_lats = []
        all_lons = []

        for fname in os.listdir(input_dir):
            if not fname.lower().endswith('.png'):
                continue

            result = self.parse_filename(fname)
            if result is None:
                print(f"Неверный формат имени файла: {fname}")
                continue

            corners, name = result
            full_path = os.path.join(input_dir, name)
            self.fragments.append((corners, full_path))

            # Собираем все координаты для определения границ
            all_lats.extend(corners[:, 0])
            all_lons.extend(corners[:, 1])

        if not self.fragments:
            raise ValueError("Не найдено подходящих файлов с фрагментами.")

        # Определяем границы всех фрагментов
        self.min_lat = np.min(all_lats)
        self.max_lat = np.max(all_lats)
        self.min_lon = np.min(all_lons)
        self.max_lon = np.max(all_lons)

        print(f"Границы координат:")
        print(f"Широта: {self.min_lat:.6f} - {self.max_lat:.6f}")
        print(f"Долгота: {self.min_lon:.6f} - {self.max_lon:.6f}")

    def calculate_bounds(self):
        """Расчет границ холста в метрах (Меркатор)"""
        min_x, min_y = self.latlon_to_mercator(self.min_lat, self.min_lon)
        max_x, max_y = self.latlon_to_mercator(self.max_lat, self.max_lon)

        # Корректируем для северного полушария
        if min_y > max_y:
            min_y, max_y = max_y, min_y

        return min_x, max_x, min_y, max_y

    def stitch(self, input_dir, pixels_per_meter=0.1):
        """Основной процесс склейки фрагментов"""
        self.load_fragments(input_dir)
        min_x, max_x, min_y, max_y = self.calculate_bounds()

        width = int((max_x - min_x) * pixels_per_meter)
        height = int((max_y - min_y) * pixels_per_meter)
        print(f"Размер холста: {width}x{height} пикселей")

        # Создаем холст с альфа-каналом для смешивания
        canvas = np.zeros((height, width, 4), dtype=np.float32)

        fragment_info = []
        for corners, img_path in self.fragments:
            # Переводим географические координаты в пиксельные
            pixel_pts = []
            for lat, lon in corners:
                x, y = self.latlon_to_mercator(lat, lon)
                px = int((x - min_x) * pixels_per_meter)
                py = int((y - min_y) * pixels_per_meter)
                pixel_pts.append([px, py])
            fragment_info.append((img_path, np.array(pixel_pts, dtype=np.float32)))

        # Расчёт перекрытий между фрагментами
        overlaps = {}
        for i, (_, pts_i) in enumerate(fragment_info):
            overlap_count = 0
            for j, (_, pts_j) in enumerate(fragment_info):
                if i != j:
                    hull_i = cv2.convexHull(pts_i)
                    hull_j = cv2.convexHull(pts_j)
                    intersection = cv2.intersectConvexConvex(hull_i, hull_j)[0]
                    if intersection:
                        overlap_count += 1
            overlaps[i] = overlap_count

        # Сортировка: сначала по перекрытиям, потом по площади
        sorted_indices = sorted(
            range(len(fragment_info)),
            key=lambda i: (-overlaps[i], -cv2.contourArea(fragment_info[i][1]))
        )
        sorted_fragment_info = [fragment_info[i] for i in sorted_indices]

        reference_fragment = None  # Первый фрагмент как эталон для цветокоррекции
        for img_path, dst_pts in tqdm(sorted_fragment_info, desc="Склейка фрагментов"):
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"Ошибка загрузки: {img_path}")
                continue

            # Добавляем альфа-канал, если его нет
            if img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

            h, w = img.shape[:2]

            # Правильный порядок точек для преобразования
            src_pts = np.array([
                [0, 0],
                [w - 1, 0],
                [w - 1, h - 1],
                [0, h - 1]
            ], dtype=np.float32)

            # Перспективное преобразование
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped = cv2.warpPerspective(
                img, M, (width, height),
                flags=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_TRANSPARENT
            )

            # Маска для текущего фрагмента
            mask = (warped[..., 3] > 0).astype(np.float32)

            # Простое альфа-смешивание
            for c in range(4):  # Смешиваем все каналы (RGB + Альфа)
                canvas[..., c] = canvas[..., c] * (1 - mask) + warped[..., c] * mask

        # Конвертируем в uint8 и удаляем альфа-канал
        result = np.clip(canvas, 0, 255).astype(np.uint8)
        result = cv2.cvtColor(result, cv2.COLOR_BGRA2BGR)
        return result



    def match_histograms(self, src, ref):
        """Выравнивание гистограмм src под ref"""
        matched = src.copy()
        for c in range(3):  # Применяем к каждому каналу RGB
            # Выравниваем гистограмму канала src под ref
            matched[:, :, c] = cv2.normalize(cv2.equalizeHist(matched[:, :, c]), None, 0, 255, cv2.NORM_MINMAX)
        return matched


def show_image(image):
    """Отображение результата"""
    plt.figure(figsize=(12, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Результат склейки карты")
    plt.tight_layout()
    plt.show()


def main():
    print("=== Программа склейки спутниковых снимков ===")
    root = Tk()
    root.withdraw()

    folder = filedialog.askdirectory(title="Выберите папку с фрагментами карты")
    if not folder:
        print("Папка не выбрана")
        return

    try:
        stitcher = MapStitcher()
        result = stitcher.stitch(folder, pixels_per_meter=0.2)

        show_image(result)

        if messagebox.askyesno("Сохранение", "Сохранить результат?"):
            output = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
            )
            if output:
                cv2.imwrite(output, result)
                print(f"Результат сохранен: {output}")

    except Exception as e:
        messagebox.showerror("Ошибка", str(e))
        print(f"Ошибка: {e}")


if __name__ == "__main__":
    main()