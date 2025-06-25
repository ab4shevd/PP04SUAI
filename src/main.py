import argparse
import os
from .stitcher import MapStitcher
from .progress import ProgressBar


def main():
    parser = argparse.ArgumentParser(
        description='Склейка географических фрагментов карт'
    )
    parser.add_argument(
        'input_dir',
        help='Директория с фрагментами карт'
    )
    parser.add_argument(
        'output',
        help='Путь для сохранения результата (PNG)'
    )
    parser.add_argument(
        '--res',
        type=float,
        default=0.1,
        help='Разрешение в пикселях на метр (по умолчанию: 0.1)'
    )

    args = parser.parse_args()

    # Проверка существования директории
    if not os.path.isdir(args.input_dir):
        print(f"Ошибка: Директория '{args.input_dir}' не существует!")
        return

    print(f"Начало обработки...")
    print(f"Источник: {args.input_dir}")
    print(f"Результат: {args.output}")
    print(f"Разрешение: {args.res} px/м")

    # Инициализация прогресс-бара
    progress = ProgressBar(0)  # временно 0, обновим позже

    def update_progress(current, total):
        if current == 1:
            progress.pbar.total = total
        progress.update()

    stitcher = MapStitcher(
        input_dir=args.input_dir,
        output_path=args.output,
        resolution=args.res
    )

    success = stitcher.stitch(progress_callback=update_progress)

    if success:
        print("\nСклейка успешно завершена!")
        print(f"Результат сохранен в: {args.output}")
    else:
        print("\nОшибка при выполнении склейки")

    progress.close()


if __name__ == "__main__":
    main()