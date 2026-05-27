import os
import shutil
import random
from pathlib import Path

CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic']
SPLITS  = {'train': 0.70, 'val': 0.15, 'test': 0.15}


def prepare_dataset(raw_path='dataset/raw', out_path='dataset'):
    random.seed(42)

    for split in SPLITS:
        for cls in CLASSES:
            os.makedirs(f'{out_path}/{split}/{cls}', exist_ok=True)

    for cls in CLASSES:
        src_dir = Path(raw_path) / cls
        if not src_dir.exists():
            print(f"⚠  Carpeta no encontrada: {src_dir} — saltando.")
            continue

        images = list(src_dir.glob('*.jpg')) + list(src_dir.glob('*.png')) + list(src_dir.glob('*.jpeg'))
        random.shuffle(images)

        n       = len(images)
        n_train = int(n * SPLITS['train'])
        n_val   = int(n * SPLITS['val'])

        split_map = (
            [('train', img) for img in images[:n_train]] +
            [('val',   img) for img in images[n_train:n_train + n_val]] +
            [('test',  img) for img in images[n_train + n_val:]]
        )

        for split, src in split_map:
            dst = Path(out_path) / split / cls / src.name
            shutil.copy2(src, dst)

        n_test = n - n_train - n_val
        print(f"  {cls:<12}: {n_train:>3} train | {n_val:>3} val | {n_test:>3} test  (total: {n})")

    print('\n✓ Dataset listo en:', out_path)


if __name__ == '__main__':
    print("=== Preparando TrashNet (5 clases) ===\n")
    prepare_dataset()