import argparse
import os
import sys
import json
import time

def banner(text):
    w = 60
    print("\n" + "═"*w)
    print(f"  {text}")
    print("═"*w)



def step_scratch():
    banner("STEP 2 ─ Training CNN from Scratch")
    sys.path.insert(0, 'models')
    from model_scratch import train_scratch_model
    model, history, acc = train_scratch_model(
        dataset_path='dataset',
        save_path='models'
    )
    print(f"\n✓ Scratch model trained  |  Test acc: {acc:.4f}")
    return acc


def step_transfer():
    banner("STEP 3 ─ Training Transfer Learning (MobileNetV2)")
    sys.path.insert(0, 'models')
    from model_transfer import train_transfer_model
    model, history, acc = train_transfer_model(
        dataset_path='dataset',
        save_path='models'
    )
    print(f"\n✓ Transfer model trained  |  Test acc: {acc:.4f}")
    return acc


def step_compare():
    banner("STEP 4 ─ Comparing Models")
    sys.path.insert(0, 'utils')
    from evaluate_models import run_comparison
    results = run_comparison(dataset_path='dataset', models_path='models')
    return results


def main():
    parser = argparse.ArgumentParser(description='Waste Classifier Training Pipeline')

    parser.add_argument('--model', choices=['scratch', 'transfer', 'both'],
                        default='both', help='Which model to train')

    parser.add_argument('--compare-only', action='store_true',
                        help='Only run model comparison (models must exist)')

    args = parser.parse_args()

    t_start = time.time()
    results = {}

    if args.compare_only:
        step_compare()
        return

    print("✓ Using existing TrashNet dataset.")

    if args.model in ('scratch', 'both'):
        results['scratch'] = step_scratch()

    if args.model in ('transfer', 'both'):
        results['transfer'] = step_transfer()

    if len(results) > 1:
        step_compare()

    elapsed = time.time() - t_start

    banner("TRAINING COMPLETE")

    print(f"  Total time: {elapsed/60:.1f} minutes")

    if results:
        for name, acc in results.items():
            print(f"  {name:<12}: {acc:.1%} test accuracy")

    print("\n  ▶  Run the app:  python app/app.py")
    print("═"*60)

if __name__ == '__main__':
    main()