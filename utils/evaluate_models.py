import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE   = 128
BATCH_SIZE = 32
CLASSES    = ['cardboard', 'glass', 'metal', 'paper', 'plastic']
CLASS_EMOJIS = {
    'cardboard': '📦', 'glass': '🍶', 'metal': '🥫',
    'paper': '📄',    'plastic': '🧴',
}
COLORS = {'scratch': '#2196F3', 'transfer': '#4CAF50'}


def load_model_safe(path):
    try:
        model = tf.keras.models.load_model(path)
        print(f"✓ Loaded: {path}")
        return model
    except Exception as e:
        print(f"✗ Could not load {path}: {e}")
        return None


def get_test_generator(dataset_path='dataset'):
    datagen = ImageDataGenerator(rescale=1./255)
    return datagen.flow_from_directory(
        os.path.join(dataset_path, 'test'),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )


def evaluate_model(model, test_gen):
    test_gen.reset()
    y_pred_proba = model.predict(test_gen, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = test_gen.classes
    return y_true, y_pred, y_pred_proba


def plot_confusion_matrix(y_true, y_pred, title, ax, color):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt='.2f', ax=ax,
                xticklabels=CLASSES, yticklabels=CLASSES,
                cmap=sns.light_palette(color, as_cmap=True),
                linewidths=0.5, cbar_kws={'shrink': 0.8})
    ax.set_title(title, fontweight='bold', pad=10)
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.tick_params(axis='x', rotation=30)


def plot_roc_curves(y_true, y_proba, ax, model_name, color):
    n_classes = len(CLASSES)
    y_true_bin = np.eye(n_classes)[y_true]
    for i, cls in enumerate(CLASSES):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f'{cls} (AUC={roc_auc:.2f})', linewidth=1.5)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curves – {model_name}', fontweight='bold')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)


def generate_comparison_report(results, save_path='models'):
    os.makedirs(save_path, exist_ok=True)

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle('Waste Classifier – Model Comparison Report\n(cardboard | glass | metal | paper | plastic)',
                 fontsize=16, fontweight='bold', y=0.98)

    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.5, wspace=0.4)

    ax_cm1 = fig.add_subplot(gs[0, :2])
    ax_cm2 = fig.add_subplot(gs[0, 2:])
    if 'scratch' in results:
        plot_confusion_matrix(results['scratch']['y_true'], results['scratch']['y_pred'],
                              'Confusion Matrix – CNN Scratch', ax_cm1, COLORS['scratch'])
    if 'transfer' in results:
        plot_confusion_matrix(results['transfer']['y_true'], results['transfer']['y_pred'],
                              'Confusion Matrix – MobileNetV2 Transfer', ax_cm2, COLORS['transfer'])

    ax_roc1 = fig.add_subplot(gs[1, :2])
    ax_roc2 = fig.add_subplot(gs[1, 2:])
    if 'scratch' in results:
        plot_roc_curves(results['scratch']['y_true'], results['scratch']['y_proba'],
                        ax_roc1, 'CNN Scratch', COLORS['scratch'])
    if 'transfer' in results:
        plot_roc_curves(results['transfer']['y_true'], results['transfer']['y_proba'],
                        ax_roc2, 'MobileNetV2 Transfer', COLORS['transfer'])

    ax_bars = fig.add_subplot(gs[2, :3])
    ax_sum  = fig.add_subplot(gs[2, 3])

    x = np.arange(len(CLASSES))
    width = 0.35
    model_names, model_accs = [], []

    for i, (name, color) in enumerate(COLORS.items()):
        if name not in results:
            continue
        report = classification_report(
            results[name]['y_true'], results[name]['y_pred'],
            target_names=CLASSES, output_dict=True
        )
        per_class = [report[c]['f1-score'] for c in CLASSES]
        ax_bars.bar(x + i*width, per_class, width, label=name.capitalize(),
                    color=color, alpha=0.85, edgecolor='white')
        model_names.append(name)
        model_accs.append(results[name].get('test_acc', 0))

    ax_bars.set_xticks(x + width/2)
    ax_bars.set_xticklabels([f"{CLASS_EMOJIS[c]} {c}" for c in CLASSES])
    ax_bars.set_ylabel('F1-Score')
    ax_bars.set_title('Per-Class F1-Score Comparison', fontweight='bold')
    ax_bars.legend(); ax_bars.set_ylim(0, 1.1)
    ax_bars.grid(True, axis='y', alpha=0.3)

    colors_list = [COLORS[n] for n in model_names]
    bars = ax_sum.bar(model_names, model_accs, color=colors_list,
                      alpha=0.85, edgecolor='white', width=0.5)
    for bar, acc in zip(bars, model_accs):
        ax_sum.text(bar.get_x() + bar.get_width()/2., bar.get_height()+0.01,
                    f'{acc:.1%}', ha='center', va='bottom', fontweight='bold')
    ax_sum.set_ylabel('Test Accuracy')
    ax_sum.set_title('Overall Accuracy', fontweight='bold')
    ax_sum.set_ylim(0, 1.15)
    ax_sum.grid(True, axis='y', alpha=0.3)

    plt.savefig(os.path.join(save_path, 'model_comparison.png'),
                dpi=120, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n✓ Comparison report → {save_path}/model_comparison.png")

    # Text report
    print("\n" + "="*60)
    print("         WASTE CLASSIFIER – MODEL COMPARISON")
    print("="*60)
    for name in ['scratch', 'transfer']:
        if name not in results:
            continue
        acc   = results[name].get('test_acc', 0)
        label = 'CNN from Scratch' if name == 'scratch' else 'MobileNetV2 Transfer'
        print(f"\n📊 {label}  |  Test Accuracy: {acc:.4f} ({acc:.1%})")
        print(classification_report(
            results[name]['y_true'], results[name]['y_pred'], target_names=CLASSES
        ))

    if 'scratch' in results and 'transfer' in results:
        s = results['scratch']['test_acc']
        t = results['transfer']['test_acc']
        winner = 'Transfer Learning (MobileNetV2)' if t > s else 'CNN from Scratch'
        diff   = abs(t - s)
        print("="*60)
        print(f"🏆  WINNER: {winner}  (+{diff:.1%})")
        print("="*60)


def run_comparison(dataset_path='dataset', models_path='models'):
    results  = {}
    test_gen = get_test_generator(dataset_path)

    for name, fname in [('scratch', 'scratch_best.keras'), ('transfer', 'transfer_best.keras')]:
        path = os.path.join(models_path, fname)
        if not os.path.exists(path):
            print(f"Model not found: {path} – skipping.")
            continue
        model = load_model_safe(path)
        if model is None:
            continue
        test_gen.reset()
        y_true, y_pred, y_proba = evaluate_model(model, test_gen)
        _, test_acc = model.evaluate(test_gen, verbose=0)
        results[name] = {'y_true': y_true, 'y_pred': y_pred,
                         'y_proba': y_proba, 'test_acc': test_acc}

    if results:
        generate_comparison_report(results, models_path)
    else:
        print("No models found. Run training first.")
    return results


if __name__ == '__main__':
    run_comparison()