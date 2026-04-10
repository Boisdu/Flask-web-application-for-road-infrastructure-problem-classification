"""
Обучение модели для классификации 5 классов дорожных проблем:
- Broken Road Sign Issues (Поврежденные дорожные знаки)
- Damaged Road issues (Повреждения дорожного покрытия)
- Illegal Parking Issues (Нарушения парковки)
- Mixed Issues (Смешанные проблемы)
- Pothole Issues (Выбоины)
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from glob import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

# Для MacOS устанавливаем метод запуска multiprocessing
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

# Установка seed для воспроизводимости
torch.manual_seed(2025)
np.random.seed(2025)

# ==================== КОНФИГУРАЦИЯ ====================
IM_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
BATCH_SIZE = 8  # Для CPU
EPOCHS = 30
PATIENCE = 5
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
SAVE_DIR = "saved_models"
MODEL_NAME = "resnext101_32x8d"

# Словарь классов с русскими названиями для удобства
CLASS_NAMES_RU = {
    "Broken Road Sign Issues": "Поврежденные дорожные знаки",
    "Damaged Road issues": "Повреждения дорожного покрытия",
    "Illegal Parking Issues": "Нарушения парковки",
    "Mixed Issues": "Смешанные проблемы",
    "Pothole Issues": "Выбоины"
}

# Создаем папку для сохранения модели
os.makedirs(SAVE_DIR, exist_ok=True)


# ==================== КЛАСС ДЛЯ ЗАГРУЗКИ ДАННЫХ ====================
class CustomDataset(Dataset):
    """Кастомный датасет для загрузки изображений из папок"""

    def __init__(self, root, transformations=None, im_paths=None, im_lbls=None,
                 im_files=[".png", ".jpg", ".jpeg", ".bmp", ".JPG"]):

        self.transformations = transformations

        if im_paths and im_lbls:
            self.im_paths = im_paths
            self.im_lbls = im_lbls
        else:
            self.im_paths = []
            for im_file in im_files:
                for path in glob(f"{root}/*/*{im_file}"):
                    # Проверяем, что класс входит в наш список
                    class_name = self.get_class(path)
                    if class_name in CLASS_NAMES_RU:
                        self.im_paths.append(path)

        # Создаем словарь классов только для нужных категорий
        self.cls_names = {}
        self.cls_counts = {}
        count = 0
        for im_path in self.im_paths:
            class_name = self.get_class(im_path)
            if class_name not in self.cls_names:
                self.cls_names[class_name] = count
                self.cls_counts[class_name] = 1
                count += 1
            else:
                self.cls_counts[class_name] += 1

        print(f"Загружено {len(self.im_paths)} изображений")
        print(f"Классы: {list(self.cls_names.keys())}")
        for cls_name, count in self.cls_counts.items():
            print(f"  {cls_name} ({CLASS_NAMES_RU.get(cls_name, cls_name)}): {count}")

    def get_class(self, path):
        """Извлечение имени класса из пути к файлу"""
        return os.path.basename(os.path.dirname(path))

    def __len__(self):
        return len(self.im_paths)

    def __getitem__(self, idx):
        im_path = self.im_paths[idx]
        im = Image.open(im_path).convert("RGB")

        if hasattr(self, 'im_lbls'):
            gt = self.im_lbls[idx]
        else:
            gt = self.cls_names[self.get_class(im_path)]

        if self.transformations is not None:
            im = self.transformations(im)

        return im, gt

    @classmethod
    def stratified_split_dls(cls, root, transformations, bs=8, split=[0.8, 0.1, 0.1], ns=0):
        """Стратифицированное разделение данных на train/val/test"""
        dataset = cls(root=root, transformations=transformations)

        if len(dataset) == 0:
            raise ValueError("Нет изображений для загрузки! Проверьте путь к датасету.")

        im_paths = dataset.im_paths
        labels = [dataset.cls_names[dataset.get_class(p)] for p in im_paths]

        # Первое разделение: train и temp (val+test)
        train_paths, temp_paths, train_lbls, temp_lbls = train_test_split(
            im_paths, labels, test_size=(split[1] + split[2]),
            stratify=labels, random_state=2025
        )

        # Второе разделение: val и test
        val_ratio = split[1] / (split[1] + split[2])
        val_paths, test_paths, val_lbls, test_lbls = train_test_split(
            temp_paths, temp_lbls, test_size=(1 - val_ratio),
            stratify=temp_lbls, random_state=2025
        )

        # Создаем датасеты
        train_ds = cls(root=root, transformations=transformations,
                       im_paths=train_paths, im_lbls=train_lbls)
        val_ds = cls(root=root, transformations=transformations,
                     im_paths=val_paths, im_lbls=val_lbls)
        test_ds = cls(root=root, transformations=transformations,
                      im_paths=test_paths, im_lbls=test_lbls)

        # Создаем DataLoader'ы (num_workers=0 для MacOS)
        train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=ns)
        val_dl = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=ns)
        test_dl = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=ns)

        return train_dl, val_dl, test_dl, dataset.cls_names


# ==================== КЛАСС ДЛЯ ОБУЧЕНИЯ ====================
class ModelTrainer:
    """Класс для обучения и валидации модели"""

    def __init__(self, model, classes, train_dl, val_dl, device,
                 save_dir="saved_models", save_prefix="road",
                 lr=1e-4, epochs=50, patience=5, threshold=0.01):

        self.model = model
        self.classes = classes
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.device = device
        self.save_dir = save_dir
        self.save_prefix = save_prefix
        self.lr = lr
        self.epochs = epochs
        self.patience = patience
        self.threshold = threshold

        # Функция потерь и оптимизатор
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

        # Метрики
        self.best_accuracy = 0
        self.best_f1 = 0
        self.patience_counter = 0

        # Хранение истории обучения
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.train_f1s = []
        self.val_f1s = []

        # Создаем папку для сохранения
        os.makedirs(save_dir, exist_ok=True)

    def calculate_accuracy(self, outputs, targets):
        """Вычисление accuracy"""
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == targets).sum().item()
        return correct / targets.size(0)

    def calculate_f1(self, outputs, targets, num_classes):
        """Вычисление F1 score (macro)"""
        _, predicted = torch.max(outputs, 1)

        f1_scores = []
        for c in range(num_classes):
            tp = ((predicted == c) & (targets == c)).sum().item()
            fp = ((predicted == c) & (targets != c)).sum().item()
            fn = ((predicted != c) & (targets == c)).sum().item()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0

            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            f1_scores.append(f1)

        return np.mean(f1_scores)

    def train_epoch(self):
        """Обучение одной эпохи"""
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        all_outputs = []
        all_targets = []

        for batch_idx, (images, targets) in enumerate(self.train_dl):
            images = images.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Статистика
            total_loss += loss.item() * images.size(0)
            total_correct += (outputs.argmax(1) == targets).sum().item()
            total_samples += images.size(0)

            all_outputs.append(outputs)
            all_targets.append(targets)

            # Прогресс
            if (batch_idx + 1) % 20 == 0:
                print(f"  Batch {batch_idx + 1}/{len(self.train_dl)}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples

        # Вычисление F1
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        f1 = self.calculate_f1(all_outputs, all_targets, len(self.classes))

        return avg_loss, accuracy, f1

    def validate_epoch(self):
        """Валидация одной эпохи"""
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        all_outputs = []
        all_targets = []

        with torch.no_grad():
            for images, targets in self.val_dl:
                images = images.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

                total_loss += loss.item() * images.size(0)
                total_correct += (outputs.argmax(1) == targets).sum().item()
                total_samples += images.size(0)

                all_outputs.append(outputs)
                all_targets.append(targets)

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples

        # Вычисление F1
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        f1 = self.calculate_f1(all_outputs, all_targets, len(self.classes))

        return avg_loss, accuracy, f1

    def save_model(self, filename):
        """Сохранение модели"""
        save_path = os.path.join(self.save_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'classes': self.classes,
            'classes_ru': CLASS_NAMES_RU,
            'model_name': MODEL_NAME,
            'im_size': IM_SIZE,
            'mean': MEAN,
            'std': STD
        }, save_path)
        print(f"✅ Модель сохранена: {save_path}")

    def train(self):
        """Основной цикл обучения"""
        print("=" * 60)
        print("🚀 НАЧАЛО ОБУЧЕНИЯ МОДЕЛИ")
        print("=" * 60)
        print(f"Устройство: {self.device}")
        print(f"Количество классов: {len(self.classes)}")
        print("\nКлассы:")
        for cls_name, cls_id in self.classes.items():
            print(f"  {cls_id}: {cls_name} ({CLASS_NAMES_RU.get(cls_name, cls_name)})")
        print(f"\nРазмер батча: {self.train_dl.batch_size}")
        print(f"Количество эпох: {self.epochs}")
        print(f"Скорость обучения: {self.lr}")
        print("=" * 60)

        for epoch in range(self.epochs):
            print(f"\n{'='*60}")
            print(f"ЭПОХА {epoch + 1}/{self.epochs}")
            print(f"{'='*60}")

            # Обучение
            print("📚 Обучение...")
            train_loss, train_acc, train_f1 = self.train_epoch()
            print(f"  Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}, F1: {train_f1:.4f}")

            # Валидация
            print("🔍 Валидация...")
            val_loss, val_acc, val_f1 = self.validate_epoch()
            print(f"  Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}, F1: {val_f1:.4f}")

            # Сохранение истории
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)
            self.train_f1s.append(train_f1)
            self.val_f1s.append(val_f1)

            # Сохранение лучшей модели
            if val_f1 > self.best_f1 + self.threshold:
                self.best_f1 = val_f1
                self.save_model(f"{self.save_prefix}_best_model.pth")
                self.patience_counter = 0
                print(f"🎉 Новая лучшая модель! F1: {self.best_f1:.4f}")
            else:
                self.patience_counter += 1
                print(f"Нет улучшения. Счетчик patience: {self.patience_counter}/{self.patience}")

            # Early stopping
            if self.patience_counter >= self.patience:
                print(f"\n⏹️ Early stopping на эпохе {epoch + 1}")
                break

        print("\n" + "=" * 60)
        print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО")
        print(f"Лучший F1: {self.best_f1:.4f}")
        print("=" * 60)

        return self.best_f1


# ==================== ВИЗУАЛИЗАЦИЯ ====================
def plot_training_history(trainer):
    """Визуализация истории обучения"""
    try:
        import matplotlib.pyplot as plt

        epochs = range(1, len(trainer.train_losses) + 1)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Loss
        axes[0].plot(epochs, trainer.train_losses, 'b-', label='Train Loss')
        axes[0].plot(epochs, trainer.val_losses, 'r-', label='Val Loss')
        axes[0].set_xlabel('Epochs')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)

        # Accuracy
        axes[1].plot(epochs, trainer.train_accs, 'b-', label='Train Accuracy')
        axes[1].plot(epochs, trainer.val_accs, 'r-', label='Val Accuracy')
        axes[1].set_xlabel('Epochs')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True)

        # F1 Score
        axes[2].plot(epochs, trainer.train_f1s, 'b-', label='Train F1')
        axes[2].plot(epochs, trainer.val_f1s, 'r-', label='Val F1')
        axes[2].set_xlabel('Epochs')
        axes[2].set_ylabel('F1 Score')
        axes[2].set_title('Training and Validation F1 Score')
        axes[2].legend()
        axes[2].grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, 'training_history.png'))
        plt.show()
        print(f"✅ Графики сохранены в {SAVE_DIR}/training_history.png")
    except Exception as e:
        print(f"Визуализация не выполнена: {e}")


# ==================== ОСНОВНАЯ ФУНКЦИЯ ====================
def main():
    """Основная функция для запуска обучения"""

    # Проверка наличия CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используется устройство: {device}")

    # Путь к датасету - измените на ваш путь
    # Пример: root = "/Users/username/Downloads/data/Road Issues"
    root = "data/Road Issues"

    # Проверяем различные возможные пути
    possible_paths = [
        root,
        "Road Issues",
        "../data/Road Issues",
        "/kaggle/input/road-issues-detection-dataset/data/Road Issues",
        os.path.expanduser("~/Downloads/data/Road Issues")
    ]

    found_path = None
    for path in possible_paths:
        if os.path.exists(path):
            found_path = path
            break

    if found_path is None:
        print(f"❌ Папка с датасетом не найдена!")
        print("Пожалуйста, укажите правильный путь к датасету в переменной 'root'")
        print("Ожидаемая структура:")
        print("  data/Road Issues/")
        print("    ├── Broken Road Sign Issues/")
        print("    ├── Damaged Road issues/")
        print("    ├── Illegal Parking Issues/")
        print("    ├── Mixed Issues/")
        print("    └── Pothole Issues/")
        return

    root = found_path
    print(f"✅ Датасет найден: {root}")

    # Трансформации для изображений
    train_transform = T.Compose([
        T.Resize((IM_SIZE, IM_SIZE)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=10),
        T.ColorJitter(brightness=0.1, contrast=0.1),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])

    val_transform = T.Compose([
        T.Resize((IM_SIZE, IM_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])

    print("\n📁 Загрузка данных...")

    # Загрузка и разделение данных
    try:
        train_dl, val_dl, test_dl, classes = CustomDataset.stratified_split_dls(
            root=root,
            transformations=train_transform,
            bs=BATCH_SIZE,
            ns=0  # Важно для MacOS!
        )
    except Exception as e:
        print(f"❌ Ошибка загрузки данных: {e}")
        return

    print(f"\n📊 Информация о датасете:")
    print(f"  Количество классов: {len(classes)}")
    print(f"  Размер обучающей выборки: {len(train_dl.dataset)}")
    print(f"  Размер валидационной выборки: {len(val_dl.dataset)}")
    print(f"  Размер тестовой выборки: {len(test_dl.dataset)}")

    if len(train_dl.dataset) == 0:
        print("❌ Нет данных для обучения!")
        return

    print("\n🔧 Создание модели...")

    # Создание модели
    import timm
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=len(classes))
    model = model.to(device)

    print(f"  Модель: {MODEL_NAME}")
    print(f"  Количество параметров: {sum(p.numel() for p in model.parameters()):,}")

    # Создание тренера
    trainer = ModelTrainer(
        model=model,
        classes=classes,
        train_dl=train_dl,
        val_dl=val_dl,
        device=device,
        save_dir=SAVE_DIR,
        save_prefix="road",
        lr=LEARNING_RATE,
        epochs=EPOCHS,
        patience=PATIENCE
    )

    # Запуск обучения
    best_f1 = trainer.train()

    # Сохранение классов в JSON
    import json
    with open(os.path.join(SAVE_DIR, "classes.json"), "w", encoding="utf-8") as f:
        json.dump(classes, f, ensure_ascii=False, indent=2)
    print(f"✅ Классы сохранены в {SAVE_DIR}/classes.json")

    # Сохранение истории обучения
    history = {
        "train_losses": trainer.train_losses,
        "val_losses": trainer.val_losses,
        "train_accs": trainer.train_accs,
        "val_accs": trainer.val_accs,
        "train_f1s": trainer.train_f1s,
        "val_f1s": trainer.val_f1s,
        "best_f1": best_f1,
        "classes": classes,
        "classes_ru": CLASS_NAMES_RU
    }

    with open(os.path.join(SAVE_DIR, "training_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    # Визуализация
    try:
        plot_training_history(trainer)
    except:
        pass

    print("\n" + "=" * 60)
    print("🎉 ОБУЧЕНИЕ УСПЕШНО ЗАВЕРШЕНО!")
    print(f"📁 Модель сохранена в папке: {SAVE_DIR}")
    print(f"📄 Файл модели: {SAVE_DIR}/road_best_model.pth")
    print(f"📄 Файл классов: {SAVE_DIR}/classes.json")
    print(f"📄 История обучения: {SAVE_DIR}/training_history.json")
    print("=" * 60)

    # Вывод итоговых метрик
    print("\n📊 ИТОГОВЫЕ МЕТРИКИ:")
    print(f"  Лучший F1 Score: {best_f1:.4f}")
    print(f"  Финальная точность на валидации: {trainer.val_accs[-1]:.4f}")
    print(f"  Финальная потеря на валидации: {trainer.val_losses[-1]:.4f}")


if __name__ == "__main__":
    main()