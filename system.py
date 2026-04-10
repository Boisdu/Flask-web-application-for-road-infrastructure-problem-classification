"""
Tkinter приложение для классификации проблем дорожной инфраструктуры
Светлый интерфейс - принудительная установка цветов
При загрузке изображения создаётся новое окно с превью и результатами
"""

import os
import torch
import torch.nn.functional as F
import json
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText
import timm
from torchvision import transforms as T
import threading
import traceback
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# ==================== КОНФИГУРАЦИЯ ====================
IM_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
MODEL_NAME = "resnext101_32x8d"
SAVE_DIR = "saved_models"
MODEL_PATH = os.path.join(SAVE_DIR, "road_best_model.pth")
CLASSES_PATH = os.path.join(SAVE_DIR, "classes.json")

# Светлая цветовая схема (принудительная)
COLORS = {
    "bg_main": "#f0f0f0",  # Основной фон окна
    "bg_card": "#ffffff",  # Фон карточек/фреймов
    "bg_button": "#e0e0e0",  # Фон кнопок
    "bg_canvas": "#f8f8f8",  # Фон канваса
    "fg_text": "#333333",  # Основной цвет текста
    "fg_secondary": "#666666",  # Вторичный текст
    "accent": "#2196F3",  # Акцентный цвет (синий)
    "accent_hover": "#1976D2",  # Акцентный цвет при наведении
    "success": "#4CAF50",  # Успех (зеленый)
    "warning": "#FF9800",  # Предупреждение (оранжевый)
    "error": "#f44336",  # Ошибка (красный)
    "border": "#d0d0d0",  # Цвет границы
    "log_bg": "#1e1e1e",  # Фон лога (темный для контраста)
    "log_fg": "#d4d4d4"  # Текст лога
}

# Русские названия классов
CLASS_NAMES_RU = {
    "Broken Road Sign Issues": "🚫 Поврежденные дорожные знаки",
    "Damaged Road issues": "🛣️ Повреждения дорожного покрытия",
    "Illegal Parking Issues": "🚗 Нарушения парковки",
    "Mixed Issues": "📋 Смешанные проблемы",
    "Pothole Issues": "🕳️ Выбоины"
}

# Рекомендации для каждого класса
RECOMMENDATIONS = {
    "Broken Road Sign Issues": {
        "text": "Требуется замена или ремонт дорожного знака. Рекомендуется:\n• Провести внеплановый осмотр знака на месте\n• Заменить поврежденный знак в течение 1-2 недель\n• Временно установить предупреждающий знак при необходимости",
        "priority": "ВЫСОКИЙ",
        "action": "Срочная замена знака"
    },
    "Damaged Road issues": {
        "text": "Обнаружены повреждения дорожного покрытия. Рекомендуется:\n• Провести детальную инспекцию поврежденного участка\n• Выполнить ямочный ремонт в течение 1 месяца\n• При крупных повреждениях - капитальный ремонт",
        "priority": "СРЕДНИЙ",
        "action": "Плановый ремонт"
    },
    "Illegal Parking Issues": {
        "text": "Выявлены нарушения правил парковки. Рекомендуется:\n• Направить инспектора для проверки\n• Рассмотреть возможность установки дополнительных знаков\n• Провести разъяснительную работу с водителями",
        "priority": "НИЗКИЙ",
        "action": "Контроль соблюдения ПДД"
    },
    "Mixed Issues": {
        "text": "Обнаружено сочетание различных проблем. Рекомендуется:\n• Провести комплексное обследование участка\n• Составить план устранения всех выявленных проблем\n• Распределить задачи по профильным службам",
        "priority": "СРЕДНИЙ",
        "action": "Комплексное обследование"
    },
    "Pothole Issues": {
        "text": "Обнаружены выбоины на дорожном покрытии. Рекомендуется:\n• Немедленно оградить опасный участок при необходимости\n• Выполнить ямочный ремонт в течение 3-7 дней\n• Контролировать качество ремонта",
        "priority": "ВЫСОКИЙ",
        "action": "Срочный ремонт выбоин"
    }
}

# Цвета для приоритетов
PRIORITY_COLORS = {
    "ВЫСОКИЙ": "#e74c3c",
    "СРЕДНИЙ": "#f39c12",
    "НИЗКИЙ": "#27ae60"
}


class RoadIssueClassifier:
    """Класс для классификации проблем дорожной инфраструктуры"""

    def __init__(self, model_path=None, classes_path=None, log_callback=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log_callback = log_callback
        self._log(f"Используется устройство: {self.device}")

        self.model = None
        self.classes = None
        self.classes_reverse = None
        self.model_loaded = False

        # Трансформации для изображений
        self.transform = T.Compose([
            T.Resize((IM_SIZE, IM_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])

        # Загрузка модели и классов
        self.load_model(model_path, classes_path)

    def _log(self, message):
        """Логирование сообщений"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        if self.log_callback:
            self.log_callback(log_msg)

    def load_model(self, model_path=None, classes_path=None):
        """Загрузка обученной модели и классов"""

        # Загрузка классов из JSON файла
        class_path = classes_path or CLASSES_PATH
        if os.path.exists(class_path):
            try:
                with open(class_path, "r", encoding="utf-8") as f:
                    self.classes = json.load(f)
                self.classes_reverse = {v: k for k, v in self.classes.items()}
                self._log(f"✅ Загружены классы: {list(self.classes.keys())}")
            except Exception as e:
                self._log(f"❌ Ошибка загрузки классов: {e}")
                self._set_default_classes()
        else:
            self._log(f"⚠️ Файл классов не найден: {class_path}")
            self._set_default_classes()

        # Загрузка модели
        model_path = model_path or MODEL_PATH
        if not os.path.exists(model_path):
            self._log(f"❌ Модель не найдена: {model_path}")
            return False

        try:
            num_classes = len(self.classes)
            self._log(f"Создание модели с {num_classes} классами...")
            self.model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=num_classes)

            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                if 'classes' in checkpoint and checkpoint['classes']:
                    self.classes = checkpoint['classes']
                    self.classes_reverse = {v: k for k, v in self.classes.items()}
            else:
                self.model.load_state_dict(checkpoint)

            self.model.to(self.device)
            self.model.eval()
            self.model_loaded = True
            self._log(f"✅ Модель успешно загружена из {model_path}")
            return True

        except Exception as e:
            self._log(f"❌ Ошибка загрузки модели: {e}")
            traceback.print_exc()
            self.model_loaded = False
            return False

    def _set_default_classes(self):
        """Установка стандартных классов"""
        self.classes = {
            "Broken Road Sign Issues": 0,
            "Damaged Road issues": 1,
            "Illegal Parking Issues": 2,
            "Mixed Issues": 3,
            "Pothole Issues": 4
        }
        self.classes_reverse = {v: k for k, v in self.classes.items()}

    def predict(self, image):
        """Предсказание класса проблемы"""
        if not self.model_loaded:
            self._log("Модель не загружена!")
            return None, None, None, None, None

        try:
            if isinstance(image, str):
                img = Image.open(image).convert("RGB")
            else:
                img = image

            img_tensor = self.transform(img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(img_tensor)
                probabilities = F.softmax(outputs, dim=1)
                pred_class_id = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][pred_class_id].item()
                all_probs = probabilities[0].cpu().numpy()

            self._log(f"Предсказанный класс ID: {pred_class_id}, уверенность: {confidence:.4f}")

            if self.classes_reverse and pred_class_id in self.classes_reverse:
                class_name = self.classes_reverse[pred_class_id]
            else:
                class_name = list(self.classes.keys())[pred_class_id] if self.classes else f"Класс {pred_class_id}"

            self._log(f"Имя класса: {class_name}")

            class_name_ru = CLASS_NAMES_RU.get(class_name, class_name)
            recommendation = RECOMMENDATIONS.get(class_name, RECOMMENDATIONS.get("Mixed Issues", {
                "text": "Требуется осмотр специалистом.",
                "priority": "СРЕДНИЙ",
                "action": "Провести осмотр"
            }))

            return class_name, class_name_ru, confidence, recommendation, all_probs

        except Exception as e:
            self._log(f"Ошибка при предсказании: {e}")
            traceback.print_exc()
            return None, None, None, None, None


class RoadIssueApp:
    """Главное окно приложения (светлая тема)"""

    def __init__(self, root):
        self.root = root

        # Принудительная установка светлой темы для Tkinter
        self.root.configure(bg=COLORS["bg_main"])

        # Попытка установить светлую тему для ttk (если доступно)
        try:
            style = ttk.Style()
            available_themes = style.theme_names()
            if 'clam' in available_themes:
                style.theme_use('clam')
            elif 'default' in available_themes:
                style.theme_use('default')
            style.configure('TFrame', background=COLORS["bg_main"])
            style.configure('TLabel', background=COLORS["bg_main"], foreground=COLORS["fg_text"])
            style.configure('TButton', background=COLORS["bg_button"])
        except:
            pass

        self.root.title("Система мониторинга дорожной инфраструктуры")
        self.root.geometry("1500x950")
        self.root.minsize(1200, 700)

        # Переменные для хранения текущего изображения
        self.current_image = None
        self.current_photo = None
        self.current_image_path = None

        # Переменные для окна результатов
        self.result_window = None
        self.result_canvas = None
        self.result_photo = None
        self.result_output = None
        self.result_image_info = None

        # Создание интерфейса
        self._create_widgets()

        # Инициализация классификатора
        self.classifier = RoadIssueClassifier(log_callback=self._add_log)

        # Проверка загрузки модели
        self._check_model_status()

    def _add_log(self, message):
        """Добавление сообщения в лог"""
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()

    def _check_model_status(self):
        """Проверка статуса загрузки модели"""
        if not self.classifier.model_loaded:
            self.model_status_label.config(
                text="⚠️ МОДЕЛЬ НЕ ЗАГРУЖЕНА",
                fg=COLORS["error"]
            )
            self.btn_analyze.config(state=tk.DISABLED)
            self.status_label.config(text="❌ Модель не загружена")
        else:
            self.model_status_label.config(
                text="✅ МОДЕЛЬ УСПЕШНО ЗАГРУЖЕНА",
                fg=COLORS["success"]
            )
            self.status_label.config(text="✅ Система готова к работе | Загрузите изображение")

    def _create_widgets(self):
        """Создание элементов интерфейса"""

        # ==================== Верхняя панель (шапка) ====================
        top_frame = tk.Frame(self.root, bg=COLORS["accent"], height=90)
        top_frame.pack(fill=tk.X, side=tk.TOP)
        top_frame.pack_propagate(False)

        title_label = tk.Label(
            top_frame,
            text="🏗️ СИСТЕМА АВТОМАТИЧЕСКОГО МОНИТОРИНГА ДОРОЖНОЙ ИНФРАСТРУКТУРЫ 🏗️",
            font=("Arial", 14, "bold"),
            fg="white",
            bg=COLORS["accent"]
        )
        title_label.pack(expand=True)

        # ==================== Панель кнопок ====================
        button_container = tk.Frame(self.root, bg=COLORS["bg_main"], height=70)
        button_container.pack(fill=tk.X, padx=20, pady=(15, 10))
        button_container.pack_propagate(False)

        btn_frame = tk.Frame(button_container, bg=COLORS["bg_main"])
        btn_frame.pack(expand=True)

        self.btn_load = tk.Button(
            btn_frame,
            text="📁 ЗАГРУЗИТЬ ИЗОБРАЖЕНИЕ",
            font=("Arial", 11, "bold"),
            bg=COLORS["accent"],
            fg="white",
            padx=25,
            pady=10,
            cursor="hand2",
            relief=tk.FLAT,
            bd=0,
            activebackground=COLORS["accent_hover"],
            activeforeground="white",
            command=self.load_image
        )
        self.btn_load.pack(side=tk.LEFT, padx=8)

        self.btn_analyze = tk.Button(
            btn_frame,
            text="🔍 ВЫПОЛНИТЬ АНАЛИЗ",
            font=("Arial", 11, "bold"),
            bg=COLORS["success"],
            fg="white",
            padx=25,
            pady=10,
            cursor="hand2",
            relief=tk.FLAT,
            bd=0,
            activebackground="#45a049",
            activeforeground="white",
            command=self.analyze_image,
            state=tk.DISABLED
        )
        self.btn_analyze.pack(side=tk.LEFT, padx=8)

        self.btn_clear = tk.Button(
            btn_frame,
            text="🗑 ОЧИСТИТЬ ВСЁ",
            font=("Arial", 11, "bold"),
            bg=COLORS["warning"],
            fg="white",
            padx=25,
            pady=10,
            cursor="hand2",
            relief=tk.FLAT,
            bd=0,
            activebackground="#e68900",
            activeforeground="white",
            command=self.clear_all
        )
        self.btn_clear.pack(side=tk.LEFT, padx=8)

        # ==================== Основная область ====================
        main_container = tk.Frame(self.root, bg=COLORS["bg_main"])
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Левая колонка - изображение
        left_card = tk.Frame(main_container, bg=COLORS["bg_card"], relief=tk.RAISED, bd=1)
        left_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        left_title = tk.Label(
            left_card,
            text="📷 ЗАГРУЖЕННОЕ ИЗОБРАЖЕНИЕ",
            font=("Arial", 11, "bold"),
            bg=COLORS["bg_card"],
            fg=COLORS["fg_text"],
            pady=10
        )
        left_title.pack(fill=tk.X)

        sep1 = tk.Frame(left_card, bg=COLORS["border"], height=1)
        sep1.pack(fill=tk.X, padx=10)

        self.canvas = tk.Canvas(
            left_card,
            bg=COLORS["bg_canvas"],
            relief=tk.SUNKEN,
            bd=1,
            highlightthickness=0
        )
        self.canvas.pack(padx=15, pady=15, fill=tk.BOTH, expand=True)

        self.image_info_label = tk.Label(
            left_card,
            text="",
            font=("Arial", 9),
            bg=COLORS["bg_card"],
            fg=COLORS["fg_secondary"],
            pady=8
        )
        self.image_info_label.pack()

        # Правая колонка - результаты
        right_card = tk.Frame(main_container, bg=COLORS["bg_card"], relief=tk.RAISED, bd=1)
        right_card.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))

        right_title = tk.Label(
            right_card,
            text="📊 РЕЗУЛЬТАТЫ АНАЛИЗА",
            font=("Arial", 11, "bold"),
            bg=COLORS["accent"],
            fg="white",
            pady=10
        )
        right_title.pack(fill=tk.X)

        status_frame = tk.Frame(right_card, bg=COLORS["bg_card"])
        status_frame.pack(fill=tk.X, padx=15, pady=(10, 5))

        self.model_status_label = tk.Label(
            status_frame,
            text="⏳ Проверка загрузки модели...",
            font=("Arial", 10, "bold"),
            bg=COLORS["bg_card"],
            fg=COLORS["warning"]
        )
        self.model_status_label.pack()

        results_frame = tk.Frame(right_card, bg=COLORS["bg_card"])
        results_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)

        self.result_text = ScrolledText(
            results_frame,
            font=("Segoe UI", 10),
            wrap=tk.WORD,
            bg=COLORS["bg_canvas"],
            fg=COLORS["fg_text"],
            relief=tk.FLAT,
            bd=0,
            height=12
        )
        self.result_text.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Настройка тегов для результатов
        self.result_text.tag_config("header", font=("Arial", 11, "bold"), foreground=COLORS["accent"])
        self.result_text.tag_config("class_name", font=("Arial", 11, "bold"), foreground=COLORS["error"])
        self.result_text.tag_config("high_priority", foreground=PRIORITY_COLORS["ВЫСОКИЙ"], font=("Arial", 10, "bold"))
        self.result_text.tag_config("mid_priority", foreground=PRIORITY_COLORS["СРЕДНИЙ"], font=("Arial", 10, "bold"))
        self.result_text.tag_config("low_priority", foreground=PRIORITY_COLORS["НИЗКИЙ"], font=("Arial", 10, "bold"))
        self.result_text.tag_config("info", foreground=COLORS["fg_secondary"])
        self.result_text.tag_config("recommend", foreground=COLORS["success"])

        # Лог-терминал
        log_frame = tk.Frame(right_card, bg=COLORS["bg_card"])
        log_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 10))

        log_title = tk.Label(
            log_frame,
            text="📋 ЛОГ СИСТЕМЫ",
            font=("Arial", 10, "bold"),
            bg=COLORS["bg_card"],
            fg=COLORS["fg_text"],
            anchor=tk.W
        )
        log_title.pack(fill=tk.X, pady=(0, 5))

        self.log_text = ScrolledText(
            log_frame,
            font=("Consolas", 9),
            wrap=tk.WORD,
            bg=COLORS["log_bg"],
            fg=COLORS["log_fg"],
            relief=tk.FLAT,
            bd=0,
            height=8
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # Нижняя панель статуса
        status_bar = tk.Frame(self.root, bg=COLORS["accent"], height=35)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)

        self.status_label = tk.Label(
            status_bar,
            text="🔄 Инициализация системы...",
            font=("Arial", 9),
            bg=COLORS["accent"],
            fg="white",
            anchor=tk.W
        )
        self.status_label.pack(fill=tk.X, padx=15, pady=8)

    def _create_result_window(self):
        """Создание нового окна с изображением и результатами"""
        # Закрываем старое окно, если есть
        if hasattr(self, 'result_window') and self.result_window:
            try:
                self.result_window.destroy()
            except:
                pass

        title = f"Анализ: {os.path.basename(self.current_image_path)}" if self.current_image_path else "Результаты анализа"
        self.result_window = tk.Toplevel(self.root)
        self.result_window.title(title)
        self.result_window.geometry("1400x800")
        self.result_window.configure(bg=COLORS["bg_main"])

        # Обработчик закрытия окна
        self.result_window.protocol("WM_DELETE_WINDOW", self._on_result_window_close)

        # --- Левая панель: изображение ---
        left_panel = tk.Frame(self.result_window, bg=COLORS["bg_card"], width=600)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        left_panel.pack_propagate(False)

        tk.Label(
            left_panel,
            text="📷 ЗАГРУЖЕННОЕ ИЗОБРАЖЕНИЕ",
            font=("Arial", 11, "bold"),
            bg=COLORS["bg_card"],
            fg=COLORS["fg_text"]
        ).pack(pady=(10, 5))

        self.result_canvas = tk.Canvas(
            left_panel,
            bg=COLORS["bg_canvas"],
            relief=tk.SUNKEN,
            bd=1,
            highlightthickness=0
        )
        self.result_canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.result_image_info = tk.Label(
            left_panel,
            text="",
            font=("Arial", 9),
            bg=COLORS["bg_card"],
            fg=COLORS["fg_secondary"],
            justify=tk.LEFT
        )
        self.result_image_info.pack(pady=5)

        # --- Правая панель: результаты ---
        right_panel = tk.Frame(self.result_window, bg=COLORS["bg_card"], width=700)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        right_panel.pack_propagate(False)

        tk.Label(
            right_panel,
            text="📊 РЕЗУЛЬТАТЫ АНАЛИЗА",
            font=("Arial", 11, "bold"),
            bg=COLORS["accent"],
            fg="white"
        ).pack(fill=tk.X)

        self.result_output = ScrolledText(
            right_panel,
            font=("Segoe UI", 10),
            wrap=tk.WORD,
            bg=COLORS["bg_canvas"],
            fg=COLORS["fg_text"],
            relief=tk.FLAT,
            bd=0,
            state=tk.NORMAL
        )
        self.result_output.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Настройка тегов (аналогично основному окну)
        self.result_output.tag_config("header", font=("Arial", 11, "bold"), foreground=COLORS["accent"])
        self.result_output.tag_config("class_name", font=("Arial", 11, "bold"), foreground=COLORS["error"])
        self.result_output.tag_config("high_priority", foreground=PRIORITY_COLORS["ВЫСОКИЙ"],
                                      font=("Arial", 10, "bold"))
        self.result_output.tag_config("mid_priority", foreground=PRIORITY_COLORS["СРЕДНИЙ"], font=("Arial", 10, "bold"))
        self.result_output.tag_config("low_priority", foreground=PRIORITY_COLORS["НИЗКИЙ"], font=("Arial", 10, "bold"))
        self.result_output.tag_config("info", foreground=COLORS["fg_secondary"])
        self.result_output.tag_config("recommend", foreground=COLORS["success"])

        # Отображаем изображение сразу при создании окна
        if self.current_image:
            self._display_image_in_result_window(self.current_image)

    def _on_result_window_close(self):
        """Обработчик закрытия окна результатов"""
        if hasattr(self, 'result_window') and self.result_window:
            self.result_window.destroy()
        self.result_window = None
        self.result_canvas = None
        self.result_photo = None
        self.result_output = None
        self.result_image_info = None

    def _display_image_in_result_window(self, image, max_width=550, max_height=600):
        """Отображение изображения в новом окне результатов"""
        if not hasattr(self, 'result_canvas') or not self.result_canvas or not self.result_window:
            return

        canvas_width = max_width
        canvas_height = max_height

        img_width, img_height = image.size
        ratio = min(canvas_width / img_width, canvas_height / img_height)
        new_width = int(img_width * ratio)
        new_height = int(img_height * ratio)

        resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        self.result_photo = ImageTk.PhotoImage(resized)

        self.result_canvas.delete("all")
        x = (canvas_width - new_width) // 2
        y = (canvas_height - new_height) // 2
        self.result_canvas.create_image(x, y, anchor=tk.NW, image=self.result_photo)
        self.result_canvas.config(scrollregion=self.result_canvas.bbox(tk.ALL))

        # Обновляем информацию об изображении
        if self.current_image_path and self.result_image_info:
            try:
                file_size = os.path.getsize(self.current_image_path) / 1024
                self.result_image_info.config(
                    text=f"📐 {img_width}×{img_height} px | 💾 {file_size:.1f} KB\n📁 {os.path.basename(self.current_image_path)}"
                )
            except:
                self.result_image_info.config(
                    text=f"📐 {img_width}×{img_height} px | 📁 {os.path.basename(self.current_image_path)}"
                )

    def load_image(self):
        """Загрузка изображения из файла"""
        file_path = filedialog.askopenfilename(
            title="Выберите изображение",
            filetypes=[
                ("Изображения", "*.jpg *.jpeg *.png *.bmp *.JPG"),
                ("Все файлы", "*.*")
            ]
        )

        if file_path:
            try:
                self.current_image = Image.open(file_path).convert("RGB")
                self.current_image_path = file_path
                self._display_image(self.current_image)
                self.btn_analyze.config(state=tk.NORMAL)

                # === СОЗДАЁМ НОВОЕ ОКНО С ИЗОБРАЖЕНИЕМ ===
                self._create_result_window()

                img_size = self.current_image.size
                file_size = os.path.getsize(file_path) / 1024

                self.image_info_label.config(
                    text=f"📐 Размер: {img_size[0]}×{img_size[1]} px | 💾 {file_size:.1f} KB | 📁 {os.path.basename(file_path)}"
                )
                self.status_label.config(text=f"📁 Загружено: {os.path.basename(file_path)}")
                self._clear_results()
                self._add_log(f"📁 Изображение загружено: {os.path.basename(file_path)} ({img_size[0]}x{img_size[1]})")

            except Exception as e:
                self._add_log(f"❌ Ошибка загрузки: {str(e)}")
                messagebox.showerror("Ошибка", f"Не удалось загрузить изображение:\n{str(e)}")
                self.status_label.config(text="❌ Ошибка загрузки изображения")

    def _display_image(self, image, max_width=520, max_height=400):
        """Отображение изображения на канвасе основного окна"""
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        if canvas_width <= 1:
            canvas_width = max_width
        if canvas_height <= 1:
            canvas_height = max_height

        img_width, img_height = image.size
        ratio = min(canvas_width / img_width, canvas_height / img_height)
        new_width = int(img_width * ratio)
        new_height = int(img_height * ratio)

        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        self.current_photo = ImageTk.PhotoImage(resized_image)

        self.canvas.delete("all")
        x = (canvas_width - new_width) // 2
        y = (canvas_height - new_height) // 2
        self.canvas.create_image(x, y, anchor=tk.NW, image=self.current_photo)

    def analyze_image(self):
        """Анализ загруженного изображения"""
        self._add_log("=" * 50)
        self._add_log("🔍 Начало анализа изображения")

        if self.current_image is None:
            self._add_log("❌ Ошибка: изображение не загружено")
            messagebox.showwarning("Предупреждение", "Сначала загрузите изображение!")
            return

        if not self.classifier.model_loaded:
            self._add_log("❌ Ошибка: модель не загружена")
            messagebox.showerror("Ошибка", "Модель не загружена!")
            return

        self._add_log(f"📷 Изображение: {os.path.basename(self.current_image_path)}")
        self._add_log(f"📐 Размер: {self.current_image.size}")

        self.status_label.config(text="🔍 Выполняется анализ изображения...")
        self.btn_analyze.config(state=tk.DISABLED)
        self._clear_results()

        def analyze():
            try:
                self._add_log("🔄 Запуск нейросети...")
                result = self.classifier.predict(self.current_image)

                class_name, class_name_ru, confidence, recommendation, all_probs = result

                if class_name is None:
                    self._add_log("❌ Ошибка: предсказание вернуло None")
                    self.root.after(0, lambda: self._show_error("Ошибка при анализе изображения"))
                else:
                    self._add_log(f"✅ Анализ завершен успешно!")
                    self._add_log(f"📊 Результат: {class_name_ru} (уверенность: {confidence:.2%})")

                    self.root.after(0, lambda: self._display_results(
                        class_name, class_name_ru, confidence, recommendation, all_probs
                    ))
                    self.root.after(0, lambda: self.status_label.config(
                        text=f"✅ Анализ завершен | Результат: {class_name_ru}"
                    ))

            except Exception as e:
                self._add_log(f"❌ Ошибка в потоке анализа: {str(e)}")
                traceback.print_exc()
                self.root.after(0, lambda: self._show_error(str(e)))

            finally:
                self.root.after(0, lambda: self.btn_analyze.config(state=tk.NORMAL))

        threading.Thread(target=analyze, daemon=True).start()

    def _format_result_text(self, class_name, class_name_ru, confidence, recommendation, all_probs):
        """Форматирование текста результатов (возвращает строку)"""
        lines = []
        lines.append("═" * 65 + "\n")
        lines.append(" " * 22 + "РЕЗУЛЬТАТ АНАЛИЗА\n")
        lines.append("═" * 65 + "\n\n")

        lines.append("📷 ИНФОРМАЦИЯ ОБ ИЗОБРАЖЕНИИ:\n")
        lines.append(f"   Файл: {os.path.basename(self.current_image_path)}\n")
        lines.append(f"   Размер: {self.current_image.size[0]}×{self.current_image.size[1]} px\n\n")

        lines.append("🏷️  ОПРЕДЕЛЕННЫЙ ТИП ПРОБЛЕМЫ:\n")
        lines.append(f"   {class_name_ru}\n\n")

        if confidence is not None:
            lines.append("📊 УВЕРЕННОСТЬ ПРЕДСКАЗАНИЯ:\n")
            confidence_percent = confidence * 100
            if confidence_percent > 80:
                lines.append(f"   {confidence_percent:.1f}% (высокая)\n\n")
            elif confidence_percent > 60:
                lines.append(f"   {confidence_percent:.1f}% (средняя)\n\n")
            else:
                lines.append(f"   {confidence_percent:.1f}% (низкая)\n\n")

        if recommendation:
            priority = recommendation.get("priority", "СРЕДНИЙ")
            lines.append("⚠️  ПРИОРИТЕТ ОБСЛУЖИВАНИЯ:\n")
            lines.append(f"   {priority}\n\n")

            lines.append("🛠️  РЕКОМЕНДУЕМЫЕ ДЕЙСТВИЯ:\n")
            lines.append(f"   {recommendation.get('action', 'Провести осмотр')}\n\n")

            lines.append("📋  ПОДРОБНЫЕ РЕКОМЕНДАЦИИ:\n")
            lines.append(f"{recommendation.get('text', 'Требуется осмотр специалистом.')}\n\n")

        if all_probs is not None and self.classifier.classes_reverse:
            lines.append("═" * 65 + "\n")
            lines.append("📊  ВЕРОЯТНОСТИ ПО ВСЕМ КЛАССАМ:\n")
            lines.append("═" * 65 + "\n")

            for i, prob in enumerate(all_probs):
                if self.classifier.classes_reverse and i in self.classifier.classes_reverse:
                    class_key = self.classifier.classes_reverse[i]
                else:
                    class_key = list(self.classifier.classes.keys())[i] if self.classifier.classes else f"Класс {i}"

                class_ru = CLASS_NAMES_RU.get(class_key, class_key)
                bar_length = int(prob * 30)
                bar = "█" * bar_length + "░" * (30 - bar_length)
                lines.append(f"   {class_ru[:30]:<30} {bar} {prob * 100:.1f}%\n")

        lines.append("\n" + "═" * 65 + "\n")
        lines.append(f"⏱️  Время анализа: {datetime.now().strftime('%H:%M:%S')}\n")
        lines.append("ℹ️  ДОПОЛНИТЕЛЬНАЯ ИНФОРМАЦИЯ:\n")
        lines.append("   • Для точной оценки рекомендуется несколько изображений\n")
        lines.append("   • Результат носит рекомендательный характер\n")
        lines.append("   • Окончательное решение принимает инспектор\n")

        return "".join(lines)

    def _apply_result_tags(self, text_widget):
        """Применение форматирования к тексту результатов через теги"""
        content = text_widget.get("1.0", tk.END)
        text_widget.delete("1.0", tk.END)

        lines = content.split('\n')
        for line in lines:
            if "РЕЗУЛЬТАТ АНАЛИЗА" in line or "═" * 10 in line:
                text_widget.insert(tk.END, line + "\n", "header")
            elif "🏷️" in line or "ОПРЕДЕЛЕННЫЙ ТИП" in line:
                text_widget.insert(tk.END, line + "\n", "header")
            elif any(cn in line for cn in CLASS_NAMES_RU.values()):
                text_widget.insert(tk.END, line + "\n", "class_name")
            elif "ПРИОРИТЕТ" in line or "⚠️" in line:
                text_widget.insert(tk.END, line + "\n", "header")
            elif "ВЫСОКИЙ" in line:
                text_widget.insert(tk.END, line + "\n", "high_priority")
            elif "СРЕДНИЙ" in line:
                text_widget.insert(tk.END, line + "\n", "mid_priority")
            elif "НИЗКИЙ" in line:
                text_widget.insert(tk.END, line + "\n", "low_priority")
            elif "🛠️" in line or "📋" in line or "РЕКОМЕНДУЕМЫЕ" in line:
                text_widget.insert(tk.END, line + "\n", "header")
            elif "•" in line or "Файл:" in line or "Размер:" in line or "Время анализа" in line:
                text_widget.insert(tk.END, line + "\n", "info")
            elif any(action in line for action in ["Срочная", "Плановый", "Контроль", "Комплексное"]):
                text_widget.insert(tk.END, line + "\n", "recommend")
            else:
                text_widget.insert(tk.END, line + "\n")

    def _display_results(self, class_name, class_name_ru, confidence, recommendation, all_probs):
        """Отображение результатов анализа в обоих окнах"""
        result_content = self._format_result_text(class_name, class_name_ru, confidence, recommendation, all_probs)

        # Обновляем основное окно
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, result_content)
        self._apply_result_tags(self.result_text)
        self.result_text.see(tk.END)

        # Обновляем новое окно результатов, если оно открыто
        if hasattr(self,
                   'result_output') and self.result_output and self.result_window and self.result_window.winfo_exists():
            self.result_output.delete(1.0, tk.END)
            self.result_output.insert(tk.END, result_content)
            self._apply_result_tags(self.result_output)
            self.result_output.see(tk.END)

    def _clear_results(self):
        """Очистка результатов в обоих окнах"""
        self.result_text.delete(1.0, tk.END)
        if hasattr(self, 'result_output') and self.result_output:
            self.result_output.delete(1.0, tk.END)

    def _show_error(self, error_message):
        """Отображение ошибки"""
        self._add_log(f"❌ Ошибка: {error_message}")
        messagebox.showerror("Ошибка анализа", f"Произошла ошибка:\n{error_message}")
        self.status_label.config(text="❌ Ошибка анализа")

    def clear_all(self):
        """Очистка всех данных"""
        # Закрываем окно результатов, если открыто
        if hasattr(self, 'result_window') and self.result_window:
            try:
                self.result_window.destroy()
            except:
                pass
            self.result_window = None
            self.result_canvas = None
            self.result_photo = None
            self.result_output = None
            self.result_image_info = None

        self.current_image = None
        self.current_image_path = None
        self.current_photo = None
        self.canvas.delete("all")
        self.btn_analyze.config(state=tk.DISABLED)
        self.image_info_label.config(text="")
        self._clear_results()
        self.status_label.config(text="✅ Система очищена | Загрузите новое изображение")
        self._add_log("🗑 Все данные очищены")


def main():
    """Главная функция запуска приложения"""
    print("Запуск приложения...")
    root = tk.Tk()

    try:
        root.tk.call('tk', 'windowingsystem')
    except:
        pass

    app = RoadIssueApp(root)

    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'{width}x{height}+{x}+{y}')

    print("Приложение запущено")
    root.mainloop()


if __name__ == "__main__":
    main()