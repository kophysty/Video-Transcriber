"""Главное окно приложения."""

import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
from typing import Optional

import customtkinter as ctk

from app.utils.config import AppConfig
from app.models.gpu_detector import detect_gpu, get_device_string, GPUInfo
from app.models.model_manager import ModelManager
from app.models.model_registry import get_all_models, get_model_info
from app.core.pipeline import TranscriptionPipeline, PipelineProgress, CancelledException
from app.core.audio_extractor import is_supported_file


# Поддерживаемые форматы файлов
SUPPORTED_FORMATS = [
    ("Видео и аудио файлы", "*.mp4 *.mkv *.avi *.mov *.webm *.mp3 *.wav *.flac *.ogg *.m4a"),
    ("Видео файлы", "*.mp4 *.mkv *.avi *.mov *.webm"),
    ("Аудио файлы", "*.mp3 *.wav *.flac *.ogg *.m4a"),
    ("Все файлы", "*.*"),
]

# Языки
LANGUAGES = {
    "Авто": "auto",
    "Русский": "ru",
    "English": "en",
    "Deutsch": "de",
    "Français": "fr",
    "Español": "es",
    "中文": "zh",
    "日本語": "ja",
}


class MainWindow(ctk.CTk):
    """Главное окно приложения."""

    def __init__(self):
        super().__init__()

        # Конфигурация
        self.config = AppConfig.load()
        self.model_manager = ModelManager()
        self.gpu_info: Optional[GPUInfo] = None

        # Состояние
        self._pipeline: Optional[TranscriptionPipeline] = None
        self._processing = False

        # Настройка окна
        self.title("Video Transcriber")
        self.geometry("600x550")
        self.minsize(500, 500)

        # Тема
        ctk.set_appearance_mode(self.config.theme)
        ctk.set_default_color_theme("blue")

        # Определяем GPU
        self._detect_gpu()

        # Создаём интерфейс
        self._create_widgets()

        # Загружаем сохранённые настройки
        self._load_settings()

        # Закрытие окна
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _detect_gpu(self) -> None:
        """Определить GPU."""
        self.gpu_info = detect_gpu()

        if self.gpu_info:
            # Обновляем конфиг рекомендациями GPU
            if not self.config.compute_type:
                self.config.compute_type = self.gpu_info.recommended_compute_type
            self.config.device = "cuda"
        else:
            self.config.device = "cpu"
            self.config.compute_type = "int8"

    def _create_widgets(self) -> None:
        """Создать виджеты."""
        # Главный контейнер с отступами
        main_frame = ctk.CTkFrame(self, fg_color="transparent")
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # === Входной файл ===
        self._create_input_section(main_frame)

        # === Выходная папка ===
        self._create_output_section(main_frame)

        # === Настройки ===
        self._create_settings_section(main_frame)

        # === GPU информация ===
        self._create_gpu_section(main_frame)

        # === Прогресс ===
        self._create_progress_section(main_frame)

        # === Кнопки ===
        self._create_buttons_section(main_frame)

    def _create_input_section(self, parent: ctk.CTkFrame) -> None:
        """Секция выбора входного файла."""
        label = ctk.CTkLabel(parent, text="ВХОДНОЙ ФАЙЛ", font=("", 12, "bold"))
        label.pack(anchor="w", pady=(0, 5))

        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.pack(fill="x", pady=(0, 15))

        self.input_entry = ctk.CTkEntry(frame, placeholder_text="Выберите видео или аудио файл...")
        self.input_entry.pack(side="left", fill="x", expand=True, padx=(0, 10))

        browse_btn = ctk.CTkButton(
            frame, text="Обзор", width=80,
            command=self._browse_input
        )
        browse_btn.pack(side="right")

    def _create_output_section(self, parent: ctk.CTkFrame) -> None:
        """Секция выбора выходной папки."""
        label = ctk.CTkLabel(parent, text="ПАПКА ДЛЯ РЕЗУЛЬТАТОВ", font=("", 12, "bold"))
        label.pack(anchor="w", pady=(0, 5))

        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.pack(fill="x", pady=(0, 15))

        self.output_entry = ctk.CTkEntry(frame, placeholder_text="Выберите папку для сохранения...")
        self.output_entry.pack(side="left", fill="x", expand=True, padx=(0, 10))

        browse_btn = ctk.CTkButton(
            frame, text="Обзор", width=80,
            command=self._browse_output
        )
        browse_btn.pack(side="right")

    def _create_settings_section(self, parent: ctk.CTkFrame) -> None:
        """Секция настроек."""
        # Заголовок
        header = ctk.CTkFrame(parent, fg_color="transparent")
        header.pack(fill="x", pady=(10, 10))

        ctk.CTkLabel(header, text="─── Настройки ───", font=("", 12)).pack()

        # Модель
        model_frame = ctk.CTkFrame(parent, fg_color="transparent")
        model_frame.pack(fill="x", pady=(0, 10))

        ctk.CTkLabel(model_frame, text="Модель:", width=80, anchor="w").pack(side="left")

        # Получаем скачанные модели
        downloaded = self.model_manager.list_downloaded_whisper_models()
        model_names = downloaded if downloaded else ["(нет моделей)"]

        self.model_var = ctk.StringVar(value=model_names[0] if downloaded else "")
        self.model_dropdown = ctk.CTkOptionMenu(
            model_frame,
            values=model_names,
            variable=self.model_var,
            width=200,
        )
        self.model_dropdown.pack(side="left", padx=(0, 10))

        manage_btn = ctk.CTkButton(
            model_frame, text="Управление", width=100,
            command=self._open_model_manager
        )
        manage_btn.pack(side="left")

        # Язык
        lang_frame = ctk.CTkFrame(parent, fg_color="transparent")
        lang_frame.pack(fill="x", pady=(0, 10))

        ctk.CTkLabel(lang_frame, text="Язык:", width=80, anchor="w").pack(side="left")

        self.lang_var = ctk.StringVar(value="Авто")
        self.lang_dropdown = ctk.CTkOptionMenu(
            lang_frame,
            values=list(LANGUAGES.keys()),
            variable=self.lang_var,
            width=200,
        )
        self.lang_dropdown.pack(side="left")

        # Диаризация
        diar_frame = ctk.CTkFrame(parent, fg_color="transparent")
        diar_frame.pack(fill="x", pady=(0, 10))

        self.diarization_var = ctk.BooleanVar(value=False)
        self.diarization_check = ctk.CTkCheckBox(
            diar_frame,
            text="Диаризация спикеров (определение кто говорит)",
            variable=self.diarization_var,
        )
        self.diarization_check.pack(side="left")

        # AI-анализ
        ai_frame = ctk.CTkFrame(parent, fg_color="transparent")
        ai_frame.pack(fill="x", pady=(0, 10))

        self.ai_analysis_var = ctk.BooleanVar(value=self.config.enable_ai_analysis)
        self.ai_analysis_check = ctk.CTkCheckBox(
            ai_frame,
            text="AI-анализ (саммари, ключевые моменты)",
            variable=self.ai_analysis_var,
            command=self._on_ai_analysis_toggle,
        )
        self.ai_analysis_check.pack(side="left")

        self.api_key_btn = ctk.CTkButton(
            ai_frame,
            text="API ключ",
            width=80,
            command=self._open_api_key_dialog,
        )
        self.api_key_btn.pack(side="left", padx=(10, 0))

        # Обновляем состояние чекбокса AI-анализа
        self._update_ai_analysis_state()

    def _create_gpu_section(self, parent: ctk.CTkFrame) -> None:
        """Секция информации о GPU."""
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.pack(fill="x", pady=(5, 10))

        gpu_text = get_device_string(self.gpu_info)
        compute_text = self.config.compute_type if self.gpu_info else "int8"

        self.gpu_label = ctk.CTkLabel(
            frame,
            text=f"GPU: {gpu_text} · {compute_text}",
            font=("", 11),
            text_color="gray",
        )
        self.gpu_label.pack(anchor="w")

    def _create_progress_section(self, parent: ctk.CTkFrame) -> None:
        """Секция прогресса."""
        # Заголовок
        header = ctk.CTkFrame(parent, fg_color="transparent")
        header.pack(fill="x", pady=(10, 10))

        ctk.CTkLabel(header, text="─── Прогресс ───", font=("", 12)).pack()

        # Прогресс-бар
        self.progress_bar = ctk.CTkProgressBar(parent)
        self.progress_bar.pack(fill="x", pady=(0, 5))
        self.progress_bar.set(0)

        # Статус
        self.status_label = ctk.CTkLabel(
            parent,
            text="Готов к работе",
            font=("", 11),
        )
        self.status_label.pack(anchor="w")

    def _create_buttons_section(self, parent: ctk.CTkFrame) -> None:
        """Секция кнопок."""
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.pack(fill="x", pady=(20, 0))

        self.start_btn = ctk.CTkButton(
            frame,
            text="Начать транскрибацию",
            font=("", 14, "bold"),
            height=40,
            command=self._start_transcription,
        )
        self.start_btn.pack(fill="x")

        self.cancel_btn = ctk.CTkButton(
            frame,
            text="Отмена",
            font=("", 14),
            height=40,
            fg_color="gray",
            command=self._cancel_transcription,
        )
        # Кнопка отмены изначально скрыта
        self.cancel_btn.pack_forget()

    def _browse_input(self) -> None:
        """Выбрать входной файл."""
        initial_dir = self.config.last_input_dir or str(Path.home())

        file_path = filedialog.askopenfilename(
            title="Выберите видео или аудио файл",
            initialdir=initial_dir,
            filetypes=SUPPORTED_FORMATS,
        )

        if file_path:
            self.input_entry.delete(0, "end")
            self.input_entry.insert(0, file_path)

            # Запоминаем папку
            self.config.last_input_dir = str(Path(file_path).parent)

            # Автоматически предлагаем ту же папку для вывода
            if not self.output_entry.get():
                self.output_entry.insert(0, str(Path(file_path).parent))

    def _browse_output(self) -> None:
        """Выбрать выходную папку."""
        initial_dir = self.config.last_output_dir or self.config.last_input_dir or str(Path.home())

        dir_path = filedialog.askdirectory(
            title="Выберите папку для результатов",
            initialdir=initial_dir,
        )

        if dir_path:
            self.output_entry.delete(0, "end")
            self.output_entry.insert(0, dir_path)
            self.config.last_output_dir = dir_path

    def _open_model_manager(self) -> None:
        """Открыть менеджер моделей."""
        from app.gui.dialogs import ModelManagerDialog

        dialog = ModelManagerDialog(self, self.model_manager, self.config)
        dialog.grab_set()
        self.wait_window(dialog)

        # Обновляем список моделей
        self._refresh_model_list()

    def _refresh_model_list(self) -> None:
        """Обновить список моделей в выпадающем меню."""
        downloaded = self.model_manager.list_downloaded_whisper_models()

        if downloaded:
            self.model_dropdown.configure(values=downloaded)

            # Если текущая модель не скачана, выбираем первую доступную
            if self.model_var.get() not in downloaded:
                self.model_var.set(downloaded[0])
        else:
            self.model_dropdown.configure(values=["(нет моделей)"])
            self.model_var.set("")

    def _start_transcription(self) -> None:
        """Начать транскрибацию."""
        # Валидация
        input_path = self.input_entry.get().strip()
        output_dir = self.output_entry.get().strip()
        model_name = self.model_var.get()

        if not input_path:
            messagebox.showerror("Ошибка", "Выберите входной файл")
            return

        if not Path(input_path).exists():
            messagebox.showerror("Ошибка", f"Файл не найден: {input_path}")
            return

        if not is_supported_file(Path(input_path)):
            messagebox.showerror("Ошибка", "Неподдерживаемый формат файла")
            return

        if not output_dir:
            messagebox.showerror("Ошибка", "Выберите папку для результатов")
            return

        if not model_name or model_name == "(нет моделей)":
            messagebox.showerror(
                "Ошибка",
                "Сначала скачайте модель Whisper.\nНажмите 'Управление' для скачивания."
            )
            return

        # Обновляем конфиг
        self.config.whisper_model = model_name
        self.config.language = LANGUAGES.get(self.lang_var.get(), "auto")
        self.config.enable_diarization = self.diarization_var.get()
        self.config.save()

        # Переключаем UI
        self._set_processing(True)

        # Запускаем в фоновом потоке
        thread = threading.Thread(
            target=self._run_pipeline,
            args=(Path(input_path), Path(output_dir)),
            daemon=True,
        )
        thread.start()

    def _run_pipeline(self, input_path: Path, output_dir: Path) -> None:
        """Запустить пайплайн (в фоновом потоке)."""
        try:
            self._pipeline = TranscriptionPipeline(
                config=self.config,
                gpu_info=self.gpu_info,
                progress_callback=self._on_progress,
            )

            result = self._pipeline.run(input_path, output_dir)

            # Успех
            self.after(0, self._on_complete, result)

        except CancelledException:
            self.after(0, self._on_cancelled)

        except Exception as e:
            self.after(0, self._on_error, str(e))

    def _on_progress(self, progress: PipelineProgress) -> None:
        """Обработчик прогресса (вызывается из фонового потока)."""
        self.after(0, self._update_progress, progress)

    def _update_progress(self, progress: PipelineProgress) -> None:
        """Обновить UI прогресса."""
        self.progress_bar.set(progress.total_progress)
        self.status_label.configure(text=progress.message)

    def _on_complete(self, result) -> None:
        """Обработчик успешного завершения."""
        self._set_processing(False)
        self.progress_bar.set(1.0)

        from app.utils.formats import format_duration
        duration_str = format_duration(result.duration)

        # Формируем сообщение
        message = (
            f"Транскрибация завершена!\n\n"
            f"Длительность: {duration_str}\n"
            f"Сегментов: {len(result.transcription.segments)}\n"
            f"Язык: {result.transcription.info.language}\n"
        )

        # Добавляем инфо об AI-анализе если был
        if result.analysis:
            message += f"\nAI-анализ: выполнен\n"
            if result.analysis.summary.one_liner:
                message += f"\n{result.analysis.summary.one_liner}\n"

        message += f"\nРезультаты сохранены в:\n{result.output_dir}"

        messagebox.showinfo("Готово", message)

        self.status_label.configure(text="Готово!")

    def _on_cancelled(self) -> None:
        """Обработчик отмены."""
        self._set_processing(False)
        self.progress_bar.set(0)
        self.status_label.configure(text="Отменено")

    def _on_error(self, error_msg: str) -> None:
        """Обработчик ошибки."""
        self._set_processing(False)
        self.progress_bar.set(0)
        self.status_label.configure(text="Ошибка")

        messagebox.showerror("Ошибка", f"Произошла ошибка:\n\n{error_msg}")

    def _cancel_transcription(self) -> None:
        """Отменить транскрибацию."""
        if self._pipeline:
            self._pipeline.cancel()
            self.status_label.configure(text="Отмена...")

    def _set_processing(self, processing: bool) -> None:
        """Переключить UI в режим обработки."""
        self._processing = processing

        if processing:
            self.start_btn.pack_forget()
            self.cancel_btn.pack(fill="x")

            # Блокируем ввод
            self.input_entry.configure(state="disabled")
            self.output_entry.configure(state="disabled")
            self.model_dropdown.configure(state="disabled")
            self.lang_dropdown.configure(state="disabled")
            self.diarization_check.configure(state="disabled")
            self.ai_analysis_check.configure(state="disabled")
        else:
            self.cancel_btn.pack_forget()
            self.start_btn.pack(fill="x")

            # Разблокируем ввод
            self.input_entry.configure(state="normal")
            self.output_entry.configure(state="normal")
            self.model_dropdown.configure(state="normal")
            self.lang_dropdown.configure(state="normal")
            self.diarization_check.configure(state="normal")
            self.ai_analysis_check.configure(state="normal")

    def _load_settings(self) -> None:
        """Загрузить сохранённые настройки."""
        # Модель
        if self.config.whisper_model:
            downloaded = self.model_manager.list_downloaded_whisper_models()
            if self.config.whisper_model in downloaded:
                self.model_var.set(self.config.whisper_model)

        # Язык
        for name, code in LANGUAGES.items():
            if code == self.config.language:
                self.lang_var.set(name)
                break

        # Диаризация
        self.diarization_var.set(self.config.enable_diarization)

        # AI-анализ
        self.ai_analysis_var.set(self.config.enable_ai_analysis)
        self._update_ai_analysis_state()

        # Последние пути
        if self.config.last_output_dir:
            self.output_entry.insert(0, self.config.last_output_dir)

    def _on_close(self) -> None:
        """Обработчик закрытия окна."""
        # Сохраняем настройки
        self.config.whisper_model = self.model_var.get()
        self.config.language = LANGUAGES.get(self.lang_var.get(), "auto")
        self.config.enable_diarization = self.diarization_var.get()
        self.config.enable_ai_analysis = self.ai_analysis_var.get()
        self.config.save()

        # Отменяем обработку если идёт
        if self._pipeline:
            self._pipeline.cancel()

        self.destroy()

    def _on_ai_analysis_toggle(self) -> None:
        """Обработчик переключения AI-анализа."""
        if self.ai_analysis_var.get() and not self.config.anthropic_api_key:
            # Если включаем без ключа — открываем диалог
            self._open_api_key_dialog()
            if not self.config.anthropic_api_key:
                # Если ключ не ввели — выключаем обратно
                self.ai_analysis_var.set(False)

    def _open_api_key_dialog(self) -> None:
        """Открыть диалог ввода API ключа."""
        from app.gui.dialogs import ApiKeyDialog

        dialog = ApiKeyDialog(self, self.config)
        dialog.grab_set()
        self.wait_window(dialog)

        self._update_ai_analysis_state()

    def _update_ai_analysis_state(self) -> None:
        """Обновить состояние чекбокса AI-анализа."""
        has_key = bool(self.config.anthropic_api_key)

        if has_key:
            self.api_key_btn.configure(text="API ключ", fg_color=["#3B8ED0", "#1F6AA5"])
        else:
            self.api_key_btn.configure(text="API ключ", fg_color="gray")

        # Если нет ключа — выключаем чекбокс
        if not has_key:
            self.ai_analysis_var.set(False)
