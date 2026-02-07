"""Диалоговые окна."""

import threading
import tkinter as tk
from tkinter import messagebox
from typing import Optional

import customtkinter as ctk

from app.utils.config import AppConfig
from app.models.model_manager import ModelManager
from app.models.model_registry import get_all_models, get_model_info, WHISPER_MODELS
from app.utils.formats import format_file_size


class ModelManagerDialog(ctk.CTkToplevel):
    """Диалог управления моделями."""

    def __init__(
        self,
        parent,
        model_manager: ModelManager,
        config: AppConfig,
    ):
        super().__init__(parent)

        self.model_manager = model_manager
        self.config = config
        self._downloading = False

        # Настройка окна
        self.title("Управление моделями")
        self.geometry("550x500")
        self.minsize(450, 400)

        # Создаём интерфейс
        self._create_widgets()

        # Центрируем относительно родителя
        self.transient(parent)
        self.update_idletasks()

        x = parent.winfo_x() + (parent.winfo_width() - self.winfo_width()) // 2
        y = parent.winfo_y() + (parent.winfo_height() - self.winfo_height()) // 2
        self.geometry(f"+{x}+{y}")

    def _create_widgets(self) -> None:
        """Создать виджеты."""
        # Главный контейнер
        main_frame = ctk.CTkFrame(self, fg_color="transparent")
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # === Скачанные модели ===
        ctk.CTkLabel(
            main_frame,
            text="Скачанные модели",
            font=("", 14, "bold"),
        ).pack(anchor="w", pady=(0, 10))

        self.downloaded_frame = ctk.CTkScrollableFrame(main_frame, height=150)
        self.downloaded_frame.pack(fill="x", pady=(0, 20))

        # === Доступные модели ===
        ctk.CTkLabel(
            main_frame,
            text="Доступные для скачивания",
            font=("", 14, "bold"),
        ).pack(anchor="w", pady=(0, 10))

        self.available_frame = ctk.CTkScrollableFrame(main_frame, height=150)
        self.available_frame.pack(fill="x", pady=(0, 20))

        # === Прогресс скачивания ===
        self.progress_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        self.progress_frame.pack(fill="x", pady=(0, 10))

        self.progress_bar = ctk.CTkProgressBar(self.progress_frame)
        self.progress_bar.pack(fill="x")
        self.progress_bar.set(0)

        self.progress_label = ctk.CTkLabel(
            self.progress_frame,
            text="",
            font=("", 11),
        )
        self.progress_label.pack(anchor="w")

        # Скрываем прогресс изначально
        self.progress_frame.pack_forget()

        # === Общий размер ===
        self.size_label = ctk.CTkLabel(
            main_frame,
            text="",
            font=("", 11),
            text_color="gray",
        )
        self.size_label.pack(anchor="w", pady=(10, 0))

        # Заполняем списки
        self._refresh_lists()

    def _refresh_lists(self) -> None:
        """Обновить списки моделей."""
        # Очищаем
        for widget in self.downloaded_frame.winfo_children():
            widget.destroy()
        for widget in self.available_frame.winfo_children():
            widget.destroy()

        downloaded = self.model_manager.list_downloaded_whisper_models()

        # Скачанные
        if downloaded:
            for model_name in downloaded:
                self._add_downloaded_model_row(model_name)
        else:
            ctk.CTkLabel(
                self.downloaded_frame,
                text="Нет скачанных моделей",
                text_color="gray",
            ).pack(pady=10)

        # Доступные
        for model_info in get_all_models():
            if model_info.name not in downloaded:
                self._add_available_model_row(model_info)

        # Общий размер
        total_size = self.model_manager.get_total_models_size()
        self.size_label.configure(text=f"Общий размер моделей: {format_file_size(total_size)}")

    def _add_downloaded_model_row(self, model_name: str) -> None:
        """Добавить строку скачанной модели."""
        model_info = get_model_info(model_name)
        if not model_info:
            return

        frame = ctk.CTkFrame(self.downloaded_frame)
        frame.pack(fill="x", pady=2)

        # Имя и описание
        info_frame = ctk.CTkFrame(frame, fg_color="transparent")
        info_frame.pack(side="left", fill="x", expand=True, padx=10, pady=5)

        ctk.CTkLabel(
            info_frame,
            text=model_name,
            font=("", 12, "bold"),
        ).pack(anchor="w")

        size = self.model_manager.get_whisper_model_size(model_name)
        ctk.CTkLabel(
            info_frame,
            text=f"{model_info.description_ru} · {format_file_size(size)}",
            font=("", 10),
            text_color="gray",
        ).pack(anchor="w")

        # Кнопка удаления
        delete_btn = ctk.CTkButton(
            frame,
            text="Удалить",
            width=80,
            fg_color="gray",
            command=lambda m=model_name: self._delete_model(m),
        )
        delete_btn.pack(side="right", padx=10, pady=5)

    def _add_available_model_row(self, model_info) -> None:
        """Добавить строку доступной модели."""
        frame = ctk.CTkFrame(self.available_frame)
        frame.pack(fill="x", pady=2)

        # Имя и описание
        info_frame = ctk.CTkFrame(frame, fg_color="transparent")
        info_frame.pack(side="left", fill="x", expand=True, padx=10, pady=5)

        ctk.CTkLabel(
            info_frame,
            text=model_info.name,
            font=("", 12, "bold"),
        ).pack(anchor="w")

        ctk.CTkLabel(
            info_frame,
            text=f"{model_info.description_ru} · ~{format_file_size(model_info.size_mb * 1024 * 1024)} · VRAM: {model_info.vram_fp16_mb} MB",
            font=("", 10),
            text_color="gray",
        ).pack(anchor="w")

        # Кнопка скачивания
        download_btn = ctk.CTkButton(
            frame,
            text="Скачать",
            width=80,
            command=lambda m=model_info.name: self._download_model(m),
        )
        download_btn.pack(side="right", padx=10, pady=5)

    def _download_model(self, model_name: str) -> None:
        """Скачать модель."""
        if self._downloading:
            messagebox.showwarning("Подождите", "Уже идёт скачивание")
            return

        self._downloading = True
        self.progress_frame.pack(fill="x", pady=(0, 10))
        self.progress_bar.set(0)
        self.progress_label.configure(text=f"Скачивание {model_name}...")

        # Скачиваем в фоне
        thread = threading.Thread(
            target=self._download_thread,
            args=(model_name,),
            daemon=True,
        )
        thread.start()

    def _download_thread(self, model_name: str) -> None:
        """Поток скачивания."""
        try:
            def progress_callback(progress: float, message: str):
                self.after(0, self._update_download_progress, progress, message)

            self.model_manager.download_whisper_model(
                model_name,
                progress_callback=progress_callback,
            )

            self.after(0, self._download_complete, model_name)

        except Exception as e:
            self.after(0, self._download_error, str(e))

    def _update_download_progress(self, progress: float, message: str) -> None:
        """Обновить прогресс скачивания."""
        self.progress_bar.set(progress)
        self.progress_label.configure(text=message)

    def _download_complete(self, model_name: str) -> None:
        """Скачивание завершено."""
        self._downloading = False
        self.progress_frame.pack_forget()
        self._refresh_lists()

        messagebox.showinfo("Готово", f"Модель {model_name} успешно скачана!")

    def _download_error(self, error_msg: str) -> None:
        """Ошибка скачивания."""
        self._downloading = False
        self.progress_frame.pack_forget()

        messagebox.showerror("Ошибка", f"Ошибка скачивания:\n\n{error_msg}")

    def _delete_model(self, model_name: str) -> None:
        """Удалить модель."""
        if messagebox.askyesno(
            "Удаление модели",
            f"Удалить модель {model_name}?\n\nЭто освободит место на диске.",
        ):
            try:
                self.model_manager.delete_whisper_model(model_name)
                self._refresh_lists()
            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка удаления:\n\n{e}")


class ApiKeyDialog(ctk.CTkToplevel):
    """Диалог ввода Anthropic API ключа."""

    def __init__(self, parent, config: AppConfig):
        super().__init__(parent)

        self.config = config

        # Настройка окна
        self.title("API ключ Claude")
        self.geometry("450x200")
        self.resizable(False, False)

        # Создаём интерфейс
        self._create_widgets()

        # Центрируем относительно родителя
        self.transient(parent)
        self.update_idletasks()

        x = parent.winfo_x() + (parent.winfo_width() - self.winfo_width()) // 2
        y = parent.winfo_y() + (parent.winfo_height() - self.winfo_height()) // 2
        self.geometry(f"+{x}+{y}")

    def _create_widgets(self) -> None:
        """Создать виджеты."""
        main_frame = ctk.CTkFrame(self, fg_color="transparent")
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Описание
        ctk.CTkLabel(
            main_frame,
            text="Введите API ключ Anthropic для AI-анализа",
            font=("", 12),
        ).pack(anchor="w", pady=(0, 5))

        ctk.CTkLabel(
            main_frame,
            text="Получите ключ на console.anthropic.com",
            font=("", 10),
            text_color="gray",
        ).pack(anchor="w", pady=(0, 15))

        # Поле ввода
        self.api_key_entry = ctk.CTkEntry(
            main_frame,
            placeholder_text="sk-ant-...",
            show="*",
            width=400,
        )
        self.api_key_entry.pack(fill="x", pady=(0, 15))

        # Если ключ уже есть — показываем маскированный
        if self.config.anthropic_api_key:
            masked = self.config.anthropic_api_key[:10] + "..." + self.config.anthropic_api_key[-4:]
            self.api_key_entry.insert(0, masked)

        # Кнопки
        btn_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        btn_frame.pack(fill="x")

        ctk.CTkButton(
            btn_frame,
            text="Сохранить",
            command=self._save,
        ).pack(side="left", padx=(0, 10))

        ctk.CTkButton(
            btn_frame,
            text="Удалить ключ",
            fg_color="gray",
            command=self._clear,
        ).pack(side="left", padx=(0, 10))

        ctk.CTkButton(
            btn_frame,
            text="Отмена",
            fg_color="gray",
            command=self.destroy,
        ).pack(side="right")

    def _save(self) -> None:
        """Сохранить API ключ."""
        key = self.api_key_entry.get().strip()

        # Если ключ не изменился (маскированный) — просто закрываем
        if key.startswith("sk-ant-") and "..." in key:
            self.destroy()
            return

        # Валидация формата
        if key and not key.startswith("sk-ant-"):
            messagebox.showerror(
                "Ошибка",
                "API ключ должен начинаться с 'sk-ant-'"
            )
            return

        self.config.anthropic_api_key = key
        self.config.save()
        self.destroy()

    def _clear(self) -> None:
        """Удалить API ключ."""
        self.config.anthropic_api_key = ""
        self.config.enable_ai_analysis = False
        self.config.save()
        self.destroy()
