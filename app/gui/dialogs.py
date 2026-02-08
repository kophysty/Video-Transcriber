"""Диалоговые окна."""

import threading
import tkinter as tk
from tkinter import messagebox
from typing import Optional

import customtkinter as ctk

from app.utils.config import AppConfig
from app.models.model_manager import ModelManager
from app.models.model_registry import get_all_models, get_model_info, WHISPER_MODELS, PYANNOTE_MODEL_INFO
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
        self.geometry("600x700")
        self.minsize(500, 600)

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
        # Один общий скроллабл контейнер
        self.scroll_frame = ctk.CTkScrollableFrame(self)
        self.scroll_frame.pack(fill="both", expand=True, padx=15, pady=15)

        # === Скачанные модели Whisper ===
        ctk.CTkLabel(
            self.scroll_frame,
            text="Распознавание речи (Whisper) \u2014 скачанные",
            font=("", 14, "bold"),
        ).pack(anchor="w", pady=(0, 8))

        self.downloaded_frame = ctk.CTkFrame(self.scroll_frame)
        self.downloaded_frame.pack(fill="x", pady=(0, 15))

        # === Доступные модели Whisper ===
        ctk.CTkLabel(
            self.scroll_frame,
            text="Whisper \u2014 доступные для скачивания",
            font=("", 14, "bold"),
        ).pack(anchor="w", pady=(0, 8))

        self.available_frame = ctk.CTkFrame(self.scroll_frame)
        self.available_frame.pack(fill="x", pady=(0, 15))

        # === Диаризация (pyannote) ===
        ctk.CTkLabel(
            self.scroll_frame,
            text="Диаризация спикеров (pyannote)",
            font=("", 14, "bold"),
        ).pack(anchor="w", pady=(0, 8))

        self.pyannote_frame = ctk.CTkFrame(self.scroll_frame)
        self.pyannote_frame.pack(fill="x", pady=(0, 15))

        # === Прогресс скачивания ===
        self.progress_frame = ctk.CTkFrame(self.scroll_frame, fg_color="transparent")
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
            self.scroll_frame,
            text="",
            font=("", 11),
            text_color="gray",
        )
        self.size_label.pack(anchor="w", pady=(5, 0))

        # Заполняем списки
        self._refresh_lists()

    def _refresh_lists(self) -> None:
        """Обновить списки моделей."""
        if not self.winfo_exists():
            return

        # Очищаем
        for widget in self.downloaded_frame.winfo_children():
            widget.destroy()
        for widget in self.available_frame.winfo_children():
            widget.destroy()
        for widget in self.pyannote_frame.winfo_children():
            widget.destroy()

        downloaded = self.model_manager.list_downloaded_whisper_models()

        # Скачанные Whisper
        if downloaded:
            for model_name in downloaded:
                self._add_downloaded_model_row(model_name)
        else:
            ctk.CTkLabel(
                self.downloaded_frame,
                text="Нет скачанных моделей",
                text_color="gray",
            ).pack(pady=10)

        # Доступные Whisper
        available_count = 0
        for model_info in get_all_models():
            if model_info.name not in downloaded:
                self._add_available_model_row(model_info)
                available_count += 1

        if available_count == 0:
            ctk.CTkLabel(
                self.available_frame,
                text="Все модели скачаны",
                text_color="gray",
            ).pack(pady=10)

        # Pyannote
        self._refresh_pyannote_section()

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
            text=f"{model_info.description_ru} \u00b7 {format_file_size(size)}",
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
            text=f"{model_info.description_ru} \u00b7 ~{format_file_size(model_info.size_mb * 1024 * 1024)} \u00b7 VRAM: {model_info.vram_fp16_mb} MB",
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

    def _refresh_pyannote_section(self) -> None:
        """Обновить секцию pyannote."""
        is_downloaded = self.model_manager.is_pyannote_model_downloaded()

        info_frame = ctk.CTkFrame(self.pyannote_frame, fg_color="transparent")
        info_frame.pack(side="left", fill="x", expand=True, padx=10, pady=8)

        ctk.CTkLabel(
            info_frame,
            text=PYANNOTE_MODEL_INFO["name"],
            font=("", 12, "bold"),
        ).pack(anchor="w")

        if is_downloaded:
            ctk.CTkLabel(
                info_frame,
                text=f"{PYANNOTE_MODEL_INFO['description_ru']} \u00b7 Скачана",
                font=("", 10),
                text_color="gray",
            ).pack(anchor="w")

            delete_btn = ctk.CTkButton(
                self.pyannote_frame,
                text="Удалить",
                width=80,
                fg_color="gray",
                command=self._delete_pyannote,
            )
            delete_btn.pack(side="right", padx=10, pady=8)
        else:
            has_token = bool(self.config.hf_token)
            status_text = (
                f"{PYANNOTE_MODEL_INFO['description_ru']} \u00b7 "
                f"~{format_file_size(PYANNOTE_MODEL_INFO['size_mb'] * 1024 * 1024)}"
            )

            ctk.CTkLabel(
                info_frame,
                text=status_text,
                font=("", 10),
                text_color="gray",
            ).pack(anchor="w")

            if not has_token:
                ctk.CTkLabel(
                    info_frame,
                    text="Требуется HuggingFace токен (кнопка HF в главном окне)",
                    font=("", 9),
                    text_color="#CC6600",
                ).pack(anchor="w")

            download_btn = ctk.CTkButton(
                self.pyannote_frame,
                text="Скачать",
                width=80,
                state="normal" if has_token else "disabled",
                command=self._download_pyannote,
            )
            download_btn.pack(side="right", padx=10, pady=8)

    def _download_pyannote(self) -> None:
        """Скачать pyannote модель."""
        if self._downloading:
            messagebox.showwarning("Подождите", "Уже идёт скачивание")
            return

        if not self.config.hf_token:
            messagebox.showwarning(
                "Нужен токен",
                "Для скачивания pyannote нужен HuggingFace токен.\n\n"
                "Укажите его через кнопку 'HF токен' в главном окне."
            )
            return

        self._downloading = True
        self.progress_frame.pack(fill="x", pady=(0, 10))
        self.progress_bar.set(0)
        self.progress_label.configure(text="Скачивание pyannote...")

        thread = threading.Thread(
            target=self._download_pyannote_thread,
            daemon=True,
        )
        thread.start()

    def _download_pyannote_thread(self) -> None:
        """Поток скачивания pyannote."""
        try:
            from huggingface_hub import snapshot_download

            model_path = self.model_manager.pyannote_dir / "speaker-diarization-3.1"

            self.after(0, self._update_download_progress, 0.1, "Скачивание pyannote модели...")

            snapshot_download(
                repo_id="pyannote/speaker-diarization-3.1",
                local_dir=str(model_path),
                local_dir_use_symlinks=False,
                token=self.config.hf_token,
            )

            self.after(0, self._download_complete, "pyannote speaker-diarization-3.1")

        except Exception as e:
            self.after(0, self._download_error, str(e))

    def _delete_pyannote(self) -> None:
        """Удалить pyannote модель."""
        if messagebox.askyesno(
            "Удаление модели",
            "Удалить модель pyannote speaker-diarization-3.1?\n\n"
            "Диаризация спикеров не будет работать без этой модели.",
        ):
            try:
                import shutil
                model_path = self.model_manager.pyannote_dir / "speaker-diarization-3.1"
                if model_path.exists():
                    shutil.rmtree(model_path)
                self._refresh_lists()
            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка удаления:\n\n{e}")

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
        if not self.winfo_exists():
            return
        try:
            self.progress_bar.set(progress)
            self.progress_label.configure(text=message)
        except tk.TclError:
            pass

    def _download_complete(self, model_name: str) -> None:
        """Скачивание завершено."""
        self._downloading = False
        if not self.winfo_exists():
            return
        try:
            self.progress_frame.pack_forget()
            self._refresh_lists()
            messagebox.showinfo("Готово", f"Модель {model_name} успешно скачана!")
        except tk.TclError:
            pass

    def _download_error(self, error_msg: str) -> None:
        """Ошибка скачивания."""
        self._downloading = False
        if not self.winfo_exists():
            return
        try:
            self.progress_frame.pack_forget()
            messagebox.showerror("Ошибка", f"Ошибка скачивания:\n\n{error_msg}")
        except tk.TclError:
            pass

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


class HfTokenDialog(ctk.CTkToplevel):
    """Диалог ввода HuggingFace токена."""

    def __init__(self, parent, config: AppConfig):
        super().__init__(parent)

        self.config = config

        # Настройка окна
        self.title("HuggingFace токен")
        self.geometry("450x220")
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
            text="HuggingFace токен",
            font=("", 12),
        ).pack(anchor="w", pady=(0, 5))

        ctk.CTkLabel(
            main_frame,
            text="Нужен для скачивания модели диаризации (pyannote).\n"
                 "Получите бесплатно: huggingface.co/settings/tokens",
            font=("", 10),
            text_color="gray",
        ).pack(anchor="w", pady=(0, 15))

        # Поле ввода
        self.token_entry = ctk.CTkEntry(
            main_frame,
            placeholder_text="hf_...",
            show="*",
            width=400,
        )
        self.token_entry.pack(fill="x", pady=(0, 15))

        # Если токен уже есть — показываем маскированный
        if self.config.hf_token:
            masked = self.config.hf_token[:6] + "..." + self.config.hf_token[-4:]
            self.token_entry.insert(0, masked)

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
            text="Удалить токен",
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
        """Сохранить токен."""
        token = self.token_entry.get().strip()

        # Если токен не изменился (маскированный) — просто закрываем
        if "..." in token and len(token) < 20:
            self.destroy()
            return

        # Валидация формата
        if token and not token.startswith("hf_"):
            messagebox.showerror(
                "Ошибка",
                "HuggingFace токен должен начинаться с 'hf_'"
            )
            return

        self.config.hf_token = token
        self.config.save()
        self.destroy()

    def _clear(self) -> None:
        """Удалить токен."""
        self.config.hf_token = ""
        self.config.save()
        self.destroy()


class ApiKeyDialog(ctk.CTkToplevel):
    """Диалог ввода Anthropic API ключа (опционально, для web search)."""

    def __init__(self, parent, config: AppConfig):
        super().__init__(parent)

        self.config = config

        # Настройка окна
        self.title("API ключ (опционально)")
        self.geometry("450x220")
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
            text="Anthropic API ключ (опционально)",
            font=("", 12),
        ).pack(anchor="w", pady=(0, 5))

        ctk.CTkLabel(
            main_frame,
            text="Без ключа AI-анализ работает через Claude CLI.\n"
                 "С ключом \u2014 дополнительно ищет ссылки через веб-поиск.",
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
        self.config.save()
        self.destroy()
