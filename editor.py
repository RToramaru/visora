import os
import cv2
import customtkinter as ctk
from PIL import Image, ImageTk

EXTENSOES_IMAGEM = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')
EXTENSOES_VIDEO = ('.mp4', '.avi', '.mov', '.mkv', '.webm')


class WorkspaceStudio(ctk.CTk):
    def __init__(self, project_data=None):
        super().__init__()

        self.project_data = project_data or {
            "nome": "Projeto Sem Nome",
            "caminho": "",
            "topologia": "Bounding Boxes"
        }

        self.title(f"Visora Studio - {self.project_data['nome']}")
        self.geometry("1200x750")
        self.minsize(1000, 600)
        self.configure(fg_color="#0b0e14")

        # Variáveis do Reprodutor de Vídeo
        self.cap = None
        self.is_playing = False
        self.video_fps = 30
        self.total_frames = 0
        self.current_frame = 0
        self.video_after_id = None
        self.is_seeking = False
        self.arquivo_selecionado = None

        self.build_ui()
        self.carregar_midias_da_pasta()

    def build_ui(self):
        # Header Superior
        header = ctk.CTkFrame(self, height=45, fg_color="#11151c", corner_radius=0)
        header.pack(fill="x", side="top")

        lbl_logo = ctk.CTkLabel(header, text="SPECTRA // STUDIO", font=ctk.CTkFont(size=12, weight="bold"),
                                text_color="#2563eb")
        lbl_logo.pack(side="left", padx=15)

        lbl_proj_info = ctk.CTkLabel(
            header,
            text=f"Projeto: {self.project_data['nome']}  |  Modo: {self.project_data['topologia']}",
            font=ctk.CTkFont(size=11), text_color="#94a3b8"
        )
        lbl_proj_info.pack(side="left", padx=10)

        # Corpo Principal
        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True)

        # Barra Lateral Esquerda (Arquivos)
        self.sidebar = ctk.CTkFrame(body, width=250, fg_color="#11151c", corner_radius=0)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        lbl_files_head = ctk.CTkLabel(self.sidebar, text="ARQUIVOS DE MÍDIA", font=ctk.CTkFont(size=11, weight="bold"),
                                      text_color="#475569")
        lbl_files_head.pack(anchor="w", padx=12, pady=(15, 5))

        self.scroll_files = ctk.CTkScrollableFrame(self.sidebar, fg_color="transparent")
        self.scroll_files.pack(fill="both", expand=True, padx=5, pady=5)

        # Área Central de Preview
        self.preview_area = ctk.CTkFrame(body, fg_color="#07090e", corner_radius=0)
        self.preview_area.pack(side="left", fill="both", expand=True)

        # Canvas/Label de Mídia
        self.lbl_preview = ctk.CTkLabel(self.preview_area, text="Selecione um arquivo da lista para visualizar",
                                        text_color="#475569", font=ctk.CTkFont(size=14))
        self.lbl_preview.pack(fill="both", expand=True, padx=20, pady=20)

        # Container dos Controles de Vídeo (Barra Inferior)
        self.video_controls_frame = ctk.CTkFrame(self.preview_area, fg_color="#11151c", height=50, corner_radius=6)
        # Inicia oculto, será exibido apenas quando um vídeo for selecionado

        # Botão Voltar 5s
        self.btn_rewind = ctk.CTkButton(self.video_controls_frame, text="⏪ -5s", width=50, fg_color="#1e293b",
                                        hover_color="#334155", command=lambda: self.seek_relative(-5))
        self.btn_rewind.pack(side="left", padx=(10, 5), pady=8)

        # Botão Play/Pause
        self.btn_play = ctk.CTkButton(self.video_controls_frame, text="▶ Play", width=60, fg_color="#2563eb",
                                      hover_color="#1d4ed8", command=self.toggle_play)
        self.btn_play.pack(side="left", padx=5, pady=8)

        # Botão Adiantar 5s
        self.btn_forward = ctk.CTkButton(self.video_controls_frame, text="+5s ⏩", width=50, fg_color="#1e293b",
                                         hover_color="#334155", command=lambda: self.seek_relative(5))
        self.btn_forward.pack(side="left", padx=5, pady=8)

        # Slider do Vídeo
        self.slider_video = ctk.CTkSlider(self.video_controls_frame, from_=0, to=100, command=self.on_slider_move)
        self.slider_video.pack(side="left", fill="x", expand=True, padx=10, pady=8)

        # Contador de Tempo
        self.lbl_time = ctk.CTkLabel(self.video_controls_frame, text="00:00 / 00:00", font=ctk.CTkFont(size=11),
                                     text_color="#94a3b8")
        self.lbl_time.pack(side="right", padx=(5, 15), pady=8)

    # ==========================================
    # CARREGAMENTO E MÍDIAS
    # ==========================================
    def carregar_midias_da_pasta(self):
        caminho = self.project_data.get("caminho", "")
        for w in self.scroll_files.winfo_children():
            w.destroy()

        if not caminho or not os.path.exists(caminho):
            ctk.CTkLabel(self.scroll_files, text="Diretório não encontrado.", font=ctk.CTkFont(size=11),
                         text_color="#64748b").pack(pady=20)
            return

        arquivos = [f for f in os.listdir(caminho) if f.lower().endswith(EXTENSOES_IMAGEM + EXTENSOES_VIDEO)]

        if not arquivos:
            ctk.CTkLabel(self.scroll_files, text="Nenhuma imagem ou vídeo na pasta.", font=ctk.CTkFont(size=10),
                         text_color="#64748b").pack(pady=20)
            return

        for arq in sorted(arquivos):
            path_completo = os.path.join(caminho, arq)
            is_video = arq.lower().endswith(EXTENSOES_VIDEO)
            icon = "🎥" if is_video else "🖼️"

            btn_file = ctk.CTkButton(
                self.scroll_files,
                text=f"{icon} {arq}",
                anchor="w",
                fg_color="transparent",
                text_color="#cbd5e1",
                hover_color="#1e293b",
                height=30,
                font=ctk.CTkFont(size=11),
                command=lambda p=path_completo, v=is_video: self.selecionar_midia(p, v)
            )
            btn_file.pack(fill="x", pady=1)

    def selecionar_midia(self, filepath, is_video):
        self.parar_video()
        self.arquivo_selecionado = filepath

        if is_video:
            self.video_controls_frame.pack(side="bottom", fill="x", padx=15, pady=10)
            self.carregar_video(filepath)
        else:
            self.video_controls_frame.pack_forget()
            self.exibir_imagem(filepath)

    # ==========================================
    # EXIBIÇÃO DE IMAGEM
    # ==========================================
    def exibir_imagem(self, filepath):
        try:
            pil_img = Image.open(filepath)

            # Redimensiona mantendo a proporção
            w_box = max(self.preview_area.winfo_width() - 40, 400)
            h_box = max(self.preview_area.winfo_height() - 40, 400)

            pil_img.thumbnail((w_box, h_box), Image.Resampling.LANCZOS)
            ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=pil_img.size)

            self.lbl_preview.configure(image=ctk_img, text="")
            self.lbl_preview.image = ctk_img
        except Exception as e:
            self.lbl_preview.configure(image="", text=f"Erro ao carregar imagem: {e}")

    # ==========================================
    # EXIBIÇÃO E CONTROLE DE VÍDEO (OPENCV)
    # ==========================================
    def carregar_video(self, filepath):
        if self.cap:
            self.cap.release()

        self.cap = cv2.VideoCapture(filepath)
        if not self.cap.isOpened():
            self.lbl_preview.configure(text="Erro ao carregar o vídeo.")
            return

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.current_frame = 0

        self.slider_video.configure(from_=0, to=self.total_frames - 1)
        self.slider_video.set(0)

        self.atualizar_frame_video()
        self.atualizar_label_tempo()

    def toggle_play(self):
        if not self.cap:
            return

        self.is_playing = not self.is_playing
        if self.is_playing:
            self.btn_play.configure(text="⏸ Pause", fg_color="#e11d48")
            self.play_loop()
        else:
            self.btn_play.configure(text="▶ Play", fg_color="#2563eb")

    def play_loop(self):
        if not self.is_playing or not self.cap:
            return

        if self.current_frame >= self.total_frames - 1:
            self.is_playing = False
            self.btn_play.configure(text="▶ Play", fg_color="#2563eb")
            return

        self.current_frame += 1
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        self.atualizar_frame_video()

        self.slider_video.set(self.current_frame)
        self.atualizar_label_tempo()

        delay = int(1000 / self.video_fps)
        self.video_after_id = self.after(delay, self.play_loop)

    def atualizar_frame_video(self):
        if not self.cap:
            return

        ret, frame = self.cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)

            w_box = max(self.preview_area.winfo_width() - 40, 400)
            h_box = max(self.preview_area.winfo_height() - 100, 300)

            pil_img.thumbnail((w_box, h_box), Image.Resampling.LANCZOS)
            ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=pil_img.size)

            self.lbl_preview.configure(image=ctk_img, text="")
            self.lbl_preview.image = ctk_img

    def on_slider_move(self, value):
        if not self.cap:
            return

        self.current_frame = int(value)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        self.atualizar_frame_video()
        self.atualizar_label_tempo()

    def seek_relative(self, seconds):
        if not self.cap:
            return

        frames_to_jump = int(seconds * self.video_fps)
        target_frame = max(0, min(self.total_frames - 1, self.current_frame + frames_to_jump))

        self.current_frame = target_frame
        self.slider_video.set(self.current_frame)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        self.atualizar_frame_video()
        self.atualizar_label_tempo()

    def atualizar_label_tempo(self):
        pos_sec = int(self.current_frame / self.video_fps) if self.video_fps else 0
        tot_sec = int(self.total_frames / self.video_fps) if self.video_fps else 0

        str_pos = f"{pos_sec // 60:02d}:{pos_sec % 60:02d}"
        str_tot = f"{tot_sec // 60:02d}:{tot_sec % 60:02d}"

        self.lbl_time.configure(text=f"{str_pos} / {str_tot}")

    def parar_video(self):
        self.is_playing = False
        if self.video_after_id:
            self.after_cancel(self.video_after_id)
            self.video_after_id = None
        if self.cap:
            self.cap.release()
            self.cap = None
        self.btn_play.configure(text="▶ Play", fg_color="#2563eb")


if __name__ == "__main__":
    app = WorkspaceStudio()
    app.mainloop()