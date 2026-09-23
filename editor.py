import os
import cv2
import customtkinter as ctk
from tkinter import filedialog
from PIL import Image

EXTENSOES_IMAGEM = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')
EXTENSOES_VIDEO = ('.mp4', '.avi', '.mov', '.mkv', '.webm')


class VisoraStudioFrame(ctk.CTkFrame):
    def __init__(self, parent, project_data=None, **kwargs):
        super().__init__(parent, fg_color="#080a0f", **kwargs)

        self.project_data = project_data or {
            "nome": "Thermal_Infrared_Anomaly_v2",
            "caminho": "",
            "topologia": "Bounding Boxes",
            "stream": "Sensor_Cam_01.raw"
        }

        # Estado do Player & Mídia
        self.cap = None
        self.is_playing = False
        self.video_fps = 30.0
        self.total_frames = 0
        self.current_frame = 0
        self.playback_speed = 1.0
        self.video_after_id = None
        self.arquivo_selecionado = None
        self.step_atual = 1

        self.build_ui()
        self.after(200, self.carregar_midias_da_pasta)

    def destruir_frame(self):
        self.parar_video()
        self.destroy()

    def build_ui(self):
        # ==================== 1. HEADER SUPERIOR ====================
        header = ctk.CTkFrame(self, height=36, fg_color="#0d1117", corner_radius=0)
        header.pack(fill="x", side="top")

        lbl_logo = ctk.CTkLabel(header, text="VISORA STUDIO", font=ctk.CTkFont(size=12, weight="bold"), text_color="#2563eb")
        lbl_logo.pack(side="left", padx=15)

        lbl_ds_info = ctk.CTkLabel(
            header,
            text=f"Dataset:  {self.project_data['nome']}    |    Stream:  {self.project_data.get('stream', 'Sensor_Cam_01.raw')}",
            font=ctk.CTkFont(size=11), text_color="#64748b"
        )
        lbl_ds_info.pack(side="left", padx=10)

        # ==================== 2. STEPPER NAVIGATION (CABEÇALHO) ====================
        stepper_bar = ctk.CTkFrame(self, height=42, fg_color="#0b0e14", corner_radius=0)
        stepper_bar.pack(fill="x", side="top")

        self.steps_btn = {}
        passos = [
            (1, "IMPORTAR"),
            (2, "SELECIONAR OBJETO"),
            (3, "PROPAGAR ANOTAÇÃO"),
            (4, "REVISAR"),
            (5, "EXPORTAR DATASET")
        ]

        steps_container = ctk.CTkFrame(stepper_bar, fg_color="transparent")
        steps_container.pack(side="left", padx=10, pady=5)

        for idx, text in passos:
            btn = ctk.CTkButton(
                steps_container,
                text=f"{idx}  {text}",
                font=ctk.CTkFont(size=11, weight="bold"),
                height=30,
                corner_radius=4,
                fg_color="#1d283a" if idx == 1 else "transparent",
                text_color="#38bdf8" if idx == 1 else "#64748b",
                hover_color="#1e293b",
                command=lambda i=idx: self.mudar_passo(i)
            )
            btn.pack(side="left", padx=3)
            self.steps_btn[idx] = btn

        # ==================== 3. BARRA DE AÇÕES (IMPORTAR PASTA) ====================
        action_bar = ctk.CTkFrame(self, height=45, fg_color="#0d1117", corner_radius=0)
        action_bar.pack(fill="x", side="top")

        lbl_imp = ctk.CTkLabel(action_bar, text="IMPORTAR", font=ctk.CTkFont(size=10, weight="bold"), text_color="#475569")
        lbl_imp.pack(side="left", padx=(15, 10))

        self.btn_pasta = ctk.CTkButton(
            action_bar,
            text="📁 Selecionar pasta",
            width=140, height=28,
            fg_color="#1e293b",
            hover_color="#334155",
            text_color="#cbd5e1",
            font=ctk.CTkFont(size=11, weight="bold"),
            command=self.selecionar_pasta_importacao
        )
        self.btn_pasta.pack(side="left", padx=5)

        self.btn_camera = ctk.CTkButton(action_bar, text="📷 Capturar câmera", width=130, height=28, fg_color="#1e293b", hover_color="#334155", text_color="#cbd5e1")
        self.btn_camera.pack(side="right", padx=15)

        # ==================== 4. CORPO PRINCIPAL ====================
        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True)

        # Visualizador Central
        self.preview_area = ctk.CTkFrame(body, fg_color="#05070a", corner_radius=0)
        self.preview_area.pack(side="left", fill="both", expand=True)

        # Barra Superior do Preview (Tag e Coordenadas)
        self.preview_top_bar = ctk.CTkFrame(self.preview_area, fg_color="transparent", height=30)
        self.preview_top_bar.pack(fill="x", side="top", padx=10, pady=5)

        self.lbl_amostra_tag = ctk.CTkLabel(
            self.preview_top_bar,
            text="🔴 AMOSTRA #SEM_ARQUIVO",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#ef4444",
            fg_color="#1f1519",
            corner_radius=4,
            padx=8, pady=2
        )
        self.lbl_amostra_tag.pack(side="left")

        self.lbl_coords = ctk.CTkLabel(
            self.preview_top_bar,
            text="X: 1920.04  Y: 1080.5012  1.00x",
            font=ctk.CTkFont(size=10),
            text_color="#475569"
        )
        self.lbl_coords.pack(side="right")

        # ==================== CONTROLES DE VÍDEO (RODAPÉ DO PREVIEW) ====================
        # Fixando no fundo antes da Label da imagem para não ser sobreposto
        self.video_controls_frame = ctk.CTkFrame(self.preview_area, fg_color="#0d1117", height=50, corner_radius=6)
        self.video_controls_frame.pack(side="bottom", fill="x", padx=10, pady=10)

        # Retroceder -5s
        self.btn_rewind = ctk.CTkButton(self.video_controls_frame, text="⏪ -5s", width=50, height=28, fg_color="#1e293b", hover_color="#334155", command=lambda: self.seek_relative(-5))
        self.btn_rewind.pack(side="left", padx=(10, 4), pady=8)

        # Play / Pause
        self.btn_play = ctk.CTkButton(self.video_controls_frame, text="▶ Play", width=65, height=28, fg_color="#2563eb", hover_color="#1d4ed8", command=self.toggle_play)
        self.btn_play.pack(side="left", padx=4, pady=8)

        # Avançar +5s
        self.btn_forward = ctk.CTkButton(self.video_controls_frame, text="+5s ⏩", width=50, height=28, fg_color="#1e293b", hover_color="#334155", command=lambda: self.seek_relative(5))
        self.btn_forward.pack(side="left", padx=(4, 10), pady=8)

        # Acelerador
        self.btn_speed = ctk.CTkButton(self.video_controls_frame, text="1.0x", width=45, height=28, fg_color="#1e293b", hover_color="#334155", command=self.alternar_velocidade)
        self.btn_speed.pack(side="left", padx=(0, 10), pady=8)

        # Slider da Linha do Tempo
        self.slider_video = ctk.CTkSlider(self.video_controls_frame, from_=0, to=100, command=self.on_slider_move, height=14)
        self.slider_video.pack(side="left", fill="x", expand=True, padx=10, pady=8)

        # Tempo decorrido / total
        self.lbl_time = ctk.CTkLabel(self.video_controls_frame, text="00:00 / 00:00", font=ctk.CTkFont(size=11), text_color="#94a3b8")
        self.lbl_time.pack(side="right", padx=15, pady=8)

        # Display Central da Imagem / Vídeo
        self.lbl_preview = ctk.CTkLabel(
            self.preview_area,
            text="Selecione uma pasta para carregar mídias.",
            text_color="#334155",
            font=ctk.CTkFont(size=13)
        )
        self.lbl_preview.pack(fill="both", expand=True, padx=10, pady=5)

        # ----- PAINEL LATERAL DIREITO -----
        self.right_panel = ctk.CTkScrollableFrame(body, width=320, fg_color="#0d1117", corner_radius=0)
        self.right_panel.pack(side="right", fill="y")

        self.build_right_panel()

    def build_right_panel(self):
        # 1. Seção Propriedades da Amostra
        lbl_prop_title = ctk.CTkLabel(
            self.right_panel, text="⚙ PROPRIEDADES DA AMOSTRA",
            font=ctk.CTkFont(size=11, weight="bold"), text_color="#38bdf8", anchor="w"
        )
        lbl_prop_title.pack(fill="x", padx=10, pady=(15, 10))

        prop_frame = ctk.CTkFrame(self.right_panel, fg_color="#111622", corner_radius=6)
        prop_frame.pack(fill="x", padx=10, pady=5)

        self.lbl_val_imagens = self.criar_linha_propriedade(prop_frame, "IMAGENS", "0")
        self.lbl_val_videos = self.criar_linha_propriedade(prop_frame, "VÍDEOS", "0")
        self.lbl_val_res = self.criar_linha_propriedade(prop_frame, "RESOLUÇÃO", "1920x1080px")
        self.lbl_val_fps = self.criar_linha_propriedade(prop_frame, "FPS", "30")
        self.lbl_val_duracao = self.criar_linha_propriedade(prop_frame, "DURAÇÃO", "0s")
        self.lbl_val_frames_totais = self.criar_linha_propriedade(prop_frame, "FRAMES TOTAIS", "0")

        # 2. Seção Extração de Frames
        lbl_ext_title = ctk.CTkLabel(
            self.right_panel, text="EXTRAÇÃO DE FRAMES",
            font=ctk.CTkFont(size=11, weight="bold"), text_color="#94a3b8", anchor="w"
        )
        lbl_ext_title.pack(fill="x", padx=10, pady=(20, 10))

        self.opcao_extracao = ctk.StringVar(value="todos")

        # Frame Opção 1: Todos os frames
        self.card_opt_todos = ctk.CTkFrame(self.right_panel, fg_color="#111622", border_width=1, border_color="#1e293b", corner_radius=6)
        self.card_opt_todos.pack(fill="x", padx=10, pady=4)

        r1 = ctk.CTkRadioButton(self.card_opt_todos, text="Todos os frames", variable=self.opcao_extracao, value="todos", font=ctk.CTkFont(size=11, weight="bold"), command=self.atualizar_estilo_extracao)
        r1.pack(anchor="w", padx=12, pady=(10, 2))
        self.lbl_sub_todos = ctk.CTkLabel(self.card_opt_todos, text="0 frames brutos", font=ctk.CTkFont(size=10), text_color="#475569")
        self.lbl_sub_todos.pack(anchor="w", padx=32, pady=(0, 10))

        # Frame Opção 2: A cada N frames
        self.card_opt_nframes = ctk.CTkFrame(self.right_panel, fg_color="#111622", border_width=1, border_color="#1e293b", corner_radius=6)
        self.card_opt_nframes.pack(fill="x", padx=10, pady=4)

        r2 = ctk.CTkRadioButton(self.card_opt_nframes, text="A cada N frames", variable=self.opcao_extracao, value="n_frames", font=ctk.CTkFont(size=11, weight="bold"), command=self.atualizar_estilo_extracao)
        r2.pack(anchor="w", padx=12, pady=(10, 2))
        lbl_sub_nframes = ctk.CTkLabel(self.card_opt_nframes, text="Amostragem intervalada recomendada", font=ctk.CTkFont(size=10), text_color="#475569")
        lbl_sub_nframes.pack(anchor="w", padx=32, pady=(0, 5))

        self.box_input_nframes = ctk.CTkFrame(self.card_opt_nframes, fg_color="transparent")
        self.entry_n_frames = ctk.CTkEntry(self.box_input_nframes, placeholder_text="5", width=70, height=26, fg_color="#080a0f", border_color="#2563eb")
        self.entry_n_frames.insert(0, "5")
        self.entry_n_frames.pack(side="left", padx=(32, 5), pady=(0, 10))
        ctk.CTkLabel(self.box_input_nframes, text="frames", font=ctk.CTkFont(size=10), text_color="#64748b").pack(side="left", pady=(0, 10))

        # Frame Opção 3: A cada N segundos
        self.card_opt_nsec = ctk.CTkFrame(self.right_panel, fg_color="#111622", border_width=1, border_color="#1e293b", corner_radius=6)
        self.card_opt_nsec.pack(fill="x", padx=10, pady=4)

        r3 = ctk.CTkRadioButton(self.card_opt_nsec, text="A cada N segundos", variable=self.opcao_extracao, value="n_segundos", font=ctk.CTkFont(size=11, weight="bold"), command=self.atualizar_estilo_extracao)
        r3.pack(anchor="w", padx=12, pady=(10, 2))
        lbl_sub_nsec = ctk.CTkLabel(self.card_opt_nsec, text="Frequência temporal constante", font=ctk.CTkFont(size=10), text_color="#475569")
        lbl_sub_nsec.pack(anchor="w", padx=32, pady=(0, 5))

        self.box_input_nsec = ctk.CTkFrame(self.card_opt_nsec, fg_color="transparent")
        self.entry_n_sec = ctk.CTkEntry(self.box_input_nsec, placeholder_text="1", width=70, height=26, fg_color="#080a0f", border_color="#2563eb")
        self.entry_n_sec.insert(0, "1")
        self.entry_n_sec.pack(side="left", padx=(32, 5), pady=(0, 10))
        ctk.CTkLabel(self.box_input_nsec, text="seg", font=ctk.CTkFont(size=10), text_color="#64748b").pack(side="left", pady=(0, 10))

        self.atualizar_estilo_extracao()

    def atualizar_estilo_extracao(self):
        v = self.opcao_extracao.get()

        self.card_opt_todos.configure(fg_color="#111622", border_color="#1e293b")
        self.card_opt_nframes.configure(fg_color="#111622", border_color="#1e293b")
        self.card_opt_nsec.configure(fg_color="#111622", border_color="#1e293b")

        self.box_input_nframes.pack_forget()
        self.box_input_nsec.pack_forget()

        if v == "todos":
            self.card_opt_todos.configure(fg_color="#162032", border_color="#2563eb")
        elif v == "n_frames":
            self.card_opt_nframes.configure(fg_color="#162032", border_color="#2563eb")
            self.box_input_nframes.pack(fill="x", anchor="w")
        elif v == "n_segundos":
            self.card_opt_nsec.configure(fg_color="#162032", border_color="#2563eb")
            self.box_input_nsec.pack(fill="x", anchor="w")

    def criar_linha_propriedade(self, parent, chave, valor_padrao):
        f = ctk.CTkFrame(parent, fg_color="transparent")
        f.pack(fill="x", padx=12, pady=4)

        lbl_c = ctk.CTkLabel(f, text=chave, font=ctk.CTkFont(size=10, weight="bold"), text_color="#64748b")
        lbl_c.pack(side="left")

        lbl_v = ctk.CTkLabel(f, text=valor_padrao, font=ctk.CTkFont(size=11), text_color="#e2e8f0")
        lbl_v.pack(side="right")
        return lbl_v

    def mudar_passo(self, step_id):
        self.step_atual = step_id
        for idx, btn in self.steps_btn.items():
            if idx == step_id:
                btn.configure(fg_color="#1d283a", text_color="#38bdf8")
            else:
                btn.configure(fg_color="transparent", text_color="#64748b")

    def selecionar_pasta_importacao(self):
        caminho = filedialog.askdirectory(title="Selecionar Pasta para Importar")
        if caminho:
            self.project_data["caminho"] = caminho
            self.carregar_midias_da_pasta()

    def carregar_midias_da_pasta(self):
        caminho = self.project_data.get("caminho", "")
        if not caminho or not os.path.exists(caminho):
            return

        arquivos = [f for f in os.listdir(caminho) if f.lower().endswith(EXTENSOES_IMAGEM + EXTENSOES_VIDEO)]
        imgs = [f for f in arquivos if f.lower().endswith(EXTENSOES_IMAGEM)]
        vids = [f for f in arquivos if f.lower().endswith(EXTENSOES_VIDEO)]

        self.lbl_val_imagens.configure(text=str(len(imgs)))
        self.lbl_val_videos.configure(text=str(len(vids)))

        if arquivos:
            primeiro = os.path.join(caminho, sorted(arquivos)[0])
            is_vid = primeiro.lower().endswith(EXTENSOES_VIDEO)
            self.selecionar_midia(primeiro, is_vid)

    def selecionar_midia(self, filepath, is_video):
        self.parar_video()
        self.arquivo_selecionado = filepath
        nome_arq = os.path.basename(filepath)
        self.lbl_amostra_tag.configure(text=f"🔴 AMOSTRA #{nome_arq}")

        if is_video:
            self.carregar_video(filepath)
        else:
            self.exibir_imagem(filepath)

    def get_preview_dimensions(self):
        self.update_idletasks()
        w = self.preview_area.winfo_width()
        h = self.preview_area.winfo_height()
        return max(w - 30, 300), max(h - 100, 300)

    def exibir_imagem(self, filepath):
        try:
            pil_img = Image.open(filepath)
            w_box, h_box = self.get_preview_dimensions()
            self.lbl_val_res.configure(text=f"{pil_img.width}x{pil_img.height}px")
            self.lbl_val_fps.configure(text="N/A")
            self.lbl_val_duracao.configure(text="N/A")
            self.lbl_val_frames_totais.configure(text="1")
            self.lbl_sub_todos.configure(text="1 frame")

            pil_img.thumbnail((w_box, h_box), Image.Resampling.LANCZOS)
            ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=pil_img.size)
            self.lbl_preview.configure(image=ctk_img, text="")
            self.lbl_preview.image = ctk_img
        except Exception as e:
            self.lbl_preview.configure(image="", text=f"Erro ao carregar imagem:\n{e}")

    def carregar_video(self, filepath):
        if self.cap:
            self.cap.release()

        self.cap = cv2.VideoCapture(filepath)
        if not self.cap.isOpened():
            self.lbl_preview.configure(text="Erro ao abrir arquivo de vídeo.")
            return

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.current_frame = 0

        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duracao_sec = int(self.total_frames / self.video_fps) if self.video_fps else 0

        self.lbl_val_res.configure(text=f"{width}x{height}px")
        self.lbl_val_fps.configure(text=f"{int(self.video_fps)}")
        self.lbl_val_duracao.configure(text=f"{duracao_sec}s")
        self.lbl_val_frames_totais.configure(text=f"{self.total_frames:,}".replace(",", "."))
        self.lbl_sub_todos.configure(text=f"{self.total_frames:,} frames brutos".replace(",", "."))

        self.slider_video.configure(from_=0, to=max(1, self.total_frames - 1))
        self.slider_video.set(0)

        self.atualizar_frame_video()
        self.atualizar_label_tempo()

    def alternar_velocidade(self):
        speeds = [1.0, 1.5, 2.0, 0.5]
        idx = (speeds.index(self.playback_speed) + 1) % len(speeds)
        self.playback_speed = speeds[idx]
        self.btn_speed.configure(text=f"{self.playback_speed}x")

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

        delay = int((1000 / self.video_fps) / self.playback_speed)
        self.video_after_id = self.after(delay, self.play_loop)

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

    def atualizar_frame_video(self):
        if not self.cap:
            return
        ret, frame = self.cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)

            w_box, h_box = self.get_preview_dimensions()
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

    def atualizar_label_tempo(self):
        pos_sec = int(self.current_frame / self.video_fps) if self.video_fps else 0
        tot_sec = int(self.total_frames / self.video_fps) if self.video_fps else 0
        self.lbl_time.configure(text=f"{pos_sec // 60:02d}:{pos_sec % 60:02d} / {tot_sec // 60:02d}:{tot_sec % 60:02d}")

    def parar_video(self):
        self.is_playing = False
        if self.video_after_id:
            try:
                self.after_cancel(self.video_after_id)
            except Exception:
                pass
            self.video_after_id = None
        if self.cap:
            self.cap.release()
            self.cap = None
        self.btn_play.configure(text="▶ Play", fg_color="#2563eb")