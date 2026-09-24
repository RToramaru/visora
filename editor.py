import os
import shutil
import cv2
import json
import threading
import xml.etree.ElementTree as ET
from xml.dom import minidom
import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

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
        self.midia_eh_video = False

        # Mídias/Amostras para Anotação
        self.lista_amostras = []
        self.amostra_index_atual = 0

        # Estado das Anotações
        self.annotations = []
        self.temp_shape_id = None
        self.start_x = None
        self.start_y = None

        # Específico para Polígonos (Segmentação)
        self.current_polygon_points = []
        self.polygon_line_ids = []

        self.build_ui()
        self.after(300, self.verificar_estado_inicial_projeto)

    def destruir_frame(self):
        self.parar_video()
        self.destroy()

    def build_ui(self):
        # ==================== 1. HEADER SUPERIOR ====================
        header = ctk.CTkFrame(self, height=36, fg_color="#0d1117", corner_radius=0)
        header.pack(fill="x", side="top")

        lbl_logo = ctk.CTkLabel(header, text="VISORA STUDIO", font=ctk.CTkFont(size=12, weight="bold"),
                                text_color="#2563eb")
        lbl_logo.pack(side="left", padx=15)

        topologia_atual = self.project_data.get('topologia', 'Bounding Boxes')
        lbl_ds_info = ctk.CTkLabel(
            header,
            text=f"Dataset:  {self.project_data['nome']}    |    Topologia:  {topologia_atual}",
            font=ctk.CTkFont(size=11), text_color="#64748b"
        )
        lbl_ds_info.pack(side="left", padx=10)

        # ==================== 2. STEPPER NAVIGATION ====================
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

        # ==================== 3. BARRA DE AÇÕES (SUB-HEADER) ====================
        self.action_bar = ctk.CTkFrame(self, height=45, fg_color="#0d1117", corner_radius=0)
        self.action_bar.pack(fill="x", side="top")

        self.render_action_bar_etapa1()

        # ==================== 4. CORPO PRINCIPAL ====================
        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True)

        # PAINEL LATERAL DIREITO
        self.right_panel = ctk.CTkScrollableFrame(body, width=300, fg_color="#0d1117", corner_radius=0)
        self.right_panel.pack(side="right", fill="y")

        self.build_right_panel()

        # ÁREA CENTRAL
        self.preview_area = ctk.CTkFrame(body, fg_color="#05070a", corner_radius=0)
        self.preview_area.pack(side="left", fill="both", expand=True)

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
            text="X: 0  Y: 0  1.00x",
            font=ctk.CTkFont(size=10),
            text_color="#475569"
        )
        self.lbl_coords.pack(side="right")

        # CONTROLES DO PLAYER DE VÍDEO
        self.video_controls_frame = ctk.CTkFrame(self.preview_area, fg_color="#0d1117", height=50, corner_radius=6)
        self.video_controls_frame.pack(side="bottom", fill="x", padx=10, pady=10)

        self.btn_rewind = ctk.CTkButton(self.video_controls_frame, text="⏪ -5s", width=50, height=28,
                                        fg_color="#1e293b", hover_color="#334155",
                                        command=lambda: self.seek_relative(-5))
        self.btn_rewind.pack(side="left", padx=(10, 4), pady=8)

        self.btn_play = ctk.CTkButton(self.video_controls_frame, text="▶ Play", width=65, height=28, fg_color="#2563eb",
                                      hover_color="#1d4ed8", command=self.toggle_play)
        self.btn_play.pack(side="left", padx=4, pady=8)

        self.btn_forward = ctk.CTkButton(self.video_controls_frame, text="+5s ⏩", width=50, height=28,
                                         fg_color="#1e293b", hover_color="#334155",
                                         command=lambda: self.seek_relative(5))
        self.btn_forward.pack(side="left", padx=(4, 10), pady=8)

        self.btn_speed = ctk.CTkButton(self.video_controls_frame, text="1.0x", width=45, height=28, fg_color="#1e293b",
                                       hover_color="#334155", command=self.alternar_velocidade)
        self.btn_speed.pack(side="left", padx=(0, 10), pady=8)

        self.slider_video = ctk.CTkSlider(self.video_controls_frame, from_=0, to=100, command=self.on_slider_move,
                                          height=14)
        self.slider_video.pack(side="left", fill="x", expand=True, padx=10, pady=8)

        self.lbl_time = ctk.CTkLabel(self.video_controls_frame, text="00:00 / 00:00", font=ctk.CTkFont(size=11),
                                     text_color="#94a3b8")
        self.lbl_time.pack(side="right", padx=15, pady=8)

        # DISPLAY CENTRAL
        self.display_container = ctk.CTkFrame(self.preview_area, fg_color="transparent")
        self.display_container.pack(fill="both", expand=True, padx=10, pady=5)

        self.lbl_preview = ctk.CTkLabel(self.display_container, text="Importe um vídeo ou fotos para visualizar.",
                                        text_color="#334155", font=ctk.CTkFont(size=13))
        self.lbl_preview.pack(fill="both", expand=True)

        self.canvas_anotacao = ctk.CTkCanvas(self.display_container, bg="#05070a", highlightthickness=0)
        self.vincular_eventos_canvas()

        # PAINEL OVERLAY DE CARREGAMENTO
        self.overlay_loading = ctk.CTkFrame(self.display_container, fg_color="#0b0e14", corner_radius=8)

        self.lbl_loading_msg = ctk.CTkLabel(
            self.overlay_loading,
            text="⏳ Processando e extraindo frames do vídeo...\nAguarde um momento.",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#38bdf8"
        )
        self.lbl_loading_msg.pack(padx=40, pady=30)

        self.progress_bar = ctk.CTkProgressBar(self.overlay_loading, width=300, mode="indeterminate",
                                               progress_color="#2563eb")
        self.progress_bar.pack(padx=40, pady=(0, 20))

    def vincular_eventos_canvas(self):
        self.canvas_anotacao.unbind("<ButtonPress-1>")
        self.canvas_anotacao.unbind("<B1-Motion>")
        self.canvas_anotacao.unbind("<ButtonRelease-1>")
        self.canvas_anotacao.unbind("<Double-Button-1>")
        self.canvas_anotacao.unbind("<Button-3>")

        topologia = self.project_data.get("topologia", "Bounding Boxes")
        if topologia == "Bounding Boxes":
            self.canvas_anotacao.bind("<ButtonPress-1>", self.on_canvas_click)
            self.canvas_anotacao.bind("<B1-Motion>", self.on_canvas_drag)
            self.canvas_anotacao.bind("<ButtonRelease-1>", self.on_canvas_release)
        else:
            self.canvas_anotacao.bind("<ButtonPress-1>", self.on_polygon_click)
            self.canvas_anotacao.bind("<Double-Button-1>", self.on_polygon_finish)
            self.canvas_anotacao.bind("<Button-3>", self.on_polygon_finish)

    def verificar_estado_inicial_projeto(self):
        caminho = self.project_data.get("caminho", "")
        if not caminho or not os.path.exists(caminho):
            return

        pasta_frames = os.path.join(caminho, "frames")
        if os.path.exists(pasta_frames) and os.listdir(pasta_frames):
            self.midia_eh_video = True
            self.mudar_passo(2)
            return

        arquivos = [f for f in os.listdir(caminho) if f.lower().endswith(EXTENSOES_IMAGEM)]
        if arquivos:
            self.midia_eh_video = False
            self.mudar_passo(2)
        else:
            self.carregar_midias_da_pasta()

    def mostrar_carregamento(self, mensagem="Processando..."):
        self.lbl_loading_msg.configure(text=f"⏳ {mensagem}")
        self.overlay_loading.place(relx=0.5, rely=0.5, anchor="center")
        self.progress_bar.start()
        self.update_idletasks()

    def esconder_carregamento(self):
        self.progress_bar.stop()
        self.overlay_loading.place_forget()
        self.update_idletasks()

    def render_action_bar_etapa1(self):
        for w in self.action_bar.winfo_children():
            w.destroy()

        lbl_imp = ctk.CTkLabel(self.action_bar, text="IMPORTAR:", font=ctk.CTkFont(size=10, weight="bold"),
                               text_color="#475569")
        lbl_imp.pack(side="left", padx=(15, 10))

        btn_imp_video = ctk.CTkButton(
            self.action_bar, text="🎥 Importar Vídeo", width=130, height=28,
            fg_color="#1e293b", hover_color="#334155", text_color="#cbd5e1",
            font=ctk.CTkFont(size=11, weight="bold"), command=self.importar_arquivo_video
        )
        btn_imp_video.pack(side="left", padx=4)

        btn_imp_fotos = ctk.CTkButton(
            self.action_bar, text="🖼️ Importar Fotos", width=130, height=28,
            fg_color="#1e293b", hover_color="#334155", text_color="#cbd5e1",
            font=ctk.CTkFont(size=11, weight="bold"), command=self.importar_arquivos_fotos
        )
        btn_imp_fotos.pack(side="left", padx=4)

        self.btn_next_step = ctk.CTkButton(
            self.action_bar, text="Próxima Etapa → (Anotar)", width=170, height=28,
            fg_color="#2563eb", hover_color="#1d4ed8", font=ctk.CTkFont(size=11, weight="bold"),
            command=lambda: self.mudar_passo(2)
        )
        self.btn_next_step.pack(side="right", padx=10)

    def render_action_bar_etapa2(self):
        for w in self.action_bar.winfo_children():
            w.destroy()

        topologia = self.project_data.get("topologia", "Bounding Boxes")
        lbl_top = ctk.CTkLabel(self.action_bar, text=f"MODO: {topologia.upper()}",
                               font=ctk.CTkFont(size=10, weight="bold"), text_color="#38bdf8")
        lbl_top.pack(side="left", padx=(15, 15))

        lbl_c = ctk.CTkLabel(self.action_bar, text="Classe:", font=ctk.CTkFont(size=11), text_color="#94a3b8")
        lbl_c.pack(side="left", padx=(0, 5))

        self.entry_classe = ctk.CTkEntry(self.action_bar, placeholder_text="Ex: Objeto", width=130, height=28,
                                         fg_color="#080a0f")
        self.entry_classe.insert(0, "Objeto")
        self.entry_classe.pack(side="left", padx=5)

        btn_limpar = ctk.CTkButton(self.action_bar, text="🧹 Limpar", width=80, height=28, fg_color="#ef4444",
                                   hover_color="#dc2626", command=self.limpar_anotacoes)
        btn_limpar.pack(side="left", padx=10)

        btn_salvar = ctk.CTkButton(self.action_bar, text="💾 Salvar Rótulos", width=110, height=28, fg_color="#2563eb",
                                   hover_color="#1d4ed8", command=lambda: self.salvar_anotacoes(silencioso=False))
        btn_salvar.pack(side="left", padx=5)

        btn_prox = ctk.CTkButton(self.action_bar, text="Próxima Imagem →", width=130, height=28, fg_color="#1e293b",
                                 hover_color="#334155", command=self.proxima_imagem)
        btn_prox.pack(side="right", padx=15)

        btn_ant = ctk.CTkButton(self.action_bar, text="← Anterior", width=90, height=28, fg_color="#1e293b",
                                hover_color="#334155", command=self.imagem_anterior)
        btn_ant.pack(side="right", padx=5)

    def mudar_passo(self, step_id):
        self.step_atual = step_id
        for idx, btn in self.steps_btn.items():
            if idx == step_id:
                btn.configure(fg_color="#1d283a", text_color="#38bdf8")
            else:
                btn.configure(fg_color="transparent", text_color="#64748b")

        if step_id == 1:
            self.lbl_preview.pack(fill="both", expand=True)
            self.canvas_anotacao.pack_forget()
            self.render_action_bar_etapa1()

            # Oculta propriedades na etapa 1 se não houver mídia selecionada
            if not self.arquivo_selecionado:
                self.esconder_painel_propriedades()
            else:
                self.atualizar_visibilidade_painel_direito()

            if self.arquivo_selecionado and self.arquivo_selecionado.lower().endswith(EXTENSOES_VIDEO):
                self.video_controls_frame.pack(side="bottom", fill="x", padx=10, pady=10)

        elif step_id == 2:
            self.parar_video()
            self.video_controls_frame.pack_forget()
            self.lbl_preview.pack_forget()
            self.canvas_anotacao.pack(fill="both", expand=True)
            self.render_action_bar_etapa2()
            self.vincular_eventos_canvas()

            # Ao entrar na etapa de anotação, oculta o painel lateral de propriedades/extração
            self.esconder_painel_propriedades()

            caminho_proj = self.project_data.get("caminho", "")
            pasta_frames = os.path.join(caminho_proj, "frames")

            if os.path.exists(pasta_frames) and os.listdir(pasta_frames):
                self.carregar_amostras_nao_rotuladas()
                return

            if self.arquivo_selecionado and self.arquivo_selecionado.lower().endswith(EXTENSOES_VIDEO):
                self.mostrar_carregamento("Extraindo e salvando frames na pasta do projeto...")
                threading.Thread(target=self._executar_preparacao_frames_video_threaded, daemon=True).start()
            else:
                self.preparar_amostras_fotos()

    def _executar_preparacao_frames_video_threaded(self):
        caminho_proj = self.project_data.get("caminho", "")
        if not caminho_proj:
            self.after(0, self.esconder_carregamento)
            return

        pasta_frames = os.path.join(caminho_proj, "frames")
        os.makedirs(pasta_frames, exist_ok=True)

        modo = self.opcao_extracao.get()
        step_frames = 1

        if modo == "n_frames":
            try:
                step_frames = max(1, int(self.entry_n_frames.get()))
            except ValueError:
                step_frames = 5
        elif modo == "n_segundos":
            try:
                segs = float(self.entry_n_sec.get())
                step_frames = max(1, int(segs * self.video_fps))
            except ValueError:
                step_frames = int(self.video_fps)

        saved_count = 0
        cap_temp = cv2.VideoCapture(self.arquivo_selecionado)
        current_f = 0

        while True:
            if step_frames > 1:
                cap_temp.set(cv2.CAP_PROP_POS_FRAMES, current_f)

            ret, frame = cap_temp.read()
            if not ret:
                break

            nome_frame = f"frame_{saved_count:05d}.jpg"
            path_frame = os.path.join(pasta_frames, nome_frame)

            if not os.path.exists(path_frame):
                cv2.imwrite(path_frame, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])

            saved_count += 1
            current_f += step_frames

            if step_frames == 1:
                continue
            if current_f >= self.total_frames:
                break

        if step_frames == 1:
            cap_temp.release()
            cap_temp = cv2.VideoCapture(self.arquivo_selecionado)
            saved_count = 0
            while True:
                ret, frame = cap_temp.read()
                if not ret:
                    break
                nome_frame = f"frame_{saved_count:05d}.jpg"
                path_frame = os.path.join(pasta_frames, nome_frame)
                if not os.path.exists(path_frame):
                    cv2.imwrite(path_frame, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                saved_count += 1

        cap_temp.release()

        def finalizar_carregamento():
            self.esconder_carregamento()
            self.carregar_amostras_nao_rotuladas()

        self.after(0, finalizar_carregamento)

    def carregar_amostras_nao_rotuladas(self):
        caminho_proj = self.project_data.get("caminho", "")
        pasta_frames = os.path.join(caminho_proj, "frames")
        pasta_annotations = os.path.join(caminho_proj, "annotations")
        os.makedirs(pasta_annotations, exist_ok=True)

        if not os.path.exists(pasta_frames):
            self.lista_amostras = []
            return

        topologia = self.project_data.get("topologia", "Bounding Boxes")
        ext_rotulo = ".xml" if topologia == "Bounding Boxes" else ".json"

        todos_frames = sorted(
            [os.path.join(pasta_frames, f) for f in os.listdir(pasta_frames) if f.lower().endswith(EXTENSOES_IMAGEM)])

        self.lista_amostras = []
        for f_path in todos_frames:
            nome_base = os.path.splitext(os.path.basename(f_path))[0]
            rotulo_path = os.path.join(pasta_annotations, f"{nome_base}{ext_rotulo}")
            if not os.path.exists(rotulo_path):
                self.lista_amostras.append(f_path)

        if self.lista_amostras:
            self.amostra_index_atual = 0
            self.carregar_imagem_no_canvas(self.lista_amostras[0])
        else:
            messagebox.showinfo("Concluído", "Todos os frames extraídos já foram rotulados!")

    def preparar_amostras_fotos(self):
        caminho_proj = self.project_data.get("caminho", "")
        if not caminho_proj:
            return

        pasta_annotations = os.path.join(caminho_proj, "annotations")
        os.makedirs(pasta_annotations, exist_ok=True)
        topologia = self.project_data.get("topologia", "Bounding Boxes")
        ext_rotulo = ".xml" if topologia == "Bounding Boxes" else ".json"

        todos = sorted(
            [os.path.join(caminho_proj, f) for f in os.listdir(caminho_proj) if f.lower().endswith(EXTENSOES_IMAGEM)])
        self.lista_amostras = []
        for f_path in todos:
            nome_base = os.path.splitext(os.path.basename(f_path))[0]
            if not os.path.exists(os.path.join(pasta_annotations, f"{nome_base}{ext_rotulo}")):
                self.lista_amostras.append(f_path)

        if self.lista_amostras:
            self.amostra_index_atual = 0
            self.carregar_imagem_no_canvas(self.lista_amostras[0])

    def carregar_imagem_no_canvas(self, filepath):
        if not filepath or not os.path.exists(filepath):
            return

        self.arquivo_selecionado = filepath
        nome_arq = os.path.basename(filepath)
        total_total = len(self.lista_amostras)
        pos = self.amostra_index_atual + 1
        self.lbl_amostra_tag.configure(text=f"🔴 ANOTANDO [{pos}/{total_total}]: #{nome_arq}")

        self.annotations.clear()
        self.current_polygon_points.clear()
        self.polygon_line_ids.clear()

        self.pil_canvas_img = Image.open(filepath)
        w_box, h_box = self.get_preview_dimensions()

        new_w, new_h = self.redimensionar_proporcional(self.pil_canvas_img.width, self.pil_canvas_img.height, w_box,
                                                       h_box)
        self.pil_canvas_img_resized = self.pil_canvas_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        self.tk_canvas_img = ImageTk.PhotoImage(self.pil_canvas_img_resized)
        self.canvas_anotacao.delete("all")
        self.canvas_anotacao.create_image(
            w_box // 2, h_box // 2, image=self.tk_canvas_img, anchor="center"
        )

        self.carregar_anotacoes_existentes(filepath)

    def on_canvas_click(self, event):
        self.start_x = event.x
        self.start_y = event.y
        self.temp_shape_id = self.canvas_anotacao.create_rectangle(
            self.start_x, self.start_y, event.x, event.y, outline="#38bdf8", width=2
        )

    def on_canvas_drag(self, event):
        if self.temp_shape_id:
            self.canvas_anotacao.coords(self.temp_shape_id, self.start_x, self.start_y, event.x, event.y)

    def on_canvas_release(self, event):
        if self.temp_shape_id:
            classe = self.entry_classe.get().strip() or "Objeto"

            x1 = min(self.start_x, event.x)
            y1 = min(self.start_y, event.y)
            x2 = max(self.start_x, event.x)
            y2 = max(self.start_y, event.y)

            w_box, h_box = self.get_preview_dimensions()
            orig_w, orig_h = self.pil_canvas_img.size
            new_w, new_h = self.redimensionar_proporcional(orig_w, orig_h, w_box, h_box)

            offset_x = (w_box - new_w) // 2
            offset_y = (h_box - new_h) // 2

            scale = orig_w / new_w
            real_xmin = max(0, int((x1 - offset_x) * scale))
            real_ymin = max(0, int((y1 - offset_y) * scale))
            real_xmax = min(orig_w, int((x2 - offset_x) * scale))
            real_ymax = min(orig_h, int((y2 - offset_y) * scale))

            self.annotations.append({
                "type": "bbox", "label": classe, "xmin": real_xmin, "ymin": real_ymin, "xmax": real_xmax,
                "ymax": real_ymax
            })

            self.canvas_anotacao.create_text(x1 + 5, y1 + 10, text=classe, fill="#38bdf8", anchor="w",
                                             font=("Arial", 9, "bold"))
            self.temp_shape_id = None
            self.salvar_anotacoes(silencioso=True)

    def on_polygon_click(self, event):
        x, y = event.x, event.y
        self.current_polygon_points.append((x, y))

        vid = self.canvas_anotacao.create_oval(x - 3, y - 3, x + 3, y + 3, fill="#38bdf8", outline="#ffffff")
        self.polygon_line_ids.append(vid)

        if len(self.current_polygon_points) > 1:
            p1 = self.current_polygon_points[-2]
            p2 = self.current_polygon_points[-1]
            lid = self.canvas_anotacao.create_line(p1[0], p1[1], p2[0], p2[1], fill="#38bdf8", width=2)
            self.polygon_line_ids.append(lid)

    def on_polygon_finish(self, event=None):
        if len(self.current_polygon_points) < 3:
            return

        classe = self.entry_classe.get().strip() or "Objeto"
        w_box, h_box = self.get_preview_dimensions()
        orig_w, orig_h = self.pil_canvas_img.size
        new_w, new_h = self.redimensionar_proporcional(orig_w, orig_h, w_box, h_box)

        offset_x = (w_box - new_w) // 2
        offset_y = (h_box - new_h) // 2
        scale = orig_w / new_w

        real_points = []
        for px, py in self.current_polygon_points:
            rx = max(0, min(orig_w, int((px - offset_x) * scale)))
            ry = max(0, min(orig_h, int((py - offset_y) * scale)))
            real_points.append([rx, ry])

        self.annotations.append({
            "type": "polygon", "label": classe, "points": real_points
        })

        p_first = self.current_polygon_points[0]
        p_last = self.current_polygon_points[-1]
        lid = self.canvas_anotacao.create_line(p_last[0], p_last[1], p_first[0], p_first[1], fill="#22c55e", width=2)
        self.polygon_line_ids.append(lid)

        self.current_polygon_points.clear()
        self.polygon_line_ids.clear()
        self.salvar_anotacoes(silencioso=True)

    def limpar_anotacoes(self):
        self.annotations.clear()
        self.current_polygon_points.clear()
        if self.arquivo_selecionado:
            caminho_proj = self.project_data.get("caminho", "")
            nome_base = os.path.splitext(os.path.basename(self.arquivo_selecionado))[0]
            for ext in [".xml", ".json"]:
                path_arq = os.path.join(caminho_proj, "annotations", f"{nome_base}{ext}")
                if os.path.exists(path_arq):
                    try:
                        os.remove(path_arq)
                    except Exception:
                        pass
            self.carregar_imagem_no_canvas(self.arquivo_selecionado)

    def salvar_anotacoes(self, silencioso=False):
        if not self.arquivo_selecionado:
            return

        caminho_proj = self.project_data.get("caminho", "")
        if not caminho_proj:
            return

        pasta_annotations = os.path.join(caminho_proj, "annotations")
        os.makedirs(pasta_annotations, exist_ok=True)

        nome_arquivo_img = os.path.basename(self.arquivo_selecionado)
        nome_base = os.path.splitext(nome_arquivo_img)[0]
        topologia = self.project_data.get("topologia", "Bounding Boxes")

        if not self.annotations:
            for ext in [".xml", ".json"]:
                p = os.path.join(pasta_annotations, f"{nome_base}{ext}")
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except Exception:
                        pass
            return

        largura_orig, altura_orig = self.pil_canvas_img.size

        if topologia == "Bounding Boxes":
            xml_path = os.path.join(pasta_annotations, f"{nome_base}.xml")
            annotation_node = ET.Element("annotation")

            ET.SubElement(annotation_node, "folder").text = os.path.basename(caminho_proj)
            ET.SubElement(annotation_node, "filename").text = nome_arquivo_img
            ET.SubElement(annotation_node, "path").text = os.path.abspath(self.arquivo_selecionado)

            size_node = ET.SubElement(annotation_node, "size")
            ET.SubElement(size_node, "width").text = str(largura_orig)
            ET.SubElement(size_node, "height").text = str(altura_orig)
            ET.SubElement(size_node, "depth").text = "3"

            for ann in self.annotations:
                if ann.get("type") == "bbox":
                    object_node = ET.SubElement(annotation_node, "object")
                    ET.SubElement(object_node, "name").text = ann["label"]
                    ET.SubElement(object_node, "pose").text = "Unspecified"
                    ET.SubElement(object_node, "truncated").text = "0"
                    ET.SubElement(object_node, "difficult").text = "0"

                    bndbox_node = ET.SubElement(object_node, "bndbox")
                    ET.SubElement(bndbox_node, "xmin").text = str(ann["xmin"])
                    ET.SubElement(bndbox_node, "ymin").text = str(ann["ymin"])
                    ET.SubElement(bndbox_node, "xmax").text = str(ann["xmax"])
                    ET.SubElement(bndbox_node, "ymax").text = str(ann["ymax"])

            xml_string = minidom.parseString(ET.tostring(annotation_node)).toprettyxml(indent="  ")
            with open(xml_path, "w", encoding="utf-8") as f:
                f.write(xml_string)

            if not silencioso:
                messagebox.showinfo("Salvo", f"Anotação salva em annotations/{nome_base}.xml")
        else:
            json_path = os.path.join(pasta_annotations, f"{nome_base}.json")
            data = {
                "version": "5.0.1", "flags": {}, "shapes": [],
                "imagePath": nome_arquivo_img, "imageData": None,
                "imageHeight": altura_orig, "imageWidth": largura_orig
            }

            for ann in self.annotations:
                if ann.get("type") == "polygon":
                    data["shapes"].append({
                        "label": ann["label"], "points": ann["points"],
                        "group_id": None, "shape_type": "polygon", "flags": {}
                    })

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            if not silencioso:
                messagebox.showinfo("Salvo", f"Máscara salva em annotations/{nome_base}.json")

    def carregar_anotacoes_existentes(self, filepath):
        caminho_proj = self.project_data.get("caminho", "")
        nome_base = os.path.splitext(os.path.basename(filepath))[0]
        topologia = self.project_data.get("topologia", "Bounding Boxes")

        w_box, h_box = self.get_preview_dimensions()
        orig_w, orig_h = self.pil_canvas_img.size
        new_w, new_h = self.redimensionar_proporcional(orig_w, orig_h, w_box, h_box)

        offset_x = (w_box - new_w) // 2
        offset_y = (h_box - new_h) // 2
        scale = new_w / orig_w

        if topologia == "Bounding Boxes":
            xml_path = os.path.join(caminho_proj, "annotations", f"{nome_base}.xml")
            if os.path.exists(xml_path):
                try:
                    tree = ET.parse(xml_path)
                    root = tree.getroot()
                    for obj in root.findall("object"):
                        label = obj.find("name").text
                        bndbox = obj.find("bndbox")
                        xmin = int(bndbox.find("xmin").text)
                        ymin = int(bndbox.find("ymin").text)
                        xmax = int(bndbox.find("xmax").text)
                        ymax = int(bndbox.find("ymax").text)

                        self.annotations.append({
                            "type": "bbox", "label": label, "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax
                        })

                        cx1 = int(xmin * scale) + offset_x
                        cy1 = int(ymin * scale) + offset_y
                        cx2 = int(xmax * scale) + offset_x
                        cy2 = int(ymax * scale) + offset_y

                        self.canvas_anotacao.create_rectangle(cx1, cy1, cx2, cy2, outline="#38bdf8", width=2)
                        self.canvas_anotacao.create_text(cx1 + 5, cy1 + 10, text=label, fill="#38bdf8", anchor="w",
                                                         font=("Arial", 9, "bold"))
                except Exception:
                    pass
        else:
            json_path = os.path.join(caminho_proj, "annotations", f"{nome_base}.json")
            if os.path.exists(json_path):
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        for shape in data.get("shapes", []):
                            label = shape.get("label")
                            points = shape.get("points", [])
                            self.annotations.append({"type": "polygon", "label": label, "points": points})

                            canvas_points = []
                            for rx, ry in points:
                                cx = int(rx * scale) + offset_x
                                cy = int(ry * scale) + offset_y
                                canvas_points.append((cx, cy))
                                self.canvas_anotacao.create_oval(cx - 3, cy - 3, cx + 3, cy + 3, fill="#22c55e",
                                                                 outline="#ffffff")

                            if len(canvas_points) > 1:
                                flat_pts = [coord for pt in canvas_points for coord in pt]
                                self.canvas_anotacao.create_polygon(flat_pts, outline="#22c55e", fill="", width=2)
                                self.canvas_anotacao.create_text(canvas_points[0][0] + 5, canvas_points[0][1] + 10,
                                                                 text=label, fill="#22c55e", anchor="w",
                                                                 font=("Arial", 9, "bold"))
                except Exception:
                    pass

    def proxima_imagem(self):
        if self.lista_amostras:
            self.salvar_anotacoes(silencioso=True)
            self.amostra_index_atual += 1
            if self.amostra_index_atual < len(self.lista_amostras):
                self.carregar_imagem_no_canvas(self.lista_amostras[self.amostra_index_atual])
            else:
                self.carregar_amostras_nao_rotuladas()

    def imagem_anterior(self):
        if self.lista_amostras and self.amostra_index_atual > 0:
            self.salvar_anotacoes(silencioso=True)
            self.amostra_index_atual -= 1
            self.carregar_imagem_no_canvas(self.lista_amostras[self.amostra_index_atual])

    def importar_arquivo_video(self):
        caminho_proj = self.garantir_diretorio_projeto()
        if not caminho_proj:
            return

        arquivo = filedialog.askopenfilename(
            title="Selecionar Vídeo",
            filetypes=[("Vídeos", "*.mp4 *.avi *.mov *.mkv *.webm"), ("Todos os arquivos", "*.*")]
        )
        if arquivo:
            nome_dest = os.path.basename(arquivo)
            dest = os.path.join(caminho_proj, nome_dest)
            if os.path.abspath(arquivo) != os.path.abspath(dest):
                shutil.copy2(arquivo, dest)

            self.midia_eh_video = True
            self.carregar_midias_da_pasta()
            self.selecionar_midia(dest, is_video=True)
            self.atualizar_visibilidade_painel_direito()

    def importar_arquivos_fotos(self):
        caminho_proj = self.garantir_diretorio_projeto()
        if not caminho_proj:
            return

        arquivos = filedialog.askopenfilenames(
            title="Selecionar Foto(s)",
            filetypes=[("Imagens", "*.png *.jpg *.jpeg *.bmp *.tiff *.webp"), ("Todos os arquivos", "*.*")]
        )
        if arquivos:
            ultimo = None
            for arq in arquivos:
                nome_dest = os.path.basename(arq)
                dest = os.path.join(caminho_proj, nome_dest)
                if os.path.abspath(arq) != os.path.abspath(dest):
                    shutil.copy2(arq, dest)
                ultimo = dest

            self.midia_eh_video = False
            self.carregar_midias_da_pasta()
            if ultimo:
                self.selecionar_midia(ultimo, is_video=False)
            self.atualizar_visibilidade_painel_direito()

    def garantir_diretorio_projeto(self):
        caminho_proj = self.project_data.get("caminho", "")
        if not caminho_proj or not os.path.exists(caminho_proj):
            caminho_proj = filedialog.askdirectory(title="Selecione a pasta do projeto")
            if caminho_proj:
                self.project_data["caminho"] = caminho_proj
        return caminho_proj

    def build_right_panel(self):
        # Container principal do painel direito
        self.right_container = ctk.CTkFrame(self.right_panel, fg_color="transparent")
        self.right_container.pack(fill="both", expand=True)

        self.lbl_prop_title = ctk.CTkLabel(
            self.right_container, text="⚙ PROPRIEDADES DA AMOSTRA",
            font=ctk.CTkFont(size=11, weight="bold"), text_color="#38bdf8", anchor="w"
        )
        self.lbl_prop_title.pack(fill="x", padx=10, pady=(15, 10))

        self.prop_frame = ctk.CTkFrame(self.right_container, fg_color="#111622", corner_radius=6)
        self.prop_frame.pack(fill="x", padx=10, pady=5)

        self.lbl_val_imagens = self.criar_linha_propriedade(self.prop_frame, "IMAGENS", "0")
        self.lbl_val_videos = self.criar_linha_propriedade(self.prop_frame, "VÍDEOS", "0")
        self.lbl_val_res = self.criar_linha_propriedade(self.prop_frame, "RESOLUÇÃO", "1920x1080px")
        self.lbl_val_fps = self.criar_linha_propriedade(self.prop_frame, "FPS", "30")
        self.lbl_val_duracao = self.criar_linha_propriedade(self.prop_frame, "DURAÇÃO", "0s")
        self.lbl_val_frames_totais = self.criar_linha_propriedade(self.prop_frame, "FRAMES TOTAIS", "0")

        # Bloco de Extração (Exclusivo para Vídeos)
        self.frame_extracao_container = ctk.CTkFrame(self.right_container, fg_color="transparent")
        self.frame_extracao_container.pack(fill="x", padx=0, pady=0)

        lbl_ext_title = ctk.CTkLabel(
            self.frame_extracao_container, text="EXTRAÇÃO DE FRAMES",
            font=ctk.CTkFont(size=11, weight="bold"), text_color="#94a3b8", anchor="w"
        )
        lbl_ext_title.pack(fill="x", padx=10, pady=(20, 10))

        self.opcao_extracao = ctk.StringVar(value="todos")

        self.card_opt_todos = ctk.CTkFrame(self.frame_extracao_container, fg_color="#111622", border_width=1,
                                           border_color="#1e293b", corner_radius=6)
        self.card_opt_todos.pack(fill="x", padx=10, pady=4)

        r1 = ctk.CTkRadioButton(self.card_opt_todos, text="Todos os frames", variable=self.opcao_extracao,
                                value="todos", font=ctk.CTkFont(size=11, weight="bold"),
                                command=self.atualizar_estilo_extracao)
        r1.pack(anchor="w", padx=12, pady=(10, 2))
        self.lbl_sub_todos = ctk.CTkLabel(self.card_opt_todos, text="0 frames brutos", font=ctk.CTkFont(size=10),
                                          text_color="#475569")
        self.lbl_sub_todos.pack(anchor="w", padx=32, pady=(0, 10))

        self.card_opt_nframes = ctk.CTkFrame(self.frame_extracao_container, fg_color="#111622", border_width=1,
                                             border_color="#1e293b", corner_radius=6)
        self.card_opt_nframes.pack(fill="x", padx=10, pady=4)

        r2 = ctk.CTkRadioButton(self.card_opt_nframes, text="A cada N frames", variable=self.opcao_extracao,
                                value="n_frames", font=ctk.CTkFont(size=11, weight="bold"),
                                command=self.atualizar_estilo_extracao)
        r2.pack(anchor="w", padx=12, pady=(10, 2))
        lbl_sub_nframes = ctk.CTkLabel(self.card_opt_nframes, text="Amostragem intervalada recomendada",
                                       font=ctk.CTkFont(size=10), text_color="#475569")
        lbl_sub_nframes.pack(anchor="w", padx=32, pady=(0, 5))

        self.box_input_nframes = ctk.CTkFrame(self.card_opt_nframes, fg_color="transparent")
        self.entry_n_frames = ctk.CTkEntry(self.box_input_nframes, placeholder_text="5", width=70, height=26,
                                           fg_color="#080a0f", border_color="#2563eb")
        self.entry_n_frames.insert(0, "5")
        self.entry_n_frames.pack(side="left", padx=(32, 5), pady=(0, 10))
        ctk.CTkLabel(self.box_input_nframes, text="frames", font=ctk.CTkFont(size=10), text_color="#64748b").pack(
            side="left", pady=(0, 10))

        self.card_opt_nsec = ctk.CTkFrame(self.frame_extracao_container, fg_color="#111622", border_width=1,
                                          border_color="#1e293b", corner_radius=6)
        self.card_opt_nsec.pack(fill="x", padx=10, pady=4)

        r3 = ctk.CTkRadioButton(self.card_opt_nsec, text="A cada N segundos", variable=self.opcao_extracao,
                                value="n_segundos", font=ctk.CTkFont(size=11, weight="bold"),
                                command=self.atualizar_estilo_extracao)
        r3.pack(anchor="w", padx=12, pady=(10, 2))
        lbl_sub_nsec = ctk.CTkLabel(self.card_opt_nsec, text="Frequência temporal constante", font=ctk.CTkFont(size=10),
                                    text_color="#475569")
        lbl_sub_nsec.pack(anchor="w", padx=32, pady=(0, 5))

        self.box_input_nsec = ctk.CTkFrame(self.card_opt_nsec, fg_color="transparent")
        self.entry_n_sec = ctk.CTkEntry(self.box_input_nsec, placeholder_text="1", width=70, height=26,
                                        fg_color="#080a0f", border_color="#2563eb")
        self.entry_n_sec.insert(0, "1")
        self.entry_n_sec.pack(side="left", padx=(32, 5), pady=(0, 10))
        ctk.CTkLabel(self.box_input_nsec, text="seg", font=ctk.CTkFont(size=10), text_color="#64748b").pack(side="left",
                                                                                                            pady=(0,
                                                                                                                  10))

        self.atualizar_estilo_extracao()

        # Inicialmente oculto até selecionar alguma mídia na etapa 1
        self.esconder_painel_propriedades()

    def esconder_painel_propriedades(self):
        self.right_container.pack_forget()

    def atualizar_visibilidade_painel_direito(self):
        # Mostra o painel direito apenas se estiver na Etapa 1 e houver mídia selecionada
        if self.step_atual == 1 and self.arquivo_selecionado:
            self.right_container.pack(fill="both", expand=True)

            # Controla se mostra o bloco de extração de frames (apenas para vídeos)
            if self.midia_eh_video:
                self.frame_extracao_container.pack(fill="x", padx=0, pady=0)
            else:
                self.frame_extracao_container.pack_forget()
        else:
            self.esconder_painel_propriedades()

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
            self.midia_eh_video = is_vid
            self.selecionar_midia(primeiro, is_vid)
            self.atualizar_visibilidade_painel_direito()

    def selecionar_midia(self, filepath, is_video):
        self.parar_video()
        self.arquivo_selecionado = filepath
        self.midia_eh_video = is_video
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

    def redimensionar_proporcional(self, largura_orig, altura_orig, largura_max, altura_max):
        proporcao = min(largura_max / largura_orig, altura_max / altura_orig)
        nova_largura = int(largura_orig * proporcao)
        nova_altura = int(altura_orig * proporcao)
        return nova_largura, nova_altura

    def exibir_imagem(self, filepath):
        try:
            pil_img = Image.open(filepath)
            w_box, h_box = self.get_preview_dimensions()
            self.lbl_val_res.configure(text=f"{pil_img.width}x{pil_img.height}px")
            self.lbl_val_fps.configure(text="N/A")
            self.lbl_val_duracao.configure(text="N/A")
            self.lbl_val_frames_totais.configure(text="1")
            self.lbl_sub_todos.configure(text="1 frame")

            new_w, new_h = self.redimensionar_proporcional(pil_img.width, pil_img.height, w_box, h_box)
            pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

            ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(new_w, new_h))
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
            new_w, new_h = self.redimensionar_proporcional(pil_img.width, pil_img.height, w_box, h_box)

            pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(new_w, new_h))

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