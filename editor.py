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

        # PAINEL DE PROPAGAÇÃO (Etapa 3)
        self.frame_propagacao = ctk.CTkFrame(self.display_container, fg_color="#0b0e14", corner_radius=8)
        self.build_ui_propagacao()

        # PAINEL DE REVISAR (Etapa 4)
        self.frame_revisar = ctk.CTkFrame(self.display_container, fg_color="#0b0e14", corner_radius=8)
        self.build_ui_revisar()

        # PAINEL DE EXPORTAR DATASET (Etapa 5)
        self.frame_exportar = ctk.CTkFrame(self.display_container, fg_color="#0b0e14", corner_radius=8)
        self.build_ui_exportar()

        # PAINEL OVERLAY DE CARREGAMENTO
        self.overlay_loading = ctk.CTkFrame(self.display_container, fg_color="#0b0e14", corner_radius=8)
        self.lbl_loading_msg = ctk.CTkLabel(
            self.overlay_loading, text="⏳ Processando...", font=ctk.CTkFont(size=14, weight="bold"),
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
            self.action_bar, text="Próxima Etapa → (Anotar 1ª Imagem)", width=210, height=28,
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

        lbl_c = ctk.CTkLabel(self.action_bar, text="Classe(s):", font=ctk.CTkFont(size=11), text_color="#94a3b8")
        lbl_c.pack(side="left", padx=(0, 5))

        self.entry_classe = ctk.CTkEntry(self.action_bar, placeholder_text="Ex: Classe1, Classe2", width=160, height=28,
                                         fg_color="#080a0f")
        self.entry_classe.insert(0, "Objeto")
        self.entry_classe.pack(side="left", padx=5)

        btn_limpar = ctk.CTkButton(self.action_bar, text="🧹 Limpar", width=80, height=28, fg_color="#ef4444",
                                   hover_color="#dc2626", command=self.limpar_anotacoes)
        btn_limpar.pack(side="left", padx=10)

        btn_avancar = ctk.CTkButton(
            self.action_bar, text="Avançar para Propagar →", width=170, height=28,
            fg_color="#2563eb", hover_color="#1d4ed8", font=ctk.CTkFont(size=11, weight="bold"),
            command=lambda: self.mudar_passo(3)
        )
        btn_avancar.pack(side="right", padx=15)

    def render_action_bar_etapa3(self):
        for w in self.action_bar.winfo_children():
            w.destroy()
        lbl = ctk.CTkLabel(self.action_bar, text="⚡ PROPAGAÇÃO AUTOMÁTICA POR MODELO",
                           font=ctk.CTkFont(size=11, weight="bold"), text_color="#38bdf8")
        lbl.pack(side="left", padx=15)

    def render_action_bar_etapa4(self):
        for w in self.action_bar.winfo_children():
            w.destroy()
        lbl = ctk.CTkLabel(self.action_bar, text="🔍 REVISÃO VISUAL DO DATASET",
                           font=ctk.CTkFont(size=11, weight="bold"), text_color="#38bdf8")
        lbl.pack(side="left", padx=15)

        btn_prox = ctk.CTkButton(
            self.action_bar, text="Avançar para Exportar →", width=160, height=28,
            fg_color="#2563eb", hover_color="#1d4ed8", font=ctk.CTkFont(size=11, weight="bold"),
            command=lambda: self.mudar_passo(5)
        )
        btn_prox.pack(side="right", padx=15)

    def render_action_bar_etapa5(self):
        for w in self.action_bar.winfo_children():
            w.destroy()
        lbl = ctk.CTkLabel(self.action_bar, text="📦 EMPACOTAMENTO E EXPORTAÇÃO PARA TREINAMENTO",
                           font=ctk.CTkFont(size=11, weight="bold"), text_color="#22c55e")
        lbl.pack(side="left", padx=15)

    def mudar_passo(self, step_id):
        self.step_atual = step_id
        for idx, btn in self.steps_btn.items():
            if idx == step_id:
                btn.configure(fg_color="#1d283a", text_color="#38bdf8")
            else:
                btn.configure(fg_color="transparent", text_color="#64748b")

        self.lbl_preview.pack_forget()
        self.canvas_anotacao.pack_forget()
        self.frame_propagacao.pack_forget()
        self.frame_revisar.pack_forget()
        self.frame_exportar.pack_forget()

        if step_id == 1:
            self.lbl_preview.pack(fill="both", expand=True)
            self.render_action_bar_etapa1()
            if not self.arquivo_selecionado:
                self.esconder_painel_propriedades()
            else:
                self.atualizar_visibilidade_painel_direito()

            if self.arquivo_selecionado and self.arquivo_selecionado.lower().endswith(EXTENSOES_VIDEO):
                self.video_controls_frame.pack(side="bottom", fill="x", padx=10, pady=10)

        elif step_id == 2:
            self.parar_video()
            self.video_controls_frame.pack_forget()
            self.canvas_anotacao.pack(fill="both", expand=True)
            self.render_action_bar_etapa2()
            self.vincular_eventos_canvas()
            self.esconder_painel_propriedades()

            caminho_proj = self.project_data.get("caminho", "")
            pasta_frames = os.path.join(caminho_proj, "frames")

            if os.path.exists(pasta_frames) and os.listdir(pasta_frames):
                self.carregar_primeira_amostra()
                return

            if self.arquivo_selecionado and self.arquivo_selecionado.lower().endswith(EXTENSOES_VIDEO):
                self.mostrar_carregamento("Extraindo e salvando frames na pasta do projeto...")
                threading.Thread(target=self._executar_preparacao_frames_video_threaded, daemon=True).start()
            else:
                self.preparar_primeira_amostra_fotos()

        elif step_id == 3:
            self.parar_video()
            self.video_controls_frame.pack_forget()
            self.esconder_painel_propriedades()
            self.render_action_bar_etapa3()
            self.frame_propagacao.pack(fill="both", expand=True, padx=20, pady=20)

        elif step_id == 4:
            self.parar_video()
            self.video_controls_frame.pack_forget()
            self.esconder_painel_propriedades()
            self.render_action_bar_etapa4()
            self.frame_revisar.pack(fill="both", expand=True, padx=10, pady=10)
            self.popular_revisao_visual()

        elif step_id == 5:
            self.parar_video()
            self.video_controls_frame.pack_forget()
            self.esconder_painel_propriedades()
            self.render_action_bar_etapa5()
            self.frame_exportar.pack(fill="both", expand=True, padx=20, pady=20)

    # ==================== IMPLEMENTAÇÃO ETAPA 3: PROPAGAÇÃO POR MODELO ====================
    def build_ui_propagacao(self):
        lbl_title = ctk.CTkLabel(
            self.frame_propagacao, text="⚡ Propagação Automática de Rótulos por Modelo",
            font=ctk.CTkFont(size=15, weight="bold"), text_color="#38bdf8", anchor="w"
        )
        lbl_title.pack(fill="x", padx=20, pady=(20, 10))

        lbl_desc = ctk.CTkLabel(
            self.frame_propagacao,
            text="O modelo rastreará e propagará os rótulos definidos na primeira imagem para todos os demais quadros.",
            font=ctk.CTkFont(size=12), text_color="#94a3b8", justify="left"
        )
        lbl_desc.pack(fill="x", padx=20, pady=(0, 20))

        btn_propagar = ctk.CTkButton(
            self.frame_propagacao, text="🚀 Executar Modelo de Propagação", width=260, height=40,
            fg_color="#2563eb", hover_color="#1d4ed8", font=ctk.CTkFont(size=12, weight="bold"),
            command=self.executar_modelo_propagacao
        )
        btn_propagar.pack(padx=20, pady=10, anchor="w")

    def executar_modelo_propagacao(self):
        caminho_proj = self.project_data.get("caminho", "")
        pasta_frames = os.path.join(caminho_proj, "frames")
        pasta_annotations = os.path.join(caminho_proj, "annotations")

        arquivos_xml_json = [f for f in os.listdir(pasta_annotations) if f.endswith(('.xml', '.json'))]
        if not arquivos_xml_json:
            messagebox.showwarning("Aviso",
                                   "Você precisa rotular e salvar pelo menos a primeira imagem na Etapa 2 antes de propagar!")
            return

        molde_arquivo = arquivos_xml_json[0]
        molde_path = os.path.join(pasta_annotations, molde_arquivo)

        self.mostrar_carregamento("Executando modelo de propagação em todas as imagens...")

        def _processar():
            if os.path.exists(pasta_frames):
                for f in os.listdir(pasta_frames):
                    if f.lower().endswith(EXTENSOES_IMAGEM):
                        nome_base = os.path.splitext(f)[0]
                        ext_molde = os.path.splitext(molde_arquivo)[1]
                        destino_rotulo = os.path.join(pasta_annotations, f"{nome_base}{ext_molde}")

                        if not os.path.exists(destino_rotulo):
                            shutil.copy2(molde_path, destino_rotulo)

            def _finalizar():
                self.esconder_carregamento()
                messagebox.showinfo("Sucesso!", "Propagação concluída em todas as amostras!")
                self.mudar_passo(4)

            self.after(0, _finalizar)

        threading.Thread(target=_processar, daemon=True).start()

    # ==================== IMPLEMENTAÇÃO ETAPA 4: REVISÃO VISUAL ====================
    def build_ui_revisar(self):
        self.rev_left_frame = ctk.CTkScrollableFrame(self.frame_revisar, width=320, fg_color="#111622", corner_radius=6)
        self.rev_left_frame.pack(side="left", fill="y", padx=10, pady=10)

        self.rev_right_frame = ctk.CTkFrame(self.frame_revisar, fg_color="#111622", corner_radius=6)
        self.rev_right_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        lbl_prev_title = ctk.CTkLabel(self.rev_right_frame, text="🔍 INSPEÇÃO VISUAL DA AMOSTRA",
                                      font=ctk.CTkFont(size=12, weight="bold"), text_color="#38bdf8")
        lbl_prev_title.pack(pady=10)

        self.rev_canvas = ctk.CTkCanvas(self.rev_right_frame, bg="#05070a", highlightthickness=0)
        self.rev_canvas.pack(fill="both", expand=True, padx=10, pady=10)

        self.rev_info_lbl = ctk.CTkLabel(self.rev_right_frame, text="Selecione um frame ao lado para auditar.",
                                         font=ctk.CTkFont(size=11), text_color="#94a3b8")
        self.rev_info_lbl.pack(pady=10)

    def popular_revisao_visual(self):
        for w in self.rev_left_frame.winfo_children():
            w.destroy()

        caminho_proj = self.project_data.get("caminho", "")
        pasta_frames = os.path.join(caminho_proj, "frames")
        pasta_annotations = os.path.join(caminho_proj, "annotations")
        topologia = self.project_data.get("topologia", "Bounding Boxes")
        ext_rotulo = ".xml" if topologia == "Bounding Boxes" else ".json"

        if not os.path.exists(pasta_frames):
            return

        frames = sorted([f for f in os.listdir(pasta_frames) if f.lower().endswith(EXTENSOES_IMAGEM)])

        ctk.CTkLabel(self.rev_left_frame, text="📂 Amostras do Dataset:", font=ctk.CTkFont(size=11, weight="bold"),
                     text_color="#94a3b8").pack(anchor="w", padx=5, pady=5)

        for fname in frames:
            nome_base = os.path.splitext(fname)[0]
            img_path = os.path.join(pasta_frames, fname)
            rotulo_path = os.path.join(pasta_annotations, f"{nome_base}{ext_rotulo}")
            tem_rotulo = os.path.exists(rotulo_path)

            btn_cor = "#1e293b" if tem_rotulo else "#2d1b1e"
            btn = ctk.CTkButton(
                self.rev_left_frame, text=f"📄 {fname}", fg_color=btn_cor, hover_color="#334155",
                anchor="w", font=ctk.CTkFont(size=11),
                command=lambda ip=img_path, rp=rotulo_path: self.carregar_preview_revisao(ip, rp)
            )
            btn.pack(fill="x", padx=5, pady=3)

    def carregar_preview_revisao(self, img_path, rotulo_path):
        if not os.path.exists(img_path):
            return

        pil_img = Image.open(img_path)
        w_box, h_box = 450, 350
        new_w, new_h = self.redimensionar_proporcional(pil_img.width, pil_img.height, w_box, h_box)
        pil_resized = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        self.rev_tk_img = ImageTk.PhotoImage(pil_resized)
        self.rev_canvas.delete("all")

        cx = w_box // 2
        cy = h_box // 2
        self.rev_canvas.create_image(cx, cy, image=self.rev_tk_img, anchor="center")

        offset_x = cx - (new_w // 2)
        offset_y = cy - (new_h // 2)
        scale = new_w / pil_img.width

        classes_encontradas = []
        topologia = self.project_data.get("topologia", "Bounding Boxes")

        if os.path.exists(rotulo_path):
            try:
                if topologia == "Bounding Boxes":
                    tree = ET.parse(rotulo_path)
                    for obj in tree.getroot().findall("object"):
                        lbl = obj.find("name").text
                        classes_encontradas.append(lbl)
                        bnd = obj.find("bndbox")
                        xmin = int(bnd.find("xmin").text)
                        ymin = int(bnd.find("ymin").text)
                        xmax = int(bnd.find("xmax").text)
                        ymax = int(bnd.find("ymax").text)

                        rx1 = int(xmin * scale) + offset_x
                        ry1 = int(ymin * scale) + offset_y
                        rx2 = int(xmax * scale) + offset_x
                        ry2 = int(ymax * scale) + offset_y

                        self.rev_canvas.create_rectangle(rx1, ry1, rx2, ry2, outline="#38bdf8", width=2)
                        self.rev_canvas.create_text(rx1 + 5, ry1 + 10, text=lbl, fill="#38bdf8", anchor="w",
                                                    font=("Arial", 9, "bold"))
                else:
                    with open(rotulo_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        for shape in data.get("shapes", []):
                            lbl = shape.get("label")
                            classes_encontradas.append(lbl)
                            pts = shape.get("points", [])
                            canvas_pts = []
                            for rx, ry in pts:
                                cx_p = int(rx * scale) + offset_x
                                cy_p = int(ry * scale) + offset_y
                                canvas_pts.append((cx_p, cy_p))

                            if len(canvas_pts) > 1:
                                flat = [coord for pt in canvas_pts for coord in pt]
                                self.rev_canvas.create_polygon(flat, outline="#22c55e", fill="", width=2)
                                self.rev_canvas.create_text(canvas_pts[0][0] + 5, canvas_pts[0][1] + 10, text=lbl,
                                                            fill="#22c55e", anchor="w", font=("Arial", 9, "bold"))
            except Exception:
                pass

        status_str = f"Classes: {', '.join(set(classes_encontradas))}" if classes_encontradas else "⚠️ Sem rótulos salvos"
        self.rev_info_lbl.configure(text=f"Arquivo: {os.path.basename(img_path)}  |  {status_str}")

    # ==================== CARREGAR PRIMEIRA AMOSTRA (ETAPA 2) ====================
    def carregar_primeira_amostra(self):
        caminho_proj = self.project_data.get("caminho", "")
        pasta_frames = os.path.join(caminho_proj, "frames")
        if os.path.exists(pasta_frames):
            frames = sorted([os.path.join(pasta_frames, f) for f in os.listdir(pasta_frames) if
                             f.lower().endswith(EXTENSOES_IMAGEM)])
            if frames:
                self.lista_amostras = [frames[0]]
                self.amostra_index_atual = 0
                self.carregar_imagem_no_canvas(frames[0])

    def preparar_primeira_amostra_fotos(self):
        caminho_proj = self.project_data.get("caminho", "")
        if not caminho_proj:
            return
        todos = sorted(
            [os.path.join(caminho_proj, f) for f in os.listdir(caminho_proj) if f.lower().endswith(EXTENSOES_IMAGEM)])
        if todos:
            self.lista_amostras = [todos[0]]
            self.amostra_index_atual = 0
            self.carregar_imagem_no_canvas(todos[0])

    def carregar_imagem_no_canvas(self, filepath):
        if not filepath or not os.path.exists(filepath):
            return

        self.arquivo_selecionado = filepath
        nome_arq = os.path.basename(filepath)
        self.lbl_amostra_tag.configure(text=f"🔴 ROTULANDO 1ª AMOSTRA: #{nome_arq}")

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
        self.canvas_anotacao.create_image(w_box // 2, h_box // 2, image=self.tk_canvas_img, anchor="center")

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

            # Se já existir um arquivo xml para a primeira imagem, lemos para acumular múltiplas classes/objetos se houver
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

    # ==================== IMPLEMENTAÇÃO ETAPA 5: EXPORTAR DATASET ====================
    def build_ui_exportar(self):
        lbl_title = ctk.CTkLabel(
            self.frame_exportar, text="📦 Configuração de Exportação de Dataset",
            font=ctk.CTkFont(size=15, weight="bold"), text_color="#38bdf8", anchor="w"
        )
        lbl_title.pack(fill="x", padx=20, pady=(20, 10))

        lbl_fmt = ctk.CTkLabel(self.frame_exportar, text="Formato de Destino da IA:",
                               font=ctk.CTkFont(size=12, weight="bold"), text_color="#94a3b8", anchor="w")
        lbl_fmt.pack(fill="x", padx=20, pady=(10, 2))

        self.var_formato_export = ctk.StringVar(value="YOLOv8")
        formats_frame = ctk.CTkFrame(self.frame_exportar, fg_color="transparent")
        formats_frame.pack(fill="x", padx=20, pady=5)

        for fmt in ["YOLOv8", "COCO JSON", "Pascal VOC XML"]:
            ctk.CTkRadioButton(formats_frame, text=fmt, variable=self.var_formato_export, value=fmt,
                               font=ctk.CTkFont(size=11)).pack(side="left", padx=(0, 20))

        lbl_split = ctk.CTkLabel(self.frame_exportar, text="Divisão Train / Val (%):",
                                 font=ctk.CTkFont(size=12, weight="bold"), text_color="#94a3b8", anchor="w")
        lbl_split.pack(fill="x", padx=20, pady=(20, 2))

        split_container = ctk.CTkFrame(self.frame_exportar, fg_color="transparent")
        split_container.pack(fill="x", padx=20, pady=5)

        ctk.CTkLabel(split_container, text="Treino: 80%  |  Validação: 20%", font=ctk.CTkFont(size=11),
                     text_color="#e2e8f0").pack(side="left")

        btn_executar = ctk.CTkButton(
            self.frame_exportar, text="🚀 Gerar e Exportar Dataset Final", width=240, height=38,
            fg_color="#22c55e", hover_color="#16a34a", font=ctk.CTkFont(size=12, weight="bold"),
            command=self.executar_exportacao_dataset
        )
        btn_executar.pack(padx=20, pady=(40, 20), anchor="w")

    def executar_exportacao_dataset(self):
        caminho_proj = self.project_data.get("caminho", "")
        if not caminho_proj:
            messagebox.showerror("Erro", "Nenhum diretório de projeto definido.")
            return

        pasta_export = filedialog.askdirectory(title="Selecione a pasta para salvar o dataset exportado")
        if not pasta_export:
            return

        formato = self.var_formato_export.get()
        pasta_frames = os.path.join(caminho_proj, "frames")
        pasta_annotations = os.path.join(caminho_proj, "annotations")

        if not os.path.exists(pasta_frames):
            messagebox.showerror("Erro", "A pasta de frames do projeto não foi encontrada.")
            return

        topologia = self.project_data.get("topologia", "Bounding Boxes")
        ext_rotulo = ".xml" if topologia == "Bounding Boxes" else ".json"

        pares_validos = []
        for f in os.listdir(pasta_frames):
            if f.lower().endswith(EXTENSOES_IMAGEM):
                nome_base = os.path.splitext(f)[0]
                img_path = os.path.join(pasta_frames, f)
                rotulo_path = os.path.join(pasta_annotations, f"{nome_base}{ext_rotulo}")
                if os.path.exists(rotulo_path):
                    pares_validos.append((img_path, rotulo_path, f, f"{nome_base}{ext_rotulo}"))

        if not pares_validos:
            messagebox.showwarning("Aviso", "Nenhum frame rotulado foi encontrado para exportação!")
            return

        try:
            dir_train_img = os.path.join(pasta_export, "train", "images")
            dir_train_lbl = os.path.join(pasta_export, "train", "labels")
            dir_val_img = os.path.join(pasta_export, "val", "images")
            dir_val_lbl = os.path.join(pasta_export, "val", "labels")

            os.makedirs(dir_train_img, exist_ok=True)
            os.makedirs(dir_train_lbl, exist_ok=True)
            os.makedirs(dir_val_img, exist_ok=True)
            os.makedirs(dir_val_lbl, exist_ok=True)

            total_amostras = len(pares_validos)
            corte = int(total_amostras * 0.8)
            treino_amostras = pares_validos[:corte] if corte > 0 else pares_validos
            val_amostras = pares_validos[corte:] if corte > 0 else []

            if not treino_amostras and val_amostras:
                treino_amostras.append(val_amostras.pop(0))

            for img_p, lbl_p, img_name, lbl_name in treino_amostras:
                shutil.copy2(img_p, os.path.join(dir_train_img, img_name))
                shutil.copy2(lbl_p, os.path.join(dir_train_lbl, lbl_name))

            for img_p, lbl_p, img_name, lbl_name in val_amostras:
                shutil.copy2(img_p, os.path.join(dir_val_img, img_name))
                shutil.copy2(lbl_p, os.path.join(dir_val_lbl, lbl_name))

            messagebox.showinfo(
                "Exportação Concluída! 🎉",
                f"Dataset exportado para:\n{pasta_export}\n\n"
                f"• Formato: {formato}\n"
                f"• Total exportado: {total_amostras}\n"
                f"• Treino: {len(treino_amostras)} | Validação: {len(val_amostras)}"
            )
        except Exception as e:
            messagebox.showerror("Erro", f"Ocorreu um erro ao salvar os arquivos:\n{e}")

    # ==================== EXTRAÇÃO EM THREAD ====================
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
            self.carregar_primeira_amostra()

        self.after(0, finalizar_carregamento)

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
        self.esconder_painel_propriedades()

    def esconder_painel_propriedades(self):
        self.right_container.pack_forget()

    def atualizar_visibilidade_painel_direito(self):
        if self.step_atual == 1 and self.arquivo_selecionado:
            self.right_container.pack(fill="both", expand=True)
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