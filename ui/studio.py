import os
import shutil
import cv2
import json
import numpy as np
import threading
import urllib.request
import warnings
import xml.etree.ElementTree as ET
from xml.dom import minidom
import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from annotations.storage import save_annotations
from dataset.exporter import export_dataset
from media.media_service import extract_frames
from ml.model_io import export_model, infer_image
from ml.mobile_sam_training import train_mobile_sam
from ml.yolo_training import train_yolo
from ui.panels.model_tools import ModelToolsPanel
from ui.panels.propagation import PropagationPanel
from ui.panels.review import ReviewPanel
from ui.panels.training import TrainingPanel

try:
    warnings.filterwarnings(
        "ignore", message="Importing from timm.models.layers is deprecated.*", category=FutureWarning
    )
    warnings.filterwarnings(
        "ignore", message="Importing from timm.models.registry is deprecated.*", category=FutureWarning
    )
    warnings.filterwarnings(
        "ignore", message="Overwriting tiny_vit_.* in registry with mobile_sam.*", category=UserWarning
    )
    import torch
    import torch.nn.functional as F
    from mobile_sam import SamAutomaticMaskGenerator, build_sam_vit_t
    from mobile_sam.utils.transforms import ResizeLongestSide

    MOBILE_SAM_DISPONIVEL = True
except ImportError:
    MOBILE_SAM_DISPONIVEL = False

# Importação opcional segura do Ultralytics YOLO
try:
    from ultralytics import YOLO

    ULTRALYTICS_DISPONIVEL = True
except ImportError:
    ULTRALYTICS_DISPONIVEL = False

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
        self.treino_pausar_solicitado = False
        self.modelo_treinamento_ativo = None
        self.pasta_ultimo_treinamento = None
        self.caminho_modelo_visualizacao = ""
        self.caminho_imagem_visualizacao = ""
        self.caminho_modelo_exportacao = ""

        # Referências de Imagens do Canvas
        self.pil_canvas_img = None
        self.pil_canvas_img_resized = None
        self.tk_canvas_img = None

        # Mídias/Amostras para Anotação
        self.lista_amostras = []
        self.amostra_index_atual = 0

        # Estado das Anotações na Imagem Atual
        self.annotations = []
        self.temp_shape_id = None
        self.start_x = None
        self.start_y = None

        # Específico para Polígonos (Segmentação)
        self.current_polygon_points = []
        self.polygon_line_ids = []

        # Amostra atualmente selecionada na Revisão
        self.revisao_amostra_atual = None
        self.revisao_rotulo_atual = None

        # Dicionário de Checkboxes na Revisão para Exclusão em Lote: {caminho_rotulo: BooleanVar}
        self.rev_checkboxes_vars = {}

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
            (2, "ROTULAR AMOSTRAS"),
            (3, "PROPAGAR ANOTAÇÃO"),
            (4, "REVISAR"),
            (5, "EXPORTAR DATASET"),
            (6, "TREINAR MODELO"),
            (7, "VISUALIZAR TREINAMENTO"),
            (8, "EXPORTAR MODELO")
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
        self.lbl_amostra_tag.pack_forget()

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
        self.frame_propagacao = ctk.CTkScrollableFrame(self.display_container, fg_color="#0b0e14", corner_radius=8)
        self.propagation_panel = PropagationPanel(self, self.frame_propagacao)

        # PAINEL DE REVISAR (Etapa 4)
        self.frame_revisar = ctk.CTkFrame(self.display_container, fg_color="#0b0e14", corner_radius=8)
        self.review_panel = ReviewPanel(self, self.frame_revisar)

        # PAINEL DE EXPORTAR DATASET (Etapa 5)
        self.frame_exportar = ctk.CTkFrame(self.display_container, fg_color="#0b0e14", corner_radius=8)
        self.build_ui_exportar()

        # PAINEL DE TREINAMENTO (Etapa 6)
        self.frame_treinamento = ctk.CTkFrame(self.display_container, fg_color="#0b0e14", corner_radius=8)
        self.training_panel = TrainingPanel(self, self.frame_treinamento)

        # PAINEL DE VISUALIZAÇÃO DO MODELO (Etapa 7)
        self.frame_visualizar_modelo = ctk.CTkFrame(self.display_container, fg_color="#0b0e14", corner_radius=8)

        # PAINEL DE EXPORTAÇÃO DO MODELO (Etapa 8)
        self.frame_exportar_modelo = ctk.CTkFrame(self.display_container, fg_color="#0b0e14", corner_radius=8)
        self.model_tools_panel = ModelToolsPanel(
            self, self.frame_visualizar_modelo, self.frame_exportar_modelo
        )

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
            self.action_bar, text="Próxima Etapa → (Rotular Amostras)", width=220, height=28,
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
        lbl_top.pack(side="left", padx=(15, 10))

        lbl_c = ctk.CTkLabel(self.action_bar, text="Classe(s):", font=ctk.CTkFont(size=11), text_color="#94a3b8")
        lbl_c.pack(side="left", padx=(0, 5))

        self.entry_classe = ctk.CTkEntry(self.action_bar, placeholder_text="Ex: Anomalia, Defeito", width=150,
                                         height=28, fg_color="#080a0f")
        self.entry_classe.insert(0, "Objeto")
        self.entry_classe.pack(side="left", padx=5)

        btn_limpar = ctk.CTkButton(self.action_bar, text="🧹 Limpar Frame", width=100, height=28, fg_color="#ef4444",
                                   hover_color="#dc2626", command=self.limpar_anotacoes)
        btn_limpar.pack(side="left", padx=10)

        if topologia != "Bounding Boxes":
            ctk.CTkLabel(
                self.action_bar, text="Botão direito conclui o polígono",
                font=ctk.CTkFont(size=10, weight="bold"), text_color="#86efac"
            ).pack(side="left", padx=(8, 4))
            btn_fechar_poligono = ctk.CTkButton(
                self.action_bar, text="✓ Fechar Polígono", width=130, height=28,
                fg_color="#16a34a", hover_color="#15803d", command=self.on_polygon_finish
            )
            btn_fechar_poligono.pack(side="left", padx=2)

        btn_prox_amostra = ctk.CTkButton(
            self.action_bar, text="Próxima Amostra ⏭", width=130, height=28,
            fg_color="#334155", hover_color="#475569", font=ctk.CTkFont(size=11),
            command=self.proxima_amostra_rotulo
        )
        btn_prox_amostra.pack(side="right", padx=10)

        btn_ant_amostra = ctk.CTkButton(
            self.action_bar, text="⏮ Amostra Anterior", width=130, height=28,
            fg_color="#334155", hover_color="#475569", font=ctk.CTkFont(size=11),
            command=self.anterior_amostra_rotulo
        )
        btn_ant_amostra.pack(side="right", padx=2)

        btn_avancar = ctk.CTkButton(
            self.action_bar, text="Avançar para Propagar →", width=170, height=28,
            fg_color="#2563eb", hover_color="#1d4ed8", font=ctk.CTkFont(size=11, weight="bold"),
            command=lambda: self.mudar_passo(3)
        )
        btn_avancar.pack(side="right", padx=15)

    def render_action_bar_etapa3(self):
        for w in self.action_bar.winfo_children():
            w.destroy()
        lbl = ctk.CTkLabel(self.action_bar, text="⚡ ESCOLHA O MÉTODO DE PROPAGAÇÃO BASEADO NAS SUAS AMOSTRAS",
                           font=ctk.CTkFont(size=11, weight="bold"), text_color="#38bdf8")
        lbl.pack(side="left", padx=15)

    def render_action_bar_etapa4(self):
        for w in self.action_bar.winfo_children():
            w.destroy()
        lbl = ctk.CTkLabel(self.action_bar, text="🔍 REVISÃO VISUAL E REMOÇÃO",
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

        btn_treinar = ctk.CTkButton(
            self.action_bar, text="Ir para Treinamento →", width=160, height=28,
            fg_color="#2563eb", hover_color="#1d4ed8", font=ctk.CTkFont(size=11, weight="bold"),
            command=lambda: self.mudar_passo(6)
        )
        btn_treinar.pack(side="right", padx=15)

    def render_action_bar_etapa6(self):
        for w in self.action_bar.winfo_children():
            w.destroy()
        lbl = ctk.CTkLabel(self.action_bar, text="🧠 TREINAMENTO DO MODELO DE IA",
                           font=ctk.CTkFont(size=11, weight="bold"), text_color="#a855f7")
        lbl.pack(side="left", padx=15)

    def render_action_bar_etapa7(self):
        for w in self.action_bar.winfo_children():
            w.destroy()
        lbl = ctk.CTkLabel(self.action_bar, text="🔎 VISUALIZAÇÃO E TESTE DO MODELO",
                           font=ctk.CTkFont(size=11, weight="bold"), text_color="#22c55e")
        lbl.pack(side="left", padx=15)

    def render_action_bar_etapa8(self):
        for w in self.action_bar.winfo_children():
            w.destroy()
        lbl = ctk.CTkLabel(self.action_bar, text="📤 EXPORTAÇÃO DO MODELO TREINADO",
                           font=ctk.CTkFont(size=11, weight="bold"), text_color="#f59e0b")
        lbl.pack(side="left", padx=15)

    def mudar_passo(self, step_id):
        if (self.step_atual == 2 and step_id != 2 and
            self.project_data.get("topologia", "Bounding Boxes") != "Bounding Boxes"):
            self.on_polygon_finish()
        self.step_atual = step_id
        self.lbl_amostra_tag.pack_forget()
        if step_id == 2:
            self.lbl_amostra_tag.pack(side="left")
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
        self.frame_treinamento.pack_forget()
        self.frame_visualizar_modelo.pack_forget()
        self.frame_exportar_modelo.pack_forget()

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

            # Tratamento unificado de mídias: Vídeo vs Imagens
            if self.midia_eh_video:
                if os.path.exists(pasta_frames) and os.listdir(pasta_frames):
                    self.carregar_lista_amostras_para_rotulo()
                elif self.arquivo_selecionado and self.arquivo_selecionado.lower().endswith(EXTENSOES_VIDEO):
                    self.mostrar_carregamento("Extraindo e salvando frames na pasta do projeto...")
                    threading.Thread(target=self._executar_preparacao_frames_video_threaded, daemon=True).start()
            else:
                self.carregar_lista_amostras_fotos()

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

        elif step_id == 6:
            self.parar_video()
            self.video_controls_frame.pack_forget()
            self.esconder_painel_propriedades()
            self.render_action_bar_etapa6()
            self.frame_treinamento.pack(fill="both", expand=True, padx=20, pady=20)

        elif step_id == 7:
            self.parar_video()
            self.video_controls_frame.pack_forget()
            self.esconder_painel_propriedades()
            self.render_action_bar_etapa7()
            self.frame_visualizar_modelo.pack(fill="both", expand=True, padx=20, pady=20)

        elif step_id == 8:
            self.parar_video()
            self.video_controls_frame.pack_forget()
            self.esconder_painel_propriedades()
            self.render_action_bar_etapa8()
            self.frame_exportar_modelo.pack(fill="both", expand=True, padx=20, pady=20)

    def selecionar_modelo_visualizacao(self):
        caminho = filedialog.askopenfilename(
            title="Selecione o modelo treinado",
            filetypes=[("Modelos", "*.pt *.onnx *.engine *.torchscript"), ("Todos os arquivos", "*.*")]
        )
        if caminho:
            self.caminho_modelo_visualizacao = caminho
            self.entry_modelo_visualizacao.delete(0, "end")
            self.entry_modelo_visualizacao.insert(0, caminho)

    def selecionar_imagem_visualizacao(self):
        caminho = filedialog.askopenfilename(
            title="Selecione a imagem para testar",
            filetypes=[("Imagens", "*.png *.jpg *.jpeg *.bmp *.tiff *.webp"), ("Todos os arquivos", "*.*")]
        )
        if caminho:
            self.caminho_imagem_visualizacao = caminho
            self.entry_imagem_visualizacao.delete(0, "end")
            self.entry_imagem_visualizacao.insert(0, caminho)

    def executar_inferencia_modelo(self):
        if not ULTRALYTICS_DISPONIVEL:
            messagebox.showerror("Erro", "A biblioteca Ultralytics não está instalada no ambiente.")
            return
        caminho_modelo = self.entry_modelo_visualizacao.get().strip()
        caminho_imagem = self.entry_imagem_visualizacao.get().strip()
        if not os.path.isfile(caminho_modelo) or not os.path.isfile(caminho_imagem):
            messagebox.showwarning("Aviso", "Selecione um modelo e uma imagem válidos.")
            return

        mensagem = "Aplicando MobileSAM na imagem..." if self.var_tipo_visualizacao.get() == "MobileSAM" \
            else "Aplicando YOLO na imagem..."
        self.mostrar_carregamento(mensagem)
        threading.Thread(
            target=self._executar_inferencia_modelo_threaded,
            args=(caminho_modelo, caminho_imagem, self.var_tipo_visualizacao.get()), daemon=True
        ).start()

    def _executar_inferencia_modelo_threaded(self, caminho_modelo, caminho_imagem, tipo_modelo):
        try:
            imagem = infer_image(caminho_modelo, caminho_imagem, tipo_modelo)
            self.after(0, lambda: self._mostrar_resultado_inferencia(imagem))
        except Exception as erro:
            erro_msg = str(erro)
            self.after(0, lambda: messagebox.showerror("Erro", f"Falha ao aplicar o modelo:\n{erro_msg}"))
        finally:
            self.after(0, self.esconder_carregamento)

    def _mostrar_resultado_inferencia(self, imagem):
        largura = max(500, self.frame_visualizar_modelo.winfo_width() - 50)
        altura = max(350, self.frame_visualizar_modelo.winfo_height() - 180)
        imagem.thumbnail((largura, altura), Image.Resampling.LANCZOS)
        imagem_ctk = ctk.CTkImage(
            light_image=imagem, dark_image=imagem, size=(imagem.width, imagem.height)
        )
        self.lbl_resultado_visualizacao.configure(image=imagem_ctk, text="")
        self.lbl_resultado_visualizacao.image = imagem_ctk

    def selecionar_modelo_exportacao(self):
        caminho = filedialog.askopenfilename(
            title="Selecione o modelo treinado",
            filetypes=[("Modelos", "*.pt *.onnx *.engine *.torchscript"), ("Todos os arquivos", "*.*")]
        )
        if caminho:
            self.caminho_modelo_exportacao = caminho
            self.entry_modelo_exportacao.delete(0, "end")
            self.entry_modelo_exportacao.insert(0, caminho)

    def selecionar_destino_modelo(self):
        caminho = filedialog.askdirectory(title="Selecione a pasta de destino do modelo")
        if caminho:
            self.entry_destino_modelo.delete(0, "end")
            self.entry_destino_modelo.insert(0, caminho)

    def executar_exportacao_modelo(self):
        tipo_modelo = self.var_tipo_exportacao_modelo.get()
        if tipo_modelo == "YOLO" and not ULTRALYTICS_DISPONIVEL:
            messagebox.showerror("Erro", "A biblioteca Ultralytics não está instalada no ambiente.")
            return
        if tipo_modelo == "MobileSAM" and not MOBILE_SAM_DISPONIVEL:
            messagebox.showerror("Erro", "A biblioteca MobileSAM não está instalada no ambiente.")
            return
        caminho_modelo = self.entry_modelo_exportacao.get().strip()
        pasta_destino = self.entry_destino_modelo.get().strip()
        formato = self.var_formato_modelo.get()
        if not os.path.isfile(caminho_modelo) or not os.path.isdir(pasta_destino):
            messagebox.showwarning("Aviso", "Selecione um modelo e uma pasta de destino válidos.")
            return

        self.mostrar_carregamento(f"Exportando modelo para {formato}...")
        threading.Thread(
            target=self._executar_exportacao_modelo_threaded,
            args=(caminho_modelo, pasta_destino, formato, tipo_modelo), daemon=True
        ).start()

    def _executar_exportacao_modelo_threaded(self, caminho_modelo, pasta_destino, formato, tipo_modelo):
        try:
            destino = export_model(caminho_modelo, pasta_destino, formato, tipo_modelo)
            self.after(0, lambda: messagebox.showinfo(
                "Sucesso", f"Modelo exportado com sucesso para:\n{destino}"
            ))
        except Exception as erro:
            erro_msg = str(erro)
            self.after(0, lambda: messagebox.showerror("Erro", f"Falha ao exportar o modelo:\n{erro_msg}"))
        finally:
            self.after(0, self.esconder_carregamento)

    def executar_modelo_propagacao(self, metodo):
        caminho_proj = self.project_data.get("caminho", "")
        pasta_frames = os.path.join(caminho_proj, "frames")
        pasta_annotations = os.path.join(caminho_proj, "annotations")
        topologia = self.project_data.get("topologia", "Bounding Boxes")

        if metodo == "yolo" and topologia != "Bounding Boxes":
            messagebox.showwarning(
                "Método indisponível",
                "A propagação YOLO está disponível somente para caixas delimitadoras."
            )
            return

        if not os.path.exists(pasta_annotations):
            messagebox.showwarning("Aviso", "A pasta de anotações não existe. Rotule uma amostra na Etapa 2.")
            return

        extensao_rotulo = ".xml" if topologia == "Bounding Boxes" else ".json"
        arquivos_rotulo = [
            f for f in os.listdir(pasta_annotations) if f.lower().endswith(extensao_rotulo)
        ]
        if not arquivos_rotulo:
            messagebox.showwarning("Aviso",
                                   f"Você precisa salvar pelo menos uma anotação {extensao_rotulo} antes de propagar!")
            return

        molde_path = os.path.join(pasta_annotations, arquivos_rotulo[0])

        nomes_metodos = {
            "molde": "Interpolagem de Referências",
            "opencv": "Rastreamento Temporal OpenCV",
            "yolo": "Modelo Ultralytics YOLO"
        }

        self.mostrar_carregamento(f"Executando propagação via {nomes_metodos.get(metodo, 'Modelo')}...")

        def _processar():
            sucesso = False
            erro_msg = ""
            try:
                diretorio_busca = pasta_frames if os.path.exists(pasta_frames) else caminho_proj
                frames = sorted([f for f in os.listdir(diretorio_busca) if f.lower().endswith(EXTENSOES_IMAGEM)])
                ext_rotulo = ".xml" if topologia == "Bounding Boxes" else ".json"

                if metodo == "yolo" and ULTRALYTICS_DISPONIVEL:
                    try:
                        model = YOLO("yolov8n.pt")
                        print("\n[INFO] Iniciando inferência YOLO...")

                        for idx, fname in enumerate(frames):
                            nome_base = os.path.splitext(fname)[0]
                            img_path = os.path.join(diretorio_busca, fname)
                            dest_rotulo = os.path.join(pasta_annotations, f"{nome_base}{ext_rotulo}")

                            if os.path.exists(dest_rotulo):
                                continue

                            results = model(img_path, verbose=False)
                            pil_img = Image.open(img_path)
                            orig_w, orig_h = pil_img.size

                            if topologia == "Bounding Boxes":
                                annotation_node = ET.Element("annotation")
                                ET.SubElement(annotation_node, "folder").text = os.path.basename(caminho_proj)
                                ET.SubElement(annotation_node, "filename").text = fname
                                ET.SubElement(annotation_node, "path").text = img_path

                                size_node = ET.SubElement(annotation_node, "size")
                                ET.SubElement(size_node, "width").text = str(orig_w)
                                ET.SubElement(size_node, "height").text = str(orig_h)
                                ET.SubElement(size_node, "depth").text = "3"

                                encontrou_obj = False
                                for r in results:
                                    for box in r.boxes:
                                        coords = box.xyxy[0].tolist()
                                        cls_id = int(box.cls[0])
                                        cls_name = model.names[cls_id]

                                        object_node = ET.SubElement(annotation_node, "object")
                                        ET.SubElement(object_node, "name").text = cls_name
                                        bndbox_node = ET.SubElement(object_node, "bndbox")
                                        ET.SubElement(bndbox_node, "xmin").text = str(int(coords[0]))
                                        ET.SubElement(bndbox_node, "ymin").text = str(int(coords[1]))
                                        ET.SubElement(bndbox_node, "xmax").text = str(int(coords[2]))
                                        ET.SubElement(bndbox_node, "ymax").text = str(int(coords[3]))
                                        encontrou_obj = True

                                if encontrou_obj:
                                    xml_string = minidom.parseString(ET.tostring(annotation_node)).toprettyxml(
                                        indent="  ")
                                    with open(dest_rotulo, "w", encoding="utf-8") as wf:
                                        wf.write(xml_string)
                                else:
                                    shutil.copy2(molde_path, dest_rotulo)
                            else:
                                shutil.copy2(molde_path, dest_rotulo)
                    except Exception as e:
                        print(f"[AVISO] Erro no pipeline YOLO: {e}. Revertendo para cópia de molde.")
                        for fname in frames:
                            nome_base = os.path.splitext(fname)[0]
                            dest_rotulo = os.path.join(pasta_annotations, f"{nome_base}{ext_rotulo}")
                            if not os.path.exists(dest_rotulo):
                                shutil.copy2(molde_path, dest_rotulo)
                else:
                    for f in frames:
                        nome_base = os.path.splitext(f)[0]
                        dest_rotulo = os.path.join(pasta_annotations, f"{nome_base}{ext_rotulo}")
                        if not os.path.exists(dest_rotulo):
                            shutil.copy2(molde_path, dest_rotulo)

                sucesso = True
            except Exception as e:
                erro_msg = str(e)
            finally:
                def _finalizar():
                    self.esconder_carregamento()
                    if sucesso:
                        messagebox.showinfo("Sucesso!",
                                            f"Propagação via {nomes_metodos.get(metodo, 'Modelo')} concluída!")
                        self.mudar_passo(4)
                    else:
                        messagebox.showerror("Erro na Propagação", f"Ocorreu uma falha ao propagar:\n{erro_msg}")

                self.after(0, _finalizar)

        threading.Thread(target=_processar, daemon=True).start()

    def popular_revisao_visual(self):
        for w in self.rev_left_frame.winfo_children():
            w.destroy()

        self.revisao_amostra_atual = None
        self.revisao_rotulo_atual = None
        self.rev_checkboxes_vars.clear()

        caminho_proj = self.project_data.get("caminho", "")
        pasta_frames = os.path.join(caminho_proj, "frames")
        pasta_annotations = os.path.join(caminho_proj, "annotations")
        topologia = self.project_data.get("topologia", "Bounding Boxes")
        ext_rotulo = ".xml" if topologia == "Bounding Boxes" else ".json"

        diretorio_busca = pasta_frames if os.path.exists(pasta_frames) else caminho_proj
        if not os.path.exists(diretorio_busca):
            return

        frames = sorted([f for f in os.listdir(diretorio_busca) if f.lower().endswith(EXTENSOES_IMAGEM)])

        header_chk_frame = ctk.CTkFrame(self.rev_left_frame, fg_color="transparent")
        header_chk_frame.pack(fill="x", padx=2, pady=5)

        self.var_select_all = ctk.BooleanVar(value=False)
        chk_all = ctk.CTkCheckBox(
            header_chk_frame, text="Selecionar todos", variable=self.var_select_all,
            font=ctk.CTkFont(size=11, weight="bold"), text_color="#38bdf8",
            checkbox_width=18, checkbox_height=18, command=self.toggle_select_all
        )
        chk_all.pack(side="left", padx=2)

        for fname in frames:
            nome_base = os.path.splitext(fname)[0]
            img_path = os.path.join(diretorio_busca, fname)
            rotulo_path = os.path.join(pasta_annotations, f"{nome_base}{ext_rotulo}")
            tem_rotulo = os.path.exists(rotulo_path)

            row_frame = ctk.CTkFrame(self.rev_left_frame, fg_color="transparent")
            row_frame.pack(fill="x", padx=2, pady=2)

            var_chk = ctk.BooleanVar(value=False)
            self.rev_checkboxes_vars[rotulo_path] = var_chk
            chk = ctk.CTkCheckBox(row_frame, text="", variable=var_chk, width=20, checkbox_width=18, checkbox_height=18)
            chk.pack(side="left", padx=(2, 6))

            btn_cor = "#1e293b" if tem_rotulo else "#2d1b1e"
            btn = ctk.CTkButton(
                row_frame, text=f"📄 {fname}", fg_color=btn_cor, hover_color="#334155",
                anchor="w", font=ctk.CTkFont(size=11),
                command=lambda ip=img_path, rp=rotulo_path: self.carregar_preview_revisao(ip, rp)
            )
            btn.pack(side="left", fill="x", expand=True)

    def toggle_select_all(self):
        estado_desejado = self.var_select_all.get()
        for var in self.rev_checkboxes_vars.values():
            var.set(estado_desejado)

    def carregar_preview_revisao(self, img_path, rotulo_path):
        if not os.path.exists(img_path):
            return

        self.revisao_amostra_atual = img_path
        self.revisao_rotulo_atual = rotulo_path

        pil_img = Image.open(img_path)
        self.rev_canvas.update_idletasks()
        w_box = max(self.rev_canvas.winfo_width(), 1)
        h_box = max(self.rev_canvas.winfo_height(), 1)
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
                    objs = tree.getroot().findall("object")
                    for obj in objs:
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
                        shapes = data.get("shapes", [])
                        for shape in shapes:
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

    def atualizar_preview_revisao(self, _event=None):
        if self.revisao_amostra_atual and self.revisao_rotulo_atual:
            self.carregar_preview_revisao(self.revisao_amostra_atual, self.revisao_rotulo_atual)

    def excluir_anotacoes_em_lote(self):
        removidos_count = 0
        for rotulo_path, var_chk in self.rev_checkboxes_vars.items():
            if var_chk.get() and os.path.exists(rotulo_path):
                try:
                    os.remove(rotulo_path)
                    removidos_count += 1
                except Exception:
                    pass

        if removidos_count > 0:
            messagebox.showinfo("Sucesso", f"{removidos_count} anotação(ões) removida(s) com sucesso!")
            self.popular_revisao_visual()
            self.rev_canvas.delete("all")
            self.rev_info_lbl.configure(text="Selecione um frame ao lado para auditar.")
        else:
            messagebox.showwarning("Aviso", "Nenhum item foi marcado com checkbox para remoção.")

    # ==================== CARREGAR MÚLTIPLAS AMOSTRAS (ETAPA 2) ====================
    def carregar_lista_amostras_para_rotulo(self):
        caminho_proj = self.project_data.get("caminho", "")
        pasta_frames = os.path.join(caminho_proj, "frames")
        if os.path.exists(pasta_frames):
            self.lista_amostras = sorted([os.path.join(pasta_frames, f) for f in os.listdir(pasta_frames) if
                                          f.lower().endswith(EXTENSOES_IMAGEM)])
            if self.lista_amostras:
                self.amostra_index_atual = 0
                self.carregar_imagem_no_canvas(self.lista_amostras[0])

    def carregar_lista_amostras_fotos(self):
        caminho_proj = self.project_data.get("caminho", "")
        if not caminho_proj or not os.path.exists(caminho_proj):
            return

        self.lista_amostras = sorted(
            [os.path.join(caminho_proj, f) for f in os.listdir(caminho_proj) if f.lower().endswith(EXTENSOES_IMAGEM)]
        )
        if self.lista_amostras:
            self.amostra_index_atual = 0
            self.carregar_imagem_no_canvas(self.lista_amostras[0])

    def proxima_amostra_rotulo(self):
        if not self.lista_amostras:
            return
        self.amostra_index_atual = (self.amostra_index_atual + 1) % len(self.lista_amostras)
        self.carregar_imagem_no_canvas(self.lista_amostras[self.amostra_index_atual])

    def anterior_amostra_rotulo(self):
        if not self.lista_amostras:
            return
        self.amostra_index_atual = (self.amostra_index_atual - 1) % len(self.lista_amostras)
        self.carregar_imagem_no_canvas(self.lista_amostras[self.amostra_index_atual])

    def carregar_imagem_no_canvas(self, filepath):
        if not filepath or not os.path.exists(filepath):
            return

        self.arquivo_selecionado = filepath
        nome_arq = os.path.basename(filepath)
        self.lbl_amostra_tag.configure(
            text=f"🔴 ROTULANDO AMOSTRA [{self.amostra_index_atual + 1}/{len(self.lista_amostras)}]: #{nome_arq}")

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
        if len(self.current_polygon_points) >= 3:
            primeiro_x, primeiro_y = self.current_polygon_points[0]
            if ((x - primeiro_x) ** 2 + (y - primeiro_y) ** 2) <= 12 ** 2:
                self.on_polygon_finish()
                return
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
            if not silencioso:
                messagebox.showwarning("Aviso", "Nenhuma imagem está selecionada para receber a anotação.")
            return

        caminho_proj = self.project_data.get("caminho", "")
        if not caminho_proj or not os.path.isdir(caminho_proj):
            caminho_proj = self.garantir_diretorio_projeto()
        if not caminho_proj or not os.path.isdir(caminho_proj):
            messagebox.showerror(
                "Erro ao salvar anotação",
                "Selecione uma pasta de projeto válida antes de salvar o rótulo."
            )
            return

        try:
            output_path = save_annotations(
                project_path=caminho_proj,
                image_path=self.arquivo_selecionado,
                annotations=self.annotations,
                topologia=self.project_data.get("topologia", "Bounding Boxes"),
                image_size=self.pil_canvas_img.size,
            )
        except OSError as erro:
            messagebox.showerror(
                "Erro ao salvar anotação",
                f"Não foi possível salvar a anotação:\n{erro}"
            )
            return

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
        if not caminho_proj or not os.path.isdir(caminho_proj):
            messagebox.showerror("Erro", "Nenhum diretório de projeto válido foi definido.")
            return

        pasta_export = filedialog.askdirectory(title="Selecione a pasta para salvar o dataset exportado")
        if not pasta_export:
            return

        try:
            resumo = export_dataset(caminho_proj, pasta_export, self.var_formato_export.get())
            messagebox.showinfo(
                "Exportação Concluída!",
                f"Dataset exportado para:\n{resumo['path']}\n\n"
                f"Formato: {resumo['format']}\n"
                f"Total: {resumo['total']}\n"
                f"Treino: {resumo['train']} | Validação: {resumo['val']}"
            )
        except ValueError as erro:
            messagebox.showwarning("Aviso", str(erro))
        except (OSError, json.JSONDecodeError, ET.ParseError) as erro:
            messagebox.showerror("Erro", f"Ocorreu um erro ao exportar o dataset:\n{erro}")

    # ==================== IMPLEMENTAÇÃO ETAPA 6: TREINAMENTO DO MODELO ====================
    def atualizar_modo_treinamento(self):
        continuar = self.var_modo_treinamento.get() == "continuar"
        estado = "normal" if continuar else "disabled"
        self.entry_pasta_treinamento.configure(state=estado)
        self.btn_pasta_treinamento.configure(state=estado)

    def selecionar_pasta_treinamento(self):
        pasta = filedialog.askdirectory(title="Selecione a pasta do treinamento anterior")
        if pasta:
            self.entry_pasta_treinamento.configure(state="normal")
            self.entry_pasta_treinamento.delete(0, "end")
            self.entry_pasta_treinamento.insert(0, pasta)

    def solicitar_pausa_treinamento(self):
        self.treino_pausar_solicitado = True
        self.btn_pausar_treino.configure(state="disabled", text="⏳ Pausando...")
        self.log_treinamento("[INFO] Pausa solicitada. O treinamento será interrompido ao fim da época atual...")
        treinador = getattr(self.modelo_treinamento_ativo, "trainer", None)
        if treinador is not None:
            treinador.stop = True

    def localizar_checkpoint_treinamento(self, pasta):
        if os.path.isfile(pasta) and os.path.basename(pasta).lower() == "last.pt":
            return pasta
        if not os.path.isdir(pasta):
            return None

        candidatos = [
            os.path.join(pasta, "weights", "last.pt"),
            os.path.join(pasta, "last.pt")
        ]
        for candidato in candidatos:
            if os.path.isfile(candidato):
                return candidato

        for raiz, _, arquivos in os.walk(pasta):
            if "last.pt" in arquivos:
                return os.path.join(raiz, "last.pt")
        return None

    def log_treinamento(self, mensagem):
        """Função auxiliar para escrever mensagens no console da interface Tkinter."""
        self.textbox_log.insert("end", f"{mensagem}\n")
        self.textbox_log.see("end")

    def iniciar_treinamento_modelo(self):
        topologia = self.project_data.get("topologia", "Bounding Boxes")

        self.textbox_log.delete("1.0", "end")
        self.log_treinamento(f"[INFO] Iniciando processo de treinamento para topologia: {topologia}")

        if topologia == "Bounding Boxes":
            if not ULTRALYTICS_DISPONIVEL:
                messagebox.showerror("Erro", "A biblioteca Ultralytics (YOLO) não está instalada no ambiente.")
                return

            self.btn_iniciar_treino.configure(state="disabled")
            self.btn_pausar_treino.configure(state="normal")
            self.treino_pausar_solicitado = False
            mensagem = "Continuando treinamento YOLO..." if self.var_modo_treinamento.get() == "continuar" else "Treinando modelo YOLO..."
            self.mostrar_carregamento(mensagem)
            threading.Thread(target=self._treinar_yolo_threaded, daemon=True).start()
        else:
            self.btn_iniciar_treino.configure(state="disabled")
            self.btn_pausar_treino.configure(state="normal")
            self.treino_pausar_solicitado = False
            self.mostrar_carregamento("Treinando modelo Mobile SAM...")
            threading.Thread(target=self._treinar_mobile_sam_threaded, daemon=True).start()

    def _treinar_yolo_threaded(self):
        caminho_proj = self.project_data.get("caminho", "")
        continuar = self.var_modo_treinamento.get() == "continuar"
        pasta_checkpoint = self.entry_pasta_treinamento.get().strip() if continuar else ""

        try:
            epochs = int(self.entry_epochs.get().strip())
        except ValueError:
            epochs = 50

        try:
            checkpoint = self.localizar_checkpoint_treinamento(pasta_checkpoint) if continuar else None
            output_path, paused = train_yolo(
                caminho_proj, epochs, continuar, checkpoint,
                should_pause=lambda: self.treino_pausar_solicitado,
                on_model=lambda model: setattr(self, "modelo_treinamento_ativo", model),
                log=lambda mensagem: self.after(0, lambda: self.log_treinamento(mensagem)),
            )
            self.pasta_ultimo_treinamento = output_path
            mensagem = (
                f"Treinamento pausado. Para continuar, selecione a pasta:\n{output_path}"
                if paused else "Treinamento do modelo YOLO finalizado com sucesso!"
            )
            self.after(0, lambda: self._finalizar_treino_com_sucesso(mensagem))
        except Exception as erro:
            erro_msg = str(erro)
            self.after(0, lambda: self._finalizar_treino_com_erro(
                f"Falha no treinamento YOLO: {erro_msg}"
            ))
        return

        # Validação simples de rótulos
        if not os.path.exists(pasta_annotations) or not os.listdir(pasta_annotations):
            self.after(0,
                       lambda: self._finalizar_treino_com_erro("Nenhuma anotação foi encontrada para o treinamento."))
            return

        try:
            if continuar:
                checkpoint = self.localizar_checkpoint_treinamento(pasta_checkpoint)
                if not checkpoint:
                    self.after(0, lambda: self._finalizar_treino_com_erro(
                        "Nenhum arquivo last.pt foi encontrado na pasta selecionada."
                    ))
                    return
                self.after(0, lambda: self.log_treinamento(
                    f"[YOLO] Continuando a partir de: {checkpoint}"
                ))
                model = YOLO(checkpoint)
            else:
                self.after(0, lambda: self.log_treinamento(
                    "[YOLO] Carregando modelo pré-treinado yolov8n.pt..."
                ))
                model = YOLO("yolov8n.pt")

            self.modelo_treinamento_ativo = model

            def verificar_pausa(treinador):
                if self.treino_pausar_solicitado:
                    treinador.stop = True

            model.add_callback("on_train_epoch_end", verificar_pausa)

            classes_encontradas = set()
            for f in os.listdir(pasta_annotations):
                if f.endswith(".xml"):
                    try:
                        tree = ET.parse(os.path.join(pasta_annotations, f))
                        for obj in tree.getroot().findall("object"):
                            classes_encontradas.add(obj.find("name").text)
                    except Exception:
                        pass

            lista_classes = sorted(list(classes_encontradas)) or ["Objeto"]

            pasta_frames = os.path.join(caminho_proj, "frames")
            sub_treino = pasta_frames if os.path.exists(pasta_frames) else caminho_proj

            yaml_path = os.path.join(caminho_proj, "dataset_yolo.yaml")
            with open(yaml_path, "w", encoding="utf-8") as yf:
                yf.write(f"path: {os.path.abspath(caminho_proj)}\n")
                yf.write(f"train: {os.path.abspath(sub_treino)}\n")
                yf.write(f"val: {os.path.abspath(sub_treino)}\n")
                yf.write("names:\n")
                for i, cname in enumerate(lista_classes):
                    yf.write(f"  {i}: '{cname}'\n")

            self.after(0, lambda: self.log_treinamento(f"[YOLO] Iniciando fit com {epochs} épocas..."))

            # Execução do Treinamento YOLO (workers=0 evita erros de multiprocessamento em Threads GUI)
            parametros_treino = {
                "data": yaml_path,
                "epochs": epochs,
                "imgsz": 640,
                "workers": 0,
                "project": os.path.join(caminho_proj, "runs"),
                "name": "yolo_train_results",
                "verbose": False
            }
            if continuar:
                parametros_treino["resume"] = True
            self.pasta_ultimo_treinamento = os.path.join(caminho_proj, "runs", "yolo_train_results")
            model.train(**parametros_treino)

            if self.treino_pausar_solicitado:
                self.after(0, lambda: self._finalizar_treino_com_sucesso(
                    "Treinamento pausado. Para continuar, selecione a pasta:\n"
                    f"{self.pasta_ultimo_treinamento}"
                ))
                return

            self.after(0, lambda: self.log_treinamento(
                f"[YOLO] Treinamento concluído com sucesso! Resultados salvos em 'runs/yolo_train_results'."))
            self.after(0,
                       lambda: self._finalizar_treino_com_sucesso("Treinamento do modelo YOLO finalizado com sucesso!"))

        except Exception as e:
            err_msg = str(e)
            self.after(0, lambda: self._finalizar_treino_com_erro(f"Falha no treinamento YOLO: {err_msg}"))

    def _obter_checkpoint_base_mobile_sam(self, caminho_proj):
        caminho_checkpoint = os.path.join(caminho_proj, "mobile_sam.pt")
        if not os.path.exists(caminho_checkpoint):
            url_checkpoint = (
                "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt"
            )
            self.after(0, lambda: self.log_treinamento(
                "[MOBILE SAM] Baixando checkpoint pré-treinado..."
            ))
            urllib.request.urlretrieve(url_checkpoint, caminho_checkpoint)
        return caminho_checkpoint

    def _carregar_amostras_mobile_sam(self, caminho_proj):
        pasta_annotations = os.path.join(caminho_proj, "annotations")
        diretorio_imagens = os.path.join(caminho_proj, "frames")
        if not os.path.isdir(diretorio_imagens):
            diretorio_imagens = caminho_proj

        amostras = []
        if not os.path.isdir(pasta_annotations):
            return amostras

        for nome_rotulo in os.listdir(pasta_annotations):
            if not nome_rotulo.lower().endswith(".json"):
                continue
            nome_base = os.path.splitext(nome_rotulo)[0]
            nome_imagem = next(
                (nome_base + ext for ext in EXTENSOES_IMAGEM
                 if os.path.exists(os.path.join(diretorio_imagens, nome_base + ext))),
                None
            )
            if not nome_imagem:
                continue
            try:
                with open(os.path.join(pasta_annotations, nome_rotulo), "r", encoding="utf-8") as arquivo:
                    dados = json.load(arquivo)
                for forma in dados.get("shapes", []):
                    pontos = forma.get("points", [])
                    if len(pontos) >= 3:
                        amostras.append((
                            os.path.join(diretorio_imagens, nome_imagem),
                            [[float(x), float(y)] for x, y in pontos]
                        ))
            except (OSError, ValueError, TypeError):
                continue
        return amostras

    def _treinar_mobile_sam_threaded(self):
        caminho_proj = self.project_data.get("caminho", "")
        continuar = self.var_modo_treinamento.get() == "continuar"
        pasta_checkpoint = self.entry_pasta_treinamento.get().strip() if continuar else ""
        try:
            epochs = max(1, int(self.entry_epochs.get().strip()))
        except ValueError:
            epochs = 50

        try:
            checkpoint = self.localizar_checkpoint_treinamento(pasta_checkpoint) if continuar else None
            output_path, paused = train_mobile_sam(
                caminho_proj, epochs, checkpoint,
                should_pause=lambda: self.treino_pausar_solicitado,
                on_model=lambda model: setattr(self, "modelo_treinamento_ativo", model),
                log=lambda mensagem: self.after(0, lambda: self.log_treinamento(mensagem)),
            )
            mensagem = (
                f"Treinamento MobileSAM pausado. Checkpoint salvo em:\n{output_path}"
                if paused else f"Treinamento MobileSAM concluído. Melhor modelo salvo em:\n{output_path}"
            )
            self.after(0, lambda: self._finalizar_treino_com_sucesso(mensagem))
        except Exception as erro:
            erro_msg = str(erro)
            self.after(0, lambda: self._finalizar_treino_com_erro(
                f"Falha no treinamento MobileSAM: {erro_msg}"
            ))
        return
        if not MOBILE_SAM_DISPONIVEL:
            self.after(0, lambda: self._finalizar_treino_com_erro(
                "MobileSAM e suas dependências não estão instalados no ambiente."
            ))
            return

        try:
            epochs = max(1, int(self.entry_epochs.get().strip()))
        except ValueError:
            epochs = 50

        amostras = self._carregar_amostras_mobile_sam(caminho_proj)
        if not amostras:
            self.after(0, lambda: self._finalizar_treino_com_erro(
                "Nenhum par imagem/JSON com polígonos válidos foi encontrado para o MobileSAM."
            ))
            return

        try:
            dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
            caminho_saida = os.path.join(caminho_proj, "runs", "mobile_sam_train")
            os.makedirs(caminho_saida, exist_ok=True)
            caminho_continuacao = self.entry_pasta_treinamento.get().strip()
            checkpoint = self.localizar_checkpoint_treinamento(caminho_continuacao) \
                if self.var_modo_treinamento.get() == "continuar" else None
            if checkpoint is None:
                checkpoint = self._obter_checkpoint_base_mobile_sam(caminho_proj)

            modelo = build_sam_vit_t(checkpoint=checkpoint).to(dispositivo)
            self.modelo_treinamento_ativo = modelo
            modelo.image_encoder.eval()
            for parametro in modelo.image_encoder.parameters():
                parametro.requires_grad = False
            modelo.prompt_encoder.train()
            modelo.mask_decoder.train()
            parametros = list(modelo.prompt_encoder.parameters()) + list(modelo.mask_decoder.parameters())
            otimizador = torch.optim.AdamW(parametros, lr=1e-5, weight_decay=1e-4)
            transformacao = ResizeLongestSide(modelo.image_encoder.img_size)
            melhor_loss = float("inf")
            caminho_last = os.path.join(caminho_saida, "last.pt")
            caminho_best = os.path.join(caminho_saida, "best.pt")

            self.after(0, lambda: self.log_treinamento(
                f"[MOBILE SAM] Fine-tuning real em {dispositivo}: {len(amostras)} polígonos, {epochs} épocas."
            ))

            for epoca in range(epochs):
                if self.treino_pausar_solicitado:
                    break
                perda_epoca = 0.0
                for caminho_imagem, pontos in amostras:
                    if self.treino_pausar_solicitado:
                        break
                    imagem_bgr = cv2.imread(caminho_imagem)
                    if imagem_bgr is None:
                        raise FileNotFoundError(f"Não foi possível carregar a imagem: {caminho_imagem}")
                    imagem_np = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2RGB)
                    altura, largura = imagem_np.shape[:2]
                    mascara_np = np.zeros((altura, largura), dtype="uint8")
                    cv2.fillPoly(mascara_np, [np.array(pontos, dtype="int32")], 1)
                    imagem_transformada = transformacao.apply_image(imagem_np)
                    imagem_tensor = torch.as_tensor(imagem_transformada, device=dispositivo).permute(2, 0, 1).float()
                    mascara_transformada = cv2.resize(
                        mascara_np, (imagem_transformada.shape[1], imagem_transformada.shape[0]),
                        interpolation=cv2.INTER_NEAREST
                    )
                    mascara_tensor = torch.as_tensor(mascara_transformada, device=dispositivo).float()
                    mascara_tensor = F.pad(
                        mascara_tensor,
                        (0, modelo.image_encoder.img_size - mascara_tensor.shape[1],
                         0, modelo.image_encoder.img_size - mascara_tensor.shape[0])
                    )[None, None]
                    mascara_alvo = F.interpolate(mascara_tensor, (256, 256), mode="nearest")
                    caixa = [min(x for x, _ in pontos), min(y for _, y in pontos),
                             max(x for x, _ in pontos), max(y for _, y in pontos)]
                    caixa = transformacao.apply_boxes(
                        np.array([caixa]), (altura, largura)
                    )
                    caixa_tensor = torch.as_tensor(caixa, dtype=torch.float32, device=dispositivo)

                    otimizador.zero_grad(set_to_none=True)
                    with torch.no_grad():
                        embedding = modelo.image_encoder(modelo.preprocess(imagem_tensor)[None])
                    sparse, densa = modelo.prompt_encoder(points=None, boxes=caixa_tensor, masks=None)
                    logits, _ = modelo.mask_decoder(
                        image_embeddings=embedding,
                        image_pe=modelo.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse,
                        dense_prompt_embeddings=densa,
                        multimask_output=False
                    )
                    alvo = F.interpolate(mascara_alvo, logits.shape[-2:], mode="nearest")
                    perda_bce = F.binary_cross_entropy_with_logits(logits, alvo)
                    probabilidades = torch.sigmoid(logits)
                    intersecao = (probabilidades * alvo).sum()
                    perda_dice = 1 - (2 * intersecao + 1) / (probabilidades.sum() + alvo.sum() + 1)
                    perda = perda_bce + perda_dice
                    perda.backward()
                    otimizador.step()
                    perda_epoca += perda.item()

                perda_media = perda_epoca / max(1, len(amostras))
                torch.save(modelo.state_dict(), caminho_last)
                if perda_media < melhor_loss:
                    melhor_loss = perda_media
                    torch.save(modelo.state_dict(), caminho_best)
                self.after(0, lambda epoca=epoca, perda_media=perda_media: self.log_treinamento(
                    f"[MOBILE SAM] Época {epoca + 1}/{epochs} - loss: {perda_media:.5f}"
                ))

            pausado = self.treino_pausar_solicitado
            mensagem = (
                f"Treinamento MobileSAM pausado. Checkpoint salvo em:\n{caminho_last}"
                if pausado else
                f"Treinamento MobileSAM concluído. Melhor modelo salvo em:\n{caminho_best}"
            )
            self.after(0, lambda: self._finalizar_treino_com_sucesso(mensagem))
        except Exception as erro:
            erro_msg = str(erro)
            self.after(0, lambda: self._finalizar_treino_com_erro(
                f"Falha no treinamento MobileSAM: {erro_msg}"
            ))

    def _finalizar_treino_com_sucesso(self, mensagem):
        self.esconder_carregamento()
        self.btn_iniciar_treino.configure(state="normal")
        self.btn_pausar_treino.configure(state="disabled", text="⏸ Pausar Treinamento")
        self.modelo_treinamento_ativo = None
        messagebox.showinfo("Sucesso", mensagem)

    def _finalizar_treino_com_erro(self, mensagem):
        self.esconder_carregamento()
        self.btn_iniciar_treino.configure(state="normal")
        self.btn_pausar_treino.configure(state="disabled", text="⏸ Pausar Treinamento")
        self.modelo_treinamento_ativo = None
        self.log_treinamento(f"[ERRO] {mensagem}")
        messagebox.showerror("Erro no Treinamento", mensagem)

    # ==================== EXTRAÇÃO EM THREAD ====================
    def _executar_preparacao_frames_video_threaded(self):
        caminho_proj = self.project_data.get("caminho", "")
        if not caminho_proj:
            self.after(0, self.esconder_carregamento)
            return

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

        pasta_frames = os.path.join(caminho_proj, "frames")
        extract_frames(self.arquivo_selecionado, pasta_frames, step_frames, self.total_frames)

        def finalizar_carregamento():
            self.esconder_carregamento()
            self.carregar_lista_amostras_para_rotulo()

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