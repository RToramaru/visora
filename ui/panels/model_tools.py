import customtkinter as ctk


class ModelToolsPanel:
    def __init__(self, owner, frame_visualizar, frame_exportar):
        self.owner = owner
        self.frame_visualizar = frame_visualizar
        self.frame_exportar = frame_exportar
        self._build_visualizacao()
        self._build_exportacao()

    def _build_visualizacao(self):
        ctk.CTkLabel(
            self.frame_visualizar, text="Visualizar resultado do treinamento",
            font=ctk.CTkFont(size=15, weight="bold"), text_color="#22c55e", anchor="w"
        ).pack(fill="x", padx=20, pady=(20, 10))

        controles = ctk.CTkFrame(self.frame_visualizar, fg_color="#111622", corner_radius=6)
        controles.pack(fill="x", padx=20, pady=10)
        controles.columnconfigure(1, weight=1)

        ctk.CTkLabel(controles, text="Modelo treinado:", text_color="#e2e8f0").grid(
            row=0, column=0, padx=15, pady=10, sticky="w"
        )
        self.owner.entry_modelo_visualizacao = ctk.CTkEntry(controles, height=28, fg_color="#080a0f")
        self.owner.entry_modelo_visualizacao.grid(row=0, column=1, padx=10, pady=10, sticky="ew")
        ctk.CTkButton(
            controles, text="Selecionar", width=110, height=28,
            command=self.owner.selecionar_modelo_visualizacao
        ).grid(row=0, column=2, padx=10, pady=10)

        ctk.CTkLabel(controles, text="Tipo de modelo:", text_color="#e2e8f0").grid(
            row=1, column=0, padx=15, pady=10, sticky="w"
        )
        self.owner.var_tipo_visualizacao = ctk.StringVar(
            value="MobileSAM" if self.owner.project_data.get("topologia", "Bounding Boxes") != "Bounding Boxes" else "YOLO"
        )
        ctk.CTkOptionMenu(
            controles, variable=self.owner.var_tipo_visualizacao,
            values=["YOLO", "MobileSAM"], width=150
        ).grid(row=1, column=1, padx=10, pady=10, sticky="w")

        ctk.CTkLabel(controles, text="Imagem para testar:", text_color="#e2e8f0").grid(
            row=2, column=0, padx=15, pady=10, sticky="w"
        )
        self.owner.entry_imagem_visualizacao = ctk.CTkEntry(controles, height=28, fg_color="#080a0f")
        self.owner.entry_imagem_visualizacao.grid(row=2, column=1, padx=10, pady=10, sticky="ew")
        ctk.CTkButton(
            controles, text="Selecionar", width=110, height=28,
            command=self.owner.selecionar_imagem_visualizacao
        ).grid(row=2, column=2, padx=10, pady=10)

        ctk.CTkButton(
            controles, text="Executar inferência", width=180, height=34,
            fg_color="#22c55e", hover_color="#16a34a",
            command=self.owner.executar_inferencia_modelo
        ).grid(row=3, column=1, padx=10, pady=(5, 15), sticky="w")

        self.owner.lbl_resultado_visualizacao = ctk.CTkLabel(
            self.frame_visualizar, text="Selecione um modelo e uma imagem para visualizar as detecções.",
            text_color="#94a3b8"
        )
        self.owner.lbl_resultado_visualizacao.pack(fill="both", expand=True, padx=20, pady=20)

    def _build_exportacao(self):
        ctk.CTkLabel(
            self.frame_exportar, text="Exportar modelo treinado",
            font=ctk.CTkFont(size=15, weight="bold"), text_color="#f59e0b", anchor="w"
        ).pack(fill="x", padx=20, pady=(20, 10))

        controles = ctk.CTkFrame(self.frame_exportar, fg_color="#111622", corner_radius=6)
        controles.pack(fill="x", padx=20, pady=10)
        controles.columnconfigure(1, weight=1)

        ctk.CTkLabel(controles, text="Modelo treinado:", text_color="#e2e8f0").grid(
            row=0, column=0, padx=15, pady=10, sticky="w"
        )
        self.owner.entry_modelo_exportacao = ctk.CTkEntry(controles, height=28, fg_color="#080a0f")
        self.owner.entry_modelo_exportacao.grid(row=0, column=1, padx=10, pady=10, sticky="ew")
        ctk.CTkButton(
            controles, text="Selecionar", width=110, height=28,
            command=self.owner.selecionar_modelo_exportacao
        ).grid(row=0, column=2, padx=10, pady=10)

        ctk.CTkLabel(controles, text="Formato de saída:", text_color="#e2e8f0").grid(
            row=1, column=0, padx=15, pady=10, sticky="w"
        )
        ctk.CTkLabel(controles, text="Tipo de modelo:", text_color="#e2e8f0").grid(
            row=1, column=2, padx=10, pady=10, sticky="w"
        )
        self.owner.var_formato_modelo = ctk.StringVar(value="onnx")
        ctk.CTkOptionMenu(
            controles, variable=self.owner.var_formato_modelo,
            values=["onnx", "pt", "pth", "torchscript", "engine", "coreml", "tflite"], width=180
        ).grid(row=1, column=1, padx=10, pady=10, sticky="w")
        self.owner.var_tipo_exportacao_modelo = ctk.StringVar(
            value="MobileSAM" if self.owner.project_data.get("topologia", "Bounding Boxes") != "Bounding Boxes" else "YOLO"
        )
        ctk.CTkOptionMenu(
            controles, variable=self.owner.var_tipo_exportacao_modelo,
            values=["YOLO", "MobileSAM"], width=140
        ).grid(row=1, column=3, padx=10, pady=10, sticky="w")

        ctk.CTkLabel(controles, text="Pasta de destino:", text_color="#e2e8f0").grid(
            row=2, column=0, padx=15, pady=10, sticky="w"
        )
        self.owner.entry_destino_modelo = ctk.CTkEntry(controles, height=28, fg_color="#080a0f")
        self.owner.entry_destino_modelo.grid(row=2, column=1, padx=10, pady=10, sticky="ew")
        ctk.CTkButton(
            controles, text="Selecionar", width=110, height=28,
            command=self.owner.selecionar_destino_modelo
        ).grid(row=2, column=2, padx=10, pady=10)

        ctk.CTkButton(
            controles, text="Exportar modelo", width=180, height=34,
            fg_color="#f59e0b", hover_color="#d97706",
            command=self.owner.executar_exportacao_modelo
        ).grid(row=3, column=1, padx=10, pady=(5, 15), sticky="w")

        self.owner.lbl_status_exportacao_modelo = ctk.CTkLabel(
            self.frame_exportar,
            text="YOLO usa o exportador Ultralytics; MobileSAM pode ser exportado para ONNX ou mantido em .pt/.pth.",
            text_color="#94a3b8", anchor="w", justify="left"
        )
        self.owner.lbl_status_exportacao_modelo.pack(fill="x", padx=20, pady=10)
