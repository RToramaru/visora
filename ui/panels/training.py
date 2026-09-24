import customtkinter as ctk


class TrainingPanel:
    def __init__(self, owner, frame):
        self.owner = owner
        self.frame = frame
        self.build()

    def build(self):
        ctk.CTkLabel(
            self.frame, text="🤖 Treinamento do Modelo",
            font=ctk.CTkFont(size=15, weight="bold"), text_color="#a855f7", anchor="w"
        ).pack(fill="x", padx=20, pady=(20, 10))

        topologia = self.owner.project_data.get("topologia", "Bounding Boxes")
        modelo_nome = "YOLO (Bounding Boxes)" if topologia == "Bounding Boxes" else "Mobile SAM (Polígonos / Segmentação)"
        ctk.CTkLabel(
            self.frame,
            text=f"A topologia selecionada é '{topologia}'. O treinamento utilizará o algoritmo: {modelo_nome}.",
            font=ctk.CTkFont(size=12), text_color="#94a3b8", anchor="w"
        ).pack(fill="x", padx=20, pady=(0, 15))

        params_frame = ctk.CTkFrame(self.frame, fg_color="#111622", corner_radius=6)
        params_frame.pack(fill="x", padx=20, pady=10)

        ctk.CTkLabel(
            params_frame, text="Número de Épocas:",
            font=ctk.CTkFont(size=11, weight="bold"), text_color="#e2e8f0"
        ).grid(row=0, column=0, padx=15, pady=10, sticky="w")
        self.owner.entry_epochs = ctk.CTkEntry(params_frame, width=80, height=28, fg_color="#080a0f")
        self.owner.entry_epochs.insert(0, "50")
        self.owner.entry_epochs.grid(row=0, column=1, padx=10, pady=10, sticky="w")

        self.owner.var_modo_treinamento = ctk.StringVar(value="novo")
        ctk.CTkLabel(
            params_frame, text="Modo:", font=ctk.CTkFont(size=11, weight="bold"), text_color="#e2e8f0"
        ).grid(row=1, column=0, padx=15, pady=10, sticky="w")
        ctk.CTkRadioButton(
            params_frame, text="Novo treinamento", variable=self.owner.var_modo_treinamento,
            value="novo", command=self.owner.atualizar_modo_treinamento
        ).grid(row=1, column=1, padx=10, pady=10, sticky="w")
        ctk.CTkRadioButton(
            params_frame, text="Continuar treinamento", variable=self.owner.var_modo_treinamento,
            value="continuar", command=self.owner.atualizar_modo_treinamento
        ).grid(row=1, column=2, padx=10, pady=10, sticky="w")

        self.owner.entry_pasta_treinamento = ctk.CTkEntry(
            params_frame, width=360, height=28, fg_color="#080a0f",
            placeholder_text="Selecione a pasta que contém o checkpoint last.pt"
        )
        self.owner.entry_pasta_treinamento.grid(row=2, column=1, padx=10, pady=(0, 10), sticky="ew")
        self.owner.btn_pasta_treinamento = ctk.CTkButton(
            params_frame, text="Selecionar pasta", width=130, height=28,
            command=self.owner.selecionar_pasta_treinamento
        )
        self.owner.btn_pasta_treinamento.grid(row=2, column=2, padx=10, pady=(0, 10), sticky="w")
        self.owner.atualizar_modo_treinamento()

        ctk.CTkLabel(
            self.frame, text="Logs do Treinamento:",
            font=ctk.CTkFont(size=12, weight="bold"), text_color="#94a3b8", anchor="w"
        ).pack(fill="x", padx=20, pady=(10, 2))
        self.owner.textbox_log = ctk.CTkTextbox(
            self.frame, height=200, fg_color="#05070a", text_color="#22c55e",
            font=ctk.CTkFont(family="Consolas", size=11)
        )
        self.owner.textbox_log.pack(fill="both", expand=True, padx=20, pady=(0, 15))

        self.owner.btn_iniciar_treino = ctk.CTkButton(
            self.frame, text="▶ Iniciar Treinamento", width=220, height=38,
            fg_color="#a855f7", hover_color="#9333ea", font=ctk.CTkFont(size=12, weight="bold"),
            command=self.owner.iniciar_treinamento_modelo
        )
        self.owner.btn_iniciar_treino.pack(padx=20, pady=(0, 20), anchor="w")
        self.owner.btn_pausar_treino = ctk.CTkButton(
            self.frame, text="⏸ Pausar Treinamento", width=220, height=38,
            fg_color="#d97706", hover_color="#b45309", font=ctk.CTkFont(size=12, weight="bold"),
            state="disabled", command=self.owner.solicitar_pausa_treinamento
        )
        self.owner.btn_pausar_treino.pack(padx=20, pady=(0, 20), anchor="w")
