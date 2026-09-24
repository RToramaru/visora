import customtkinter as ctk


class ReviewPanel:
    def __init__(self, owner, frame):
        self.owner = owner
        self.frame = frame
        self.build()

    def build(self):
        self.owner.rev_left_frame = ctk.CTkScrollableFrame(
            self.frame, width=340, fg_color="#111622", corner_radius=6
        )
        self.owner.rev_left_frame.pack(side="left", fill="y", padx=10, pady=10)
        self.owner.rev_right_frame = ctk.CTkFrame(
            self.frame, fg_color="#111622", corner_radius=6
        )
        self.owner.rev_right_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        ctk.CTkLabel(
            self.owner.rev_right_frame, text="🔍 INSPEÇÃO VISUAL DA AMOSTRA",
            font=ctk.CTkFont(size=12, weight="bold"), text_color="#38bdf8"
        ).pack(pady=10)
        self.owner.rev_canvas = ctk.CTkCanvas(
            self.owner.rev_right_frame, bg="#05070a", highlightthickness=0
        )
        self.owner.rev_canvas.pack(fill="both", expand=True, padx=10, pady=10)
        self.owner.rev_canvas.bind("<Configure>", self.owner.atualizar_preview_revisao)
        self.owner.rev_info_lbl = ctk.CTkLabel(
            self.owner.rev_right_frame, text="Selecione um frame ao lado para auditar.",
            font=ctk.CTkFont(size=11), text_color="#94a3b8"
        )
        self.owner.rev_info_lbl.pack(pady=5)
        self.owner.btn_excluir_lote = ctk.CTkButton(
            self.owner.rev_right_frame, text="🗑️ Remover o(s) selecionado(s)", width=300, height=35,
            fg_color="#ef4444", hover_color="#dc2626", font=ctk.CTkFont(size=12, weight="bold"),
            command=self.owner.excluir_anotacoes_em_lote
        )
        self.owner.btn_excluir_lote.pack(pady=(5, 15))
