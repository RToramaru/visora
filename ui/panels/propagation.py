import customtkinter as ctk


class PropagationPanel:
    def __init__(self, owner, frame):
        self.owner = owner
        self.frame = frame
        self.build()

    def build(self):
        ctk.CTkLabel(
            self.frame, text="⚡ Escolha o Método de Propagação Baseado nas suas Amostras",
            font=ctk.CTkFont(size=15, weight="bold"), text_color="#38bdf8", anchor="w"
        ).pack(fill="x", padx=10, pady=(15, 5))
        ctk.CTkLabel(
            self.frame,
            text="As imagens que você rotulou na etapa anterior servirão de base para preencher automaticamente os quadros restantes do seu dataset.",
            font=ctk.CTkFont(size=12), text_color="#94a3b8", justify="left", wraplength=700
        ).pack(fill="x", padx=10, pady=(0, 15))
        self._card(
            "Propagação por Interpolagem / Clonagem de Referências",
            "Utiliza as anotações feitas nas suas amostras-chave e as replica para preencher os frames vazios.",
            "Executar Interpolagem", "molde", "#1e293b", "#334155"
        )
        self._card(
            "Rastreamento Temporal OpenCV (Multi-Tracking)",
            "Emprega algoritmos tradicionais de visão computacional para seguir os pixels e contornos quadro a quadro.",
            "Executar OpenCV Tracking", "opencv", "#1e293b", "#334155"
        )
        if self.owner.project_data.get("topologia", "Bounding Boxes") == "Bounding Boxes":
            self._card(
                "Modelo Ultralytics YOLO (Treinamento Rápido / Inferência Inteligente)",
                "Utiliza YOLO para analisar as imagens e detectar objetos automaticamente.",
                "Executar YOLO AI", "yolo", "#2563eb", "#1d4ed8"
            )

    def _card(self, title, description, button_text, method, color, hover):
        card = ctk.CTkFrame(self.frame, fg_color="#111622", border_width=1,
                            border_color="#2563eb" if method == "yolo" else "#1e293b", corner_radius=6)
        card.pack(fill="x", padx=10, pady=6)
        ctk.CTkLabel(
            card, text=title, font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#38bdf8" if method == "yolo" else "#e2e8f0"
        ).pack(anchor="w", padx=15, pady=(12, 4))
        ctk.CTkLabel(
            card, text=description, font=ctk.CTkFont(size=11), text_color="#94a3b8",
            justify="left", wraplength=680
        ).pack(anchor="w", padx=15, pady=(0, 8))
        ctk.CTkButton(
            card, text=button_text, width=180, height=32, fg_color=color, hover_color=hover,
            font=ctk.CTkFont(size=11, weight="bold"),
            command=lambda: self.owner.executar_modelo_propagacao(method)
        ).pack(anchor="w", padx=15, pady=(0, 12))
