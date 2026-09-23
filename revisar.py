import customtkinter as ctk

class ReviewWindow(ctk.CTkToplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("REVISAR DATASET")
        self.geometry("900x600")
        self.configure(fg_color="#0b0e14")

        ctk.CTkLabel(self, text="🔍 REVISÃO DE AMOSTRAS E ANOTAÇÕES", font=ctk.CTkFont(size=16, weight="bold"), text_color="#f8fafc").pack(pady=20)

        # Galeria / Grid
        self.grid_frame = ctk.CTkScrollableFrame(self, fg_color="#11151c", border_color="#1e2430", border_width=1)
        self.grid_frame.pack(fill="both", expand=True, padx=20, pady=(0, 20))

        # Mockup de Itens na galeria
        for i in range(6):
            card = ctk.CTkFrame(self.grid_frame, fg_color="#181c24", width=180, height=140, corner_radius=6)
            card.grid(row=i//3, column=i%3, padx=15, pady=15)
            ctk.CTkLabel(card, text=f"Amostra #{i+1}\n[ Status: OK ]", text_color="#94a3b8").pack(expand=True)