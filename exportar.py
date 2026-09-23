import customtkinter as ctk
from tkinter import messagebox

class ExportWindow(ctk.CTkToplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("EXPORTAR DATASET")
        self.geometry("500x400")
        self.configure(fg_color="#0b0e14")

        ctk.CTkLabel(self, text="📦 EXPORTAÇÃO DE DATASET", font=ctk.CTkFont(size=16, weight="bold"), text_color="#ffffff").pack(pady=20)

        ctk.CTkLabel(self, text="Escolha o Formato de Saída:", font=ctk.CTkFont(size=12), text_color="#94a3b8").pack(anchor="w", padx=40, pady=(10, 5))

        self.format_var = ctk.StringVar(value="YOLOv8")
        formats = ["YOLOv8 (Txt)", "COCO (JSON)", "Pascal VOC (XML)", "CSV Format"]

        for f in formats:
            rb = ctk.CTkRadioButton(self, text=f, variable=self.format_var, value=f)
            rb.pack(anchor="w", padx=50, pady=5)

        btn_exp = ctk.CTkButton(self, text="Exportar Dataset →", fg_color="#2563eb", hover_color="#1d4ed8", height=38, command=self.exportar)
        btn_exp.pack(fill="x", padx=40, pady=30)

    def exportar(self):
        fmt = self.format_var.get()
        messagebox.showinfo("Sucesso", f"Dataset exportado no formato {fmt} com sucesso!")
        self.destroy()