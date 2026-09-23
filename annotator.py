import tkinter as tk
import customtkinter as ctk
from PIL import Image, ImageTk


class AnnotatorWindow(ctk.CTkToplevel):
    def __init__(self, parent, image_path):
        super().__init__(parent)
        self.title("SELECIONAR OBJETO / ANOTAR")
        self.geometry("1100x700")
        self.configure(fg_color="#0b0e14")

        self.image_path = image_path
        self.rectangles = []
        self.start_x = None
        self.start_y = None
        self.current_rect = None

        self.build_ui()
        self.load_image()

    def build_ui(self):
        # Barra superior de ferramentas
        top = ctk.CTkFrame(self, height=45, fg_color="#11151c", corner_radius=0)
        top.pack(fill="x", side="top")

        ctk.CTkLabel(top, text="✏ MODO DE ANOTAÇÃO (Bounding Boxes)", font=ctk.CTkFont(size=12, weight="bold"),
                     text_color="#38bdf8").pack(side="left", padx=15)

        btn_clear = ctk.CTkButton(top, text="Limpar Anotações", fg_color="#ef4444", hover_color="#dc2626", width=120,
                                  height=28, command=self.clear_boxes)
        btn_clear.pack(side="right", padx=15)

        # Canvas de Desenho
        self.canvas_frame = ctk.CTkFrame(self, fg_color="#000000")
        self.canvas_frame.pack(fill="both", expand=True, padx=15, pady=15)

        self.canvas = tk.Canvas(self.canvas_frame, bg="#0b0e14", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_move_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

    def load_image(self):
        try:
            self.pil_img = Image.open(self.image_path)
            self.tk_img = ImageTk.PhotoImage(self.pil_img)
            self.canvas.create_image(0, 0, image=self.tk_img, anchor="nw")
        except Exception as e:
            print(f"Erro ao carregar imagem no anotações: {e}")

    def on_button_press(self, event):
        self.start_x = event.x
        self.start_y = event.y
        self.current_rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y,
                                                         outline="#38bdf8", width=2)

    def on_move_press(self, event):
        cur_x, cur_y = (event.x, event.y)
        self.canvas.coords(self.current_rect, self.start_x, self.start_y, cur_x, cur_y)

    def on_button_release(self, event):
        self.rectangles.append(self.current_rect)

    def clear_boxes(self):
        for r in self.rectangles:
            self.canvas.delete(r)
        self.rectangles.clear()