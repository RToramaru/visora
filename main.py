import json
import os
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk

# Importa a janela do Workspace/Editor do arquivo separado editor.py
from editor import WorkspaceStudio

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

ARQUIVO_RECENTES = "recentes.json"


class CardTipoProblema(ctk.CTkFrame):
    def __init__(self, parent, icon_text, title, description, value, current_var, command=None):
        super().__init__(
            parent,
            fg_color="#181c24",
            border_color="#262c36",
            border_width=1,
            corner_radius=6,
            cursor="hand2"
        )
        self.value = value
        self.current_var = current_var
        self.command = command

        self.lbl_icon = ctk.CTkLabel(self, text=icon_text, font=ctk.CTkFont(size=18), text_color="#3b82f6")
        self.lbl_icon.pack(anchor="nw", padx=12, pady=(12, 5))

        self.lbl_title = ctk.CTkLabel(self, text=title, font=ctk.CTkFont(size=12, weight="bold"), text_color="#ffffff",
                                      anchor="w")
        self.lbl_title.pack(anchor="w", padx=12, pady=(0, 2))

        self.lbl_desc = ctk.CTkLabel(self, text=description, font=ctk.CTkFont(size=10), text_color="#8b949e",
                                     justify="left", wraplength=130, anchor="w")
        self.lbl_desc.pack(anchor="w", padx=12, pady=(0, 12))

        for widget in (self, self.lbl_icon, self.lbl_title, self.lbl_desc):
            widget.bind("<Button-1>", self._on_click)

    def _on_click(self, event=None):
        self.current_var.set(self.value)
        if self.command:
            self.command()

    def update_state(self):
        if self.current_var.get() == self.value:
            self.configure(border_color="#3b82f6", fg_color="#1e2532", border_width=2)
        else:
            self.configure(border_color="#262c36", fg_color="#181c24", border_width=1)


class VisoraAppHome(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("SPECTRA CV / VISORA")
        self.geometry("1100x700")
        self.minsize(950, 600)
        self.configure(fg_color="#0b0e14")

        self.projetos_recentes = self.carregar_recentes()[:5]
        self.overlay_canvas = None

        self.build_header()
        self.build_main_content()

    def carregar_recentes(self):
        if os.path.exists(ARQUIVO_RECENTES):
            try:
                with open(ARQUIVO_RECENTES, "r", encoding="utf-8") as f:
                    dados = json.load(f)
                    return dados if isinstance(dados, list) else []
            except Exception:
                pass
        return []

    def salvar_recentes(self):
        with open(ARQUIVO_RECENTES, "w", encoding="utf-8") as f:
            json.dump(self.projetos_recentes[:5], f, indent=4, ensure_ascii=False)

    def adicionar_aos_recentes(self, nome, caminho, topologia, volume="0 itens"):
        novo = {
            "nome": nome,
            "caminho": caminho,
            "volume": volume,
            "topologia": topologia,
            "modificado": f"Modificado hoje às {datetime.now().strftime('%H:%M')}"
        }
        self.projetos_recentes = [p for p in self.projetos_recentes if p["nome"] != nome]
        self.projetos_recentes.insert(0, novo)
        self.projetos_recentes = self.projetos_recentes[:5]
        self.salvar_recentes()
        return novo

    def abrir_workspace_studio(self, project_data):
        """Abre o Editor e FECHA a janela principal antiga."""
        self.withdraw()  # Esconde a janela atual
        editor_win = WorkspaceStudio(project_data=project_data)

        # Garante o fechamento total da aplicação ao fechar o editor
        editor_win.protocol("WM_DELETE_WINDOW", lambda: self.fechar_tudo(editor_win))
        self.destroy()  # Destroi a janela principal antiga da memória

    def fechar_tudo(self, window):
        window.destroy()

    def mostrar_overlay(self):
        if not self.overlay_canvas:
            self.overlay_canvas = tk.Canvas(self, bg="#000000", highlightthickness=0)
            self.overlay_canvas.place(relx=0, rely=0, relwidth=1, relheight=1)

    def esconder_overlay(self):
        if self.overlay_canvas:
            self.overlay_canvas.destroy()
            self.overlay_canvas = None

    def build_header(self):
        header = ctk.CTkFrame(self, height=40, corner_radius=0, fg_color="#0b0e14")
        header.pack(fill="x", side="top")

        lbl_brand = ctk.CTkLabel(header, text="SPECTRA CV", font=ctk.CTkFont(size=12, weight="bold"),
                                 text_color="#475569")
        lbl_brand.pack(side="left", padx=20, pady=10)

        lbl_status = ctk.CTkLabel(header, text="[ Workspace Local // Sem projeto ativo ]", font=ctk.CTkFont(size=11),
                                  text_color="#334155")
        lbl_status.pack(side="left", padx=5)

    def build_main_content(self):
        container = ctk.CTkFrame(self, fg_color="transparent")
        container.pack(fill="both", expand=True, padx=50, pady=20)

        lbl_title = ctk.CTkLabel(container, text="Iniciar projeto", font=ctk.CTkFont(size=32, weight="bold"),
                                 text_color="#f8fafc")
        lbl_title.pack(anchor="w", pady=(10, 2))

        lbl_sub = ctk.CTkLabel(
            container,
            text="Crie um novo projeto ou abra um projeto existente para continuar seu trabalho de curadoria e calibração dimensional.",
            font=ctk.CTkFont(size=13), text_color="#64748b"
        )
        lbl_sub.pack(anchor="w", pady=(0, 25))

        cards_frame = ctk.CTkFrame(container, fg_color="transparent")
        cards_frame.pack(fill="x", pady=(0, 30))
        cards_frame.columnconfigure((0, 1), weight=1, pad=15)

        card_open = ctk.CTkFrame(cards_frame, fg_color="#11151c", border_color="#1e2430", border_width=1,
                                 corner_radius=8)
        card_open.grid(row=0, column=0, sticky="ew", ipady=15)

        ctk.CTkLabel(card_open, text="📁", font=("Arial", 22), text_color="#3b82f6").pack(anchor="w", padx=20,
                                                                                         pady=(15, 5))
        ctk.CTkLabel(card_open, text="Abrir Projeto", font=ctk.CTkFont(size=18, weight="bold"),
                     text_color="#ffffff").pack(anchor="w", padx=20)
        ctk.CTkLabel(card_open, text="Abra um projeto existente e continue seu trabalho", font=ctk.CTkFont(size=12),
                     text_color="#64748b").pack(anchor="w", padx=20, pady=(2, 10))

        btn_procurar = ctk.CTkButton(
            card_open, text="Procurar arquivos →", fg_color="transparent", text_color="#3b82f6",
            hover=False, anchor="w", font=ctk.CTkFont(size=12, weight="bold"), command=self.abrir_projeto_existente
        )
        btn_procurar.pack(anchor="w", padx=15, pady=(0, 10))

        card_create = ctk.CTkFrame(cards_frame, fg_color="#11151c", border_color="#1e2430", border_width=1,
                                   corner_radius=8)
        card_create.grid(row=0, column=1, sticky="ew", ipady=15)

        ctk.CTkLabel(card_create, text="⛶", font=("Arial", 22), text_color="#3b82f6").pack(anchor="w", padx=20,
                                                                                           pady=(15, 5))
        ctk.CTkLabel(card_create, text="Criar Novo Projeto", font=ctk.CTkFont(size=18, weight="bold"),
                     text_color="#ffffff").pack(anchor="w", padx=20)
        ctk.CTkLabel(card_create, text="Configure um novo projeto e comece a trabalhar com seus dados",
                     font=ctk.CTkFont(size=12), text_color="#64748b").pack(anchor="w", padx=20, pady=(2, 10))

        btn_iniciar = ctk.CTkButton(
            card_create, text="Iniciar setup →", fg_color="transparent", text_color="#3b82f6",
            hover=False, anchor="w", font=ctk.CTkFont(size=12, weight="bold"), command=self.abrir_modal_criar_projeto
        )
        btn_iniciar.pack(anchor="w", padx=15, pady=(0, 10))

        sec_title = ctk.CTkLabel(container, text="🕒 PROJETOS RECENTES (MÁX 5)",
                                 font=ctk.CTkFont(size=11, weight="bold"), text_color="#475569")
        sec_title.pack(anchor="w", pady=(0, 10))

        self.table_main_frame = ctk.CTkFrame(container, fg_color="#11151c", border_color="#1e2430", border_width=1,
                                             corner_radius=8)
        self.table_main_frame.pack(fill="both", expand=True)

        self.render_recentes_table()

    def render_recentes_table(self):
        for w in self.table_main_frame.winfo_children():
            w.destroy()

        header_row = ctk.CTkFrame(self.table_main_frame, fg_color="#0d1117", height=35, corner_radius=0)
        header_row.pack(fill="x", side="top")

        header_row.columnconfigure(0, weight=4, minsize=260)
        header_row.columnconfigure(1, weight=2, minsize=160)
        header_row.columnconfigure(2, weight=3, minsize=200)
        header_row.columnconfigure(3, weight=1, minsize=100)

        ctk.CTkLabel(header_row, text="IDENTIFICADOR DE DATASET", font=ctk.CTkFont(size=10, weight="bold"),
                     text_color="#475569", anchor="w").grid(row=0, column=0, sticky="ew", padx=(20, 10), pady=8)
        ctk.CTkLabel(header_row, text="VOLUME", font=ctk.CTkFont(size=10, weight="bold"), text_color="#475569",
                     anchor="w").grid(row=0, column=1, sticky="ew", padx=10, pady=8)
        ctk.CTkLabel(header_row, text="TOPOLOGIA / FORMATO", font=ctk.CTkFont(size=10, weight="bold"),
                     text_color="#475569", anchor="w").grid(row=0, column=2, sticky="ew", padx=10, pady=8)
        ctk.CTkLabel(header_row, text="AÇÃO", font=ctk.CTkFont(size=10, weight="bold"), text_color="#475569",
                     anchor="e").grid(row=0, column=3, sticky="ew", padx=(10, 25), pady=8)

        scroll_container = ctk.CTkScrollableFrame(self.table_main_frame, fg_color="transparent", corner_radius=0)
        scroll_container.pack(fill="both", expand=True)

        scroll_container.columnconfigure(0, weight=4, minsize=260)
        scroll_container.columnconfigure(1, weight=2, minsize=160)
        scroll_container.columnconfigure(2, weight=3, minsize=200)
        scroll_container.columnconfigure(3, weight=1, minsize=100)

        if not self.projetos_recentes:
            ctk.CTkLabel(scroll_container, text="Nenhum projeto recente encontrado.", font=ctk.CTkFont(size=12),
                         text_color="#475569").grid(row=0, column=0, columnspan=4, pady=30)
            return

        for i, proj in enumerate(self.projetos_recentes[:5]):
            row_index = i * 2

            f_id = ctk.CTkFrame(scroll_container, fg_color="transparent")
            f_id.grid(row=row_index, column=0, sticky="ew", padx=(15, 10), pady=8)
            ctk.CTkLabel(f_id, text=f"• {proj['nome']}", font=ctk.CTkFont(size=12, weight="bold"), text_color="#e2e8f0",
                         anchor="w").pack(anchor="w")
            ctk.CTkLabel(f_id, text=proj.get('modificado', ''), font=ctk.CTkFont(size=10), text_color="#64748b",
                         anchor="w").pack(anchor="w")

            ctk.CTkLabel(scroll_container, text=proj.get('volume', '0 itens'), font=ctk.CTkFont(size=11),
                         text_color="#94a3b8", anchor="w").grid(row=row_index, column=1, sticky="ew", padx=10, pady=8)

            f_top = ctk.CTkFrame(scroll_container, fg_color="transparent")
            f_top.grid(row=row_index, column=2, sticky="ew", padx=10, pady=8)
            ctk.CTkLabel(f_top, text=f" {proj.get('topologia', 'Bounding Boxes')} ", font=ctk.CTkFont(size=10),
                         fg_color="#1e293b", text_color="#38bdf8", corner_radius=4).pack(side="left")

            f_act = ctk.CTkFrame(scroll_container, fg_color="transparent")
            f_act.grid(row=row_index, column=3, sticky="ew", padx=(10, 15), pady=8)
            btn_load = ctk.CTkButton(
                f_act, text="Carregar >", width=75, height=26, fg_color="#1e293b", hover_color="#334155",
                font=ctk.CTkFont(size=11), command=lambda p=proj: self.carregar_projeto(p)
            )
            btn_load.pack(side="right")

            sep = ctk.CTkFrame(scroll_container, height=1, fg_color="#1e2430")
            sep.grid(row=row_index + 1, column=0, columnspan=4, sticky="ew", pady=(2, 2))

    def abrir_modal_criar_projeto(self):
        self.mostrar_overlay()

        modal = ctk.CTkToplevel(self)
        modal.title("Criar novo projeto")
        modal.geometry("620x540")
        modal.configure(fg_color="#12161f")
        modal.grab_set()
        modal.resizable(False, False)

        def ao_fechar_modal():
            self.esconder_overlay()
            modal.destroy()

        modal.protocol("WM_DELETE_WINDOW", ao_fechar_modal)

        self.update_idletasks()
        x = self.winfo_x() + (self.winfo_width() // 2) - (620 // 2)
        y = self.winfo_y() + (self.winfo_height() // 2) - (540 // 2)
        modal.geometry(f"+{x}+{y}")

        body = ctk.CTkFrame(modal, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=25, pady=20)

        ctk.CTkLabel(body, text="⛶ CONFIGURAÇÃO INICIAL", font=ctk.CTkFont(size=10, weight="bold"),
                     text_color="#2563eb").pack(anchor="w")
        ctk.CTkLabel(body, text="Criar novo projeto", font=ctk.CTkFont(size=22, weight="bold"),
                     text_color="#ffffff").pack(anchor="w", pady=(2, 0))
        ctk.CTkLabel(body, text="Configure as definições fundamentais e o diretório de trabalho.",
                     font=ctk.CTkFont(size=11), text_color="#64748b").pack(anchor="w", pady=(0, 20))

        f_name_head = ctk.CTkFrame(body, fg_color="transparent")
        f_name_head.pack(fill="x", pady=(0, 4))
        ctk.CTkLabel(f_name_head, text="Nome do projeto *", font=ctk.CTkFont(size=11, weight="bold"),
                     text_color="#cbd5e1").pack(side="left")

        entry_nome = ctk.CTkEntry(
            body, placeholder_text="Digite o nome do projeto...",
            fg_color="#0b0e14", border_color="#2563eb", border_width=1, text_color="#ffffff", height=35
        )
        entry_nome.pack(fill="x", pady=(0, 15))

        f_path_head = ctk.CTkFrame(body, fg_color="transparent")
        f_path_head.pack(fill="x", pady=(0, 4))
        ctk.CTkLabel(f_path_head, text="Local do projeto", font=ctk.CTkFont(size=11, weight="bold"),
                     text_color="#cbd5e1").pack(side="left")

        path_row = ctk.CTkFrame(body, fg_color="transparent")
        path_row.pack(fill="x", pady=(0, 15))

        entry_path = ctk.CTkEntry(
            path_row, placeholder_text="Selecione o diretório do projeto...",
            fg_color="#0b0e14", border_color="#1e2430", text_color="#ffffff", height=35
        )
        entry_path.pack(side="left", fill="x", expand=True, padx=(0, 10))

        def selecionar_pasta():
            pasta = filedialog.askdirectory()
            if pasta:
                entry_path.delete(0, tk.END)
                entry_path.insert(0, pasta)

        btn_pasta = ctk.CTkButton(path_row, text="📁 Selecionar pasta", fg_color="#1e293b", hover_color="#334155",
                                  height=35, command=selecionar_pasta)
        btn_pasta.pack(side="right")

        var_tipo = tk.StringVar(value="Bounding Boxes")

        cards_row = ctk.CTkFrame(body, fg_color="transparent")
        cards_row.pack(fill="x", pady=(0, 20))
        cards_row.columnconfigure((0, 1, 2), weight=1, pad=10)

        cards = []

        def on_card_selected():
            for c in cards:
                c.update_state()

        c1 = CardTipoProblema(cards_row, "⛶", "Detecção de objetos", "Localize e identifique objetos.",
                              "Bounding Boxes", var_tipo, on_card_selected)
        c1.grid(row=0, column=0, sticky="nsew")
        c2 = CardTipoProblema(cards_row, "⬡", "Segmentação Instância", "Identifique cada objeto.", "Instance Mask",
                              var_tipo, on_card_selected)
        c2.grid(row=0, column=1, sticky="nsew")
        c3 = CardTipoProblema(cards_row, "▦", "Segmentação Semântica", "Classifique cada pixel.", "Semantic Mask",
                              var_tipo, on_card_selected)
        c3.grid(row=0, column=2, sticky="nsew")

        cards.extend([c1, c2, c3])
        on_card_selected()

        footer = ctk.CTkFrame(body, fg_color="transparent")
        footer.pack(fill="x", side="bottom")

        def submeter_projeto():
            nome = entry_nome.get().strip()
            caminho = entry_path.get().strip()
            topologia = var_tipo.get()

            if not nome:
                messagebox.showerror("Erro de Validação", "O nome do projeto é obrigatório.")
                return

            novo_proj = self.adicionar_aos_recentes(nome, caminho, topologia)
            ao_fechar_modal()
            self.abrir_workspace_studio(novo_proj)

        btn_criar = ctk.CTkButton(footer, text="Criar projeto →", fg_color="#2563eb", hover_color="#1d4ed8", height=36,
                                  font=ctk.CTkFont(weight="bold"), command=submeter_projeto)
        btn_criar.pack(side="right", padx=(10, 0))

        btn_cancelar = ctk.CTkButton(footer, text="Cancelar", fg_color="transparent", text_color="#94a3b8",
                                     hover_color="#1e293b", height=36, command=ao_fechar_modal)
        btn_cancelar.pack(side="right")

    def abrir_projeto_existente(self):
        pasta = filedialog.askdirectory(title="Selecione a pasta do projeto existente")
        if pasta:
            nome = os.path.basename(pasta)
            novo_proj = self.adicionar_aos_recentes(nome, pasta, "Bounding Boxes")
            self.abrir_workspace_studio(novo_proj)

    def carregar_projeto(self, proj_data):
        proj_atualizado = self.adicionar_aos_recentes(
            proj_data["nome"],
            proj_data["caminho"],
            proj_data.get("topologia", "Bounding Boxes"),
            proj_data.get("volume", "0 itens")
        )
        self.abrir_workspace_studio(proj_atualizado)


if __name__ == "__main__":
    app = VisoraAppHome()
    app.mainloop()