import json
import os
from datetime import datetime
from typing import Any, Dict, List

from domain.project import normalizar_project_data


class RecentProjectsStore:
    def __init__(self, path: str = "recentes.json", limit: int = 5):
        self.path = path
        self.limit = limit

    def load(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.path):
            return []
        try:
            with open(self.path, "r", encoding="utf-8") as arquivo:
                dados = json.load(arquivo)
            if not isinstance(dados, list):
                return []
            return [normalizar_project_data(projeto) for projeto in dados[:self.limit]]
        except (OSError, json.JSONDecodeError):
            return []

    def save(self, projects: List[Dict[str, Any]]) -> None:
        with open(self.path, "w", encoding="utf-8") as arquivo:
            json.dump(projects[:self.limit], arquivo, indent=4, ensure_ascii=False)

    def upsert(self, nome: str, caminho: str, topologia: str, volume: str = "0 itens") -> Dict[str, Any]:
        projeto = {
            "nome": nome,
            "caminho": caminho,
            "volume": volume,
            "topologia": topologia,
            "modificado": f"Modificado hoje às {datetime.now().strftime('%H:%M')}",
        }
        projetos = [projeto_atual for projeto_atual in self.load() if projeto_atual.get("nome") != nome]
        projetos.insert(0, projeto)
        self.save(projetos)
        return projeto
