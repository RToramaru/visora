from dataclasses import dataclass
from typing import Any, Dict


TOPOLOGIA_CAIXAS = "Bounding Boxes"
TOPOLOGIAS_SEGMENTACAO = ("Instance Mask", "Semantic Mask")


@dataclass
class ProjectData:
    nome: str
    caminho: str
    topologia: str = TOPOLOGIA_CAIXAS
    volume: str = "0 itens"
    modificado: str = ""

    def to_dict(self) -> Dict[str, str]:
        return {
            "nome": self.nome,
            "caminho": self.caminho,
            "topologia": self.topologia,
            "volume": self.volume,
            "modificado": self.modificado,
        }


def normalizar_project_data(data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "nome": data.get("nome", "Projeto sem nome"),
        "caminho": data.get("caminho", ""),
        "topologia": data.get("topologia", TOPOLOGIA_CAIXAS),
        "volume": data.get("volume", "0 itens"),
        "modificado": data.get("modificado", ""),
    }


def validar_project_data(data: Dict[str, Any]) -> None:
    if not data.get("nome", "").strip():
        raise ValueError("O nome do projeto é obrigatório.")
    if not data.get("caminho", "").strip():
        raise ValueError("A pasta do projeto é obrigatória.")
    if data.get("topologia") not in (TOPOLOGIA_CAIXAS, *TOPOLOGIAS_SEGMENTACAO):
        raise ValueError("A topologia do projeto não é suportada.")
