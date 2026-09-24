import os

from domain.project import TOPOLOGIA_CAIXAS, TOPOLOGIAS_SEGMENTACAO


def inferir_topologia(caminho: str) -> str:
    pasta_annotations = os.path.join(caminho, "annotations")
    if not os.path.isdir(pasta_annotations):
        return TOPOLOGIA_CAIXAS

    arquivos = os.listdir(pasta_annotations)
    if any(nome.lower().endswith(".json") for nome in arquivos):
        return TOPOLOGIAS_SEGMENTACAO[0]
    if any(nome.lower().endswith(".xml") for nome in arquivos):
        return TOPOLOGIA_CAIXAS
    return TOPOLOGIA_CAIXAS


def validar_pasta_projeto(caminho: str) -> bool:
    return bool(caminho and os.path.isdir(caminho))
