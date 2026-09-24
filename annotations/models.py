from typing import Dict, List, Union


Annotation = Dict[str, Union[str, int, List[List[float]]]]


def annotation_extension(topologia: str) -> str:
    return ".xml" if topologia == "Bounding Boxes" else ".json"
