from pathlib import Path
from fastapi import Request


def path_to_url(path: str | Path) -> str:
    path = Path(path)
    path_str = path.as_posix()

    if "outputs/" in path_str:
        return "/" + path_str[path_str.index("outputs/"):]
    if path_str.startswith("outputs"):
        return "/" + path_str
    if "models/" in path_str:
        return "/" + path_str[path_str.index("models/"):]
    if path_str.startswith("models"):
        return "/" + path_str

    return "/" + path_str


def to_absolute_url(request: Request, relative_url: str) -> str:
    return str(request.base_url).rstrip("/") + relative_url