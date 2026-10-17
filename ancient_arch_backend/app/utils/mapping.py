from app.config import settings


def group_to_obj_id(group_name: str) -> int:
    return int(group_name.split("_")[-1])


def obj_id_to_path(obj_id: int) -> str:
    return f"models/{obj_id}.obj"


def obj_id_to_url(obj_id: int) -> str:
    return public_asset_url(obj_id_to_path(obj_id))


def obj_id_to_glb_path(obj_id: int) -> str:
    return f"models_glb/{obj_id}.glb"


def obj_id_to_glb_url(obj_id: int) -> str:
    return public_asset_url(obj_id_to_glb_path(obj_id))


def public_asset_url(path: str) -> str:
    if not path.startswith("/"):
        path = f"/{path}"
    public_base = settings.normalized_public_asset_base_url
    if public_base:
        return f"{public_base}{path}"
    return path

