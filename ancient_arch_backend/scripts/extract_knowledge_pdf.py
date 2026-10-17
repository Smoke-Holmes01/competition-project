import argparse
import json
import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from pypdf import PdfReader

from app.config import settings


DYNASTY_TERMS = [
    ("清代", "清代"),
    ("清式", "清代"),
    ("宋代", "宋代"),
    ("宋式", "宋代"),
    ("明代", "明代"),
    ("明式", "明代"),
    ("唐代", "唐代"),
    ("唐式", "唐代"),
    ("辽代", "辽代"),
    ("金代", "金代"),
    ("元代", "元代"),
]


def normalize_text(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_entries(pdf_path: Path) -> list[tuple[int, str]]:
    reader = PdfReader(str(pdf_path))
    text = "\n".join((page.extract_text() or "") for page in reader.pages)
    parts = re.split(r"(?m)(?=^\s*\d+\.)", text)

    entries = []
    for part in parts:
        match = re.match(r"^\s*(\d+)\.(.*)", part, re.S)
        if match:
            entries.append((int(match.group(1)), normalize_text(match.group(2))))
    return entries


def infer_dynasty(text: str) -> str:
    candidates = []
    for term, normalized in DYNASTY_TERMS:
        pos = text.find(term)
        if pos >= 0:
            candidates.append((pos, normalized))
    return sorted(candidates)[0][1] if candidates else "待补充"


def infer_name(text: str, model_id: int) -> str:
    first_sentence = re.split(r"[。\n]", text, maxsplit=1)[0].strip()
    first_sentence = first_sentence.replace("**", "").strip()
    match = re.match(r"^(?:这是|这组模型是|属于)(.+?)(?:，|（|\(|是)", first_sentence)
    if match:
        return match.group(1).strip()
    return first_sentence[:40] if first_sentence else f"斗拱模型 {model_id}"


def build_records(entries: list[tuple[int, str]]) -> tuple[dict[str, dict], dict[str, dict]]:
    if len(entries) != 46:
        raise ValueError(f"Expected 46 knowledge entries, got {len(entries)}")
    numbers = [number for number, _ in entries]
    if numbers != list(range(1, 47)):
        raise ValueError(f"Knowledge entries are not numbered 1..46: {numbers}")

    metadata = {}
    knowledge = {}
    for model_id, text in entries:
        knowledge_id = f"knowledge_{model_id:03d}"
        description = re.sub(r"\s+", " ", text).strip()
        item = {
            "model_id": model_id,
            "group": f"group_{model_id:03d}",
            "knowledge_id": knowledge_id,
            "model_name": infer_name(description, model_id),
            "dynasty": infer_dynasty(description),
            "description": description,
            "source": "知识库.pdf",
        }
        metadata[str(model_id)] = item
        knowledge[knowledge_id] = item.copy()
    return metadata, knowledge


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, default=settings.knowledge_pdf_path)
    parser.add_argument("--metadata-output", type=Path, default=settings.model_metadata_path)
    parser.add_argument("--knowledge-output", type=Path, default=settings.knowledge_path)
    args = parser.parse_args()

    entries = extract_entries(args.pdf)
    metadata, knowledge = build_records(entries)

    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    args.knowledge_output.write_text(json.dumps(knowledge, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(metadata)} metadata entries to {args.metadata_output}")
    print(f"Wrote {len(knowledge)} knowledge entries to {args.knowledge_output}")


if __name__ == "__main__":
    main()
