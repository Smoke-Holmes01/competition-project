"""Application projections beside LangGraph checkpoints in the same local SQLite database."""

import asyncio
import json
from pathlib import Path
from uuid import uuid4

import aiosqlite

from app.schemas import Asset


class Store:
    def __init__(self, db: aiosqlite.Connection):
        self.db = db
        self.write_lock = asyncio.Lock()

    async def setup(self):
        self.db.row_factory = aiosqlite.Row
        await self.db.executescript("""
            PRAGMA journal_mode=WAL;
            PRAGMA busy_timeout=5000;
            CREATE TABLE IF NOT EXISTS conversations(id TEXT PRIMARY KEY);
            CREATE TABLE IF NOT EXISTS turns(
                id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL,
                message TEXT NOT NULL, assets TEXT NOT NULL, status TEXT NOT NULL,
                result TEXT, model_messages TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS turns_conversation ON turns(conversation_id);
            CREATE TABLE IF NOT EXISTS assets(
                id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL, batch_id TEXT NOT NULL,
                record TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS assets_conversation ON assets(conversation_id);
            UPDATE turns SET status='interrupted' WHERE status='running';
        """)

    async def one(self, sql: str, args=()):
        async with self.db.execute(sql, args) as cursor:
            return await cursor.fetchone()

    async def rows(self, sql: str, args=()):
        async with self.db.execute(sql, args) as cursor:
            return await cursor.fetchall()

    async def ensure_conversation(self, conversation_id: str):
        await self.db.execute("INSERT OR IGNORE INTO conversations VALUES (?)", (conversation_id,))

    async def begin(self, conversation_id: str, message: str, assets: list[Asset]) -> str:
        turn_id = uuid4().hex
        async with self.write_lock:
            await self.ensure_conversation(conversation_id)
            await self.db.execute(
                "INSERT INTO turns(id,conversation_id,message,assets,status) VALUES (?,?,?,?,?)",
                (turn_id, conversation_id, message, json.dumps([a.model_dump() for a in assets]), "running"),
            )
        return turn_id

    async def finish(self, turn_id: str, result: dict, model_messages: list[dict]):
        await self.db.execute(
            "UPDATE turns SET status='complete',result=?,model_messages=? WHERE id=?",
            (json.dumps(result, ensure_ascii=False), json.dumps(model_messages, ensure_ascii=False), turn_id),
        )

    async def fail(self, turn_id: str, status: str, message: str):
        await self.db.execute(
            "UPDATE turns SET status=?,result=? WHERE id=?",
            (status, json.dumps({"answer": message}, ensure_ascii=False), turn_id),
        )

    async def history_groups(self, conversation_id: str) -> list[list[dict]]:
        rows = await self.rows(
            "SELECT model_messages FROM turns WHERE conversation_id=? AND status='complete' ORDER BY rowid",
            (conversation_id,),
        )
        return [json.loads(row[0]) for row in rows if row[0]]

    async def conversation(self, conversation_id: str) -> dict | None:
        if not await self.one("SELECT id FROM conversations WHERE id=?", (conversation_id,)):
            return None
        rows = await self.rows(
            "SELECT * FROM turns WHERE conversation_id=? ORDER BY rowid", (conversation_id,)
        )
        messages = []
        last_result = None
        for row in rows:
            messages.append(
                {
                    "role": "user",
                    "content": row["message"],
                    "assets": json.loads(row["assets"]),
                    "time": row["created_at"],
                    "status": row["status"],
                }
            )
            result = json.loads(row["result"]) if row["result"] else None
            if result:
                messages.append(
                    {
                        "role": "assistant",
                        "content": result["answer"],
                        "assets": [],
                        "time": row["created_at"],
                        "status": row["status"],
                    }
                )
                if row["status"] == "complete":
                    last_result = result
            elif row["status"] in {"running", "interrupted"}:
                messages.append(
                    {
                        "role": "assistant",
                        "content": "上次请求未完成，请重新发送。",
                        "assets": [],
                        "time": row["created_at"],
                        "status": row["status"],
                    }
                )
        return {"conversation_id": conversation_id, "messages": messages, "last_result": last_result}

    async def add_asset(self, asset: Asset):
        await self.db.execute(
            "INSERT INTO assets VALUES (?,?,?,?)",
            (asset.id, asset.conversation_id, asset.batch_id, asset.model_dump_json()),
        )

    async def asset(self, asset_id: str) -> Asset | None:
        row = await self.one("SELECT record FROM assets WHERE id=?", (asset_id,))
        return Asset.model_validate_json(row[0]) if row else None

    async def latest_assets(self, conversation_id: str) -> list[Asset]:
        latest = await self.one(
            "SELECT batch_id FROM assets WHERE conversation_id=? ORDER BY rowid DESC LIMIT 1",
            (conversation_id,),
        )
        if not latest:
            return []
        rows = await self.rows(
            "SELECT record FROM assets WHERE conversation_id=? AND batch_id=? ORDER BY rowid",
            (conversation_id, latest[0]),
        )
        return [Asset.model_validate_json(row[0]) for row in rows]

    async def delete_conversation(self, conversation_id: str) -> list[str]:
        async with self.write_lock:
            rows = await self.rows("SELECT id FROM assets WHERE conversation_id=?", (conversation_id,))
            await self.db.execute("BEGIN IMMEDIATE")
            try:
                for table in ("assets", "turns"):
                    await self.db.execute(f"DELETE FROM {table} WHERE conversation_id=?", (conversation_id,))
                await self.db.execute("DELETE FROM conversations WHERE id=?", (conversation_id,))
                await self.db.commit()
            except BaseException:
                await self.db.rollback()
                raise
        return [row[0] for row in rows]


async def open_store(path: Path) -> Store:
    path.parent.mkdir(parents=True, exist_ok=True)
    db = await aiosqlite.connect(path, isolation_level=None)
    store = Store(db)
    await store.setup()
    return store
