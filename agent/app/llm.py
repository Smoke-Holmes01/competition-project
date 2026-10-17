from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
import re
import threading
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, Iterable, List, Optional, Protocol

import httpx

from app.config import Settings


TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "match_uploaded_image",
            "description": "调用模块一视觉匹配工具，识别上传斗拱图片并返回 obj_id、模型 URL 和置信度。",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "为什么需要调用视觉匹配工具。",
                    }
                },
                "required": ["reason"],
            },
        },
    }
]


class LLMError(RuntimeError):
    pass


class ChatClient(Protocol):
    @property
    def configured(self) -> bool:
        ...

    async def chat(
        self,
        messages: Iterable[Dict[str, Any]],
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> "ChatResult":
        ...

    async def stream_chat(
        self,
        messages: Iterable[Dict[str, Any]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> AsyncIterator[str]:
        ...


@dataclass
class ChatResult:
    content: str
    tool_calls: List[Dict[str, Any]]
    raw: Dict[str, Any]


def extract_tool_calls(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    choices = response.get("choices") or []
    if not choices:
        return []
    message = choices[0].get("message") or {}
    tool_calls = message.get("tool_calls") or []
    if not isinstance(tool_calls, list):
        return []
    return tool_calls


def extract_content(response: Dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content") or ""
    if isinstance(content, list):
        return "".join(str(item.get("text", "")) if isinstance(item, dict) else str(item) for item in content)
    return str(content)


def extract_delta_content(response: Dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if not choices:
        return ""
    delta = choices[0].get("delta") or {}
    content = delta.get("content") or ""
    if isinstance(content, list):
        return "".join(str(item.get("text", "")) if isinstance(item, dict) else str(item) for item in content)
    return str(content)


def build_image_data_url(
    image_bytes: bytes,
    *,
    filename: str = "upload.png",
    content_type: str = "application/octet-stream",
) -> str:
    mime_type = content_type if content_type and content_type != "application/octet-stream" else ""
    if not mime_type:
        mime_type = mimetypes.guess_type(filename)[0] or "image/png"
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def normalize_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(part for part in parts if part)
    return str(content)


class SiliconFlowChatClient:
    def __init__(self, settings: Settings):
        self.settings = settings

    @property
    def configured(self) -> bool:
        return self.settings.llm_configured

    async def chat(
        self,
        messages: Iterable[Dict[str, Any]],
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> ChatResult:
        if not self.configured or self.settings.chat_completions_url is None:
            raise LLMError("LLM 未配置：请设置 SILICONFLOW_BASE_URL 和 SILICONFLOW_MODEL")

        payload: Dict[str, Any] = {
            "model": self.settings.siliconflow_model,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        headers = {"Content-Type": "application/json"}
        if self.settings.siliconflow_api_key.strip():
            headers["Authorization"] = f"Bearer {self.settings.siliconflow_api_key.strip()}"

        try:
            async with httpx.AsyncClient(timeout=self.settings.request_timeout_seconds) as client:
                response = await client.post(
                    self.settings.chat_completions_url,
                    headers=headers,
                    content=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                )
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPStatusError as exc:
            raise LLMError(f"LLM API HTTP {exc.response.status_code}: {exc.response.text[:500]}") from exc
        except Exception as exc:
            raise LLMError(f"LLM API 调用失败: {exc}") from exc

        return ChatResult(
            content=extract_content(data),
            tool_calls=extract_tool_calls(data),
            raw=data,
        )

    async def stream_chat(
        self,
        messages: Iterable[Dict[str, Any]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> AsyncIterator[str]:
        if not self.configured or self.settings.chat_completions_url is None:
            raise LLMError("LLM 未配置：请设置 SILICONFLOW_BASE_URL 和 SILICONFLOW_MODEL")

        payload: Dict[str, Any] = {
            "model": self.settings.siliconflow_model,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }

        headers = {"Content-Type": "application/json"}
        if self.settings.siliconflow_api_key.strip():
            headers["Authorization"] = f"Bearer {self.settings.siliconflow_api_key.strip()}"

        try:
            async with httpx.AsyncClient(timeout=self.settings.request_timeout_seconds) as client:
                async with client.stream(
                    "POST",
                    self.settings.chat_completions_url,
                    headers=headers,
                    content=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                ) as response:
                    try:
                        response.raise_for_status()
                    except httpx.HTTPStatusError as exc:
                        body = (await exc.response.aread()).decode("utf-8", errors="replace")
                        raise LLMError(f"LLM API HTTP {exc.response.status_code}: {body[:500]}") from exc

                    async for raw_line in response.aiter_lines():
                        line = raw_line.strip()
                        if not line or not line.startswith("data:"):
                            continue
                        payload_text = line.removeprefix("data:").strip()
                        if payload_text == "[DONE]":
                            break
                        try:
                            data = json.loads(payload_text)
                        except json.JSONDecodeError:
                            continue
                        token = extract_delta_content(data)
                        if token:
                            yield token
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f"LLM API 流式调用失败: {exc}") from exc

    async def list_models(self) -> Dict[str, Any]:
        if not self.settings.models_url:
            raise LLMError("SILICONFLOW_BASE_URL 未配置")

        headers = {}
        if self.settings.siliconflow_api_key.strip():
            headers["Authorization"] = f"Bearer {self.settings.siliconflow_api_key.strip()}"

        try:
            async with httpx.AsyncClient(timeout=self.settings.request_timeout_seconds) as client:
                response = await client.get(self.settings.models_url, headers=headers)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as exc:
            raise LLMError(f"Models API HTTP {exc.response.status_code}: {exc.response.text[:500]}") from exc
        except Exception as exc:
            raise LLMError(f"Models API 调用失败: {exc}") from exc

    async def vision_chat(
        self,
        *,
        text: str,
        image_bytes: bytes,
        filename: str = "upload.png",
        content_type: str = "application/octet-stream",
        system_prompt: str = "",
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> ChatResult:
        if not self.configured:
            raise LLMError("LLM 未配置：请设置 SILICONFLOW_BASE_URL 和 SILICONFLOW_MODEL")

        data_url = build_image_data_url(image_bytes, filename=filename, content_type=content_type)
        messages: List[Dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        )
        return await self.chat(messages, temperature=temperature, max_tokens=max_tokens)


class LocalTransformersChatClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._load_lock = threading.Lock()
        self._tokenizer: Any = None
        self._model: Any = None

    @property
    def configured(self) -> bool:
        return self.settings.local_llm_configured

    async def chat(
        self,
        messages: Iterable[Dict[str, Any]],
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> ChatResult:
        if not self.configured:
            raise LLMError("本地 LLM 未配置：请设置 LOCAL_MODEL_PATH")

        message_list = list(messages)
        content = await asyncio.to_thread(
            self._chat_sync,
            message_list,
            tools,
            temperature,
            max_tokens,
        )
        return ChatResult(content=content, tool_calls=[], raw={"provider": "local"})

    async def stream_chat(
        self,
        messages: Iterable[Dict[str, Any]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> AsyncIterator[str]:
        result = await self.chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if result.content:
            yield result.content

    def _ensure_loaded(self) -> tuple[Any, Any]:
        if self._tokenizer is not None and self._model is not None:
            return self._tokenizer, self._model

        with self._load_lock:
            if self._tokenizer is not None and self._model is not None:
                return self._tokenizer, self._model

            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except Exception as exc:
                raise LLMError(
                    "本地 LLM 依赖未安装：请在 agent 虚拟环境中安装 requirements-local.txt"
                ) from exc

            model_path = str(self.settings.local_model_path)
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    local_files_only=True,
                )
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype="auto",
                    device_map="auto",
                    trust_remote_code=True,
                    local_files_only=True,
                )
                model.eval()
            except Exception as exc:
                raise LLMError(f"本地 LLM 加载失败({model_path}): {exc}") from exc

            self._tokenizer = tokenizer
            self._model = model
            return tokenizer, model

    def _chat_sync(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        tokenizer, model = self._ensure_loaded()
        prompt = self._build_prompt(tokenizer, messages, tools)

        try:
            import torch
        except Exception as exc:
            raise LLMError(
                "本地 LLM 依赖未安装：请在 agent 虚拟环境中安装 requirements-local.txt"
            ) from exc

        inputs = tokenizer(prompt, return_tensors="pt")
        device = getattr(model, "device", None)
        if device is not None:
            inputs = {key: value.to(device) for key, value in inputs.items()}

        generate_kwargs: Dict[str, Any] = {
            **inputs,
            "max_new_tokens": min(max_tokens, 512),
            "do_sample": temperature > 0,
            "repetition_penalty": 1.12,
            "no_repeat_ngram_size": 8,
        }
        if temperature > 0:
            generate_kwargs["temperature"] = temperature
            generate_kwargs["top_p"] = 0.9
        eos_token_ids = self._build_eos_token_ids(tokenizer)
        if eos_token_ids:
            generate_kwargs["eos_token_id"] = eos_token_ids[0] if len(eos_token_ids) == 1 else eos_token_ids
        if tokenizer.pad_token_id is not None:
            generate_kwargs["pad_token_id"] = tokenizer.pad_token_id
        elif tokenizer.eos_token_id is not None:
            generate_kwargs["pad_token_id"] = tokenizer.eos_token_id

        try:
            with torch.inference_mode():
                output_ids = model.generate(**generate_kwargs)
        except Exception as exc:
            raise LLMError(f"本地 LLM 推理失败: {exc}") from exc

        input_length = inputs["input_ids"].shape[-1]
        generated_ids = output_ids[0][input_length:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return self._clean_local_response(generated_text)

    def _build_eos_token_ids(self, tokenizer: Any) -> List[int]:
        token_ids: List[int] = []
        if tokenizer.eos_token_id is not None:
            token_ids.append(int(tokenizer.eos_token_id))
        convert_tokens_to_ids = getattr(tokenizer, "convert_tokens_to_ids", None)
        if callable(convert_tokens_to_ids):
            for token in ("<|im_end|>", "<|endoftext|>"):
                try:
                    token_id = convert_tokens_to_ids(token)
                except Exception:
                    continue
                if isinstance(token_id, int) and token_id >= 0 and token_id not in token_ids:
                    token_ids.append(token_id)
        return token_ids

    def _build_prompt(
        self,
        tokenizer: Any,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
    ) -> str:
        prompt_messages = [
            {
                "role": "system",
                "content": (
                    "/no_think\n"
                    "只输出给用户看的最终中文答案。不要输出 Thinking Process、思考过程、"
                    "<think> 标签、分析步骤、prompt、tool_result、knowledge_context 或 JSON 字段名。"
                ),
            },
            *list(messages),
        ]
        if tools:
            tool_names = ", ".join(
                str((tool.get("function") or {}).get("name", ""))
                for tool in tools
                if isinstance(tool, dict)
            )
            prompt_messages = [
                {
                    "role": "system",
                    "content": (
                        "/no_think\n"
                        "可用工具：" + tool_names + "。如果用户问题需要工具，请直接说明需要调用对应工具。"
                    ),
                },
                *prompt_messages,
            ]

        apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
        if callable(apply_chat_template):
            try:
                template_messages = [
                    {**message, "content": normalize_text_content(message.get("content", ""))}
                    for message in prompt_messages
                ]
                return str(
                    apply_chat_template(
                        template_messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                )
            except TypeError:
                try:
                    template_messages = [
                        {**message, "content": normalize_text_content(message.get("content", ""))}
                        for message in prompt_messages
                    ]
                    return str(
                        apply_chat_template(
                            template_messages,
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                    )
                except Exception:
                    pass
            except Exception:
                pass

        rendered = []
        for message in prompt_messages:
            role = message.get("role", "user")
            content = normalize_text_content(message.get("content", ""))
            rendered.append(f"{role}: {content}")
        rendered.append("assistant:")
        return "\n".join(rendered)

    def _clean_local_response(self, content: str) -> str:
        text = content.strip()
        if not text:
            return text

        text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
        if "</think>" in text.lower():
            text = re.split(r"</think>", text, flags=re.IGNORECASE)[-1].strip()

        final_markers = [
            "Final Answer:",
            "Final answer:",
            "最终答案：",
            "最终回答：",
            "答案：",
        ]
        for marker in final_markers:
            if marker in text:
                text = text.split(marker, 1)[1].strip()
                break

        if text.lower().startswith("thinking process"):
            parts = re.split(
                r"(?:\n|^)\s*(?:Final Answer|Final answer|最终答案|最终回答|答案)\s*[:：]\s*",
                text,
                maxsplit=1,
            )
            if len(parts) == 2 and parts[1].strip():
                text = parts[1].strip()
            else:
                return "我可以回答斗拱的结构、朝代差异、构件名称、历史背景，也可以在你上传图片后调用视觉匹配工具生成模型识别结果。"

        blocked_markers = [
            "Analyze the Request",
            "Drafting the Response",
            "Constraint Check",
            "knowledge_context",
            "tool_result",
        ]
        if any(marker in text for marker in blocked_markers):
            return "我可以回答斗拱的结构、朝代差异、构件名称、历史背景，也可以在你上传图片后调用视觉匹配工具生成模型识别结果。"

        return self._collapse_repeated_blocks(text).strip()

    def _collapse_repeated_blocks(self, content: str) -> str:
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", content) if part.strip()]
        if not paragraphs:
            return content.strip()

        collapsed: List[str] = []
        seen_counts: Dict[str, int] = {}
        for paragraph in paragraphs:
            normalized = re.sub(r"\s+", "", paragraph)
            if collapsed and re.sub(r"\s+", "", collapsed[-1]) == normalized:
                continue
            seen_counts[normalized] = seen_counts.get(normalized, 0) + 1
            if seen_counts[normalized] > 1 and len(normalized) > 20:
                continue
            collapsed.append(paragraph)

        if len(collapsed) == 1:
            sentences = re.split(r"(?<=[。！？!?])", collapsed[0])
            deduped_sentences: List[str] = []
            seen_sentences: set[str] = set()
            for sentence in sentences:
                normalized = re.sub(r"\s+", "", sentence)
                if not normalized:
                    continue
                if normalized in seen_sentences and len(normalized) > 12:
                    continue
                seen_sentences.add(normalized)
                deduped_sentences.append(sentence)
            return "".join(deduped_sentences)

        return "\n\n".join(collapsed)
