"""
providers.py
============

A minimal abstraction layer that hides the differences between
• Azure OpenAI
• Siemens (OpenAI-compatible endpoint)
• Ollama (local or remote)

The only public entry-point other modules need is `get_llm(model_cfg)`,
which returns an object that exposes a single method:

    chat(model_name, messages, temperature=0.1, stream=False) -> dict
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, List

from dotenv import load_dotenv, dotenv_values
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AzureOpenAI, OpenAI
from ollama import Client as OllamaClient

# ----------------------------------------------------------------------
# 1.  Model meta-data
# ----------------------------------------------------------------------
class ModelProvider(Enum):
    AZURE = auto()
    SIEMENS = auto()
    OLLAMA = auto()


@dataclass
class ModelConfig:
    """
    name         – model identifier understood by the backend
    provider     – one of ModelProvider.*
    temperature  – default temperature for generation
    host         – (only for Ollama) full URL incl. scheme/port,
                   e.g. 'http://192.168.178.187:11434'
                   If None → fall back to env var OLLAMA_HOST or
                   to 'http://127.0.0.1:11434'
    """
    name: str
    provider: ModelProvider
    temperature: float = 0.1
    host: str | None = None


# ----------------------------------------------------------------------
# 2.  Common interface every concrete backend implements
# ----------------------------------------------------------------------
class BaseLLM:
    """A tiny facade that normalises `.chat()` across providers."""

    def chat(
        self,
        model_name: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        stream: bool = False,
    ) -> Dict[str, Any]:
        raise NotImplementedError


# ----------------------------------------------------------------------
# 3.  Concrete provider implementations
# ----------------------------------------------------------------------
# --- load env once ----------------------------------------------------
load_dotenv()
_cfg = dotenv_values()

# ---------- 3a. Azure OpenAI -----------------------------------------
_token_provider = get_bearer_token_provider(
    DefaultAzureCredential(),
    "https://cognitiveservices.azure.com/.default",
)


class AzureLLM(BaseLLM):
    def __init__(self) -> None:
        self._client = AzureOpenAI(
            azure_endpoint=_cfg["AZURE_OPENAI_ENDPOINT"],
            azure_ad_token_provider=_token_provider,
            api_version=_cfg["OPENAI_API_VERSION"],
        )

    def chat(
        self,
        model_name: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        stream: bool = False,
    ) -> Dict[str, Any]:
        rsp = self._client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            stream=stream,
            max_completion_tokens=25_000
        )
        first_choice = rsp.choices[0].message.content
        return {
            "message": {"content": first_choice},
            "usage": getattr(rsp, "usage", None),
        }


# ---------- 3b. Siemens OpenAI-compatible ----------------------------
class SiemensLLM(BaseLLM):
    def __init__(self) -> None:
        self._client = OpenAI(
            api_key=_cfg["SIEMENS_API_KEY"],
            base_url=_cfg["SIEMENS_API_BASE"],
        )

    def chat(
        self,
        model_name: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        stream: bool = False,
    ) -> Dict[str, Any]:
        rsp = self._client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            stream=stream,
        )
        first_choice = rsp.choices[0].message.content
        return {
            "message": {"content": first_choice},
            "usage": getattr(rsp, "usage", None),
        }


# ---------- 3c. Ollama ----------------------------------------------
class OllamaLLM(BaseLLM):
    """
    Works with any Ollama server; host can be passed explicitly or via
    the environment variable OLLAMA_HOST.
    """

    def __init__(self, host: str | None = None) -> None:
        if host is None:
            host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
        self._client = OllamaClient(host=host)

    def chat(
        self,
        model_name: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        stream: bool = False,
    ) -> Dict[str, Any]:
        return self._client.chat(
            model=model_name,
            messages=messages,
            options={"temperature": temperature, "stream": stream},
        )


# ----------------------------------------------------------------------
# 4.  Factory
# ----------------------------------------------------------------------
def get_llm(model: ModelConfig) -> BaseLLM:
    if model.provider == ModelProvider.AZURE:
        return AzureLLM()
    if model.provider == ModelProvider.SIEMENS:
        return SiemensLLM()
    if model.provider == ModelProvider.OLLAMA:
        return OllamaLLM(host=model.host)

    raise ValueError(f"Unsupported provider: {model.provider}")