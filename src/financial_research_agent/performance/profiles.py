from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LocalModelProfile:
    id: str
    label: str
    hardware_class: str
    minimum_vram_gb: int
    suggested_context_tokens: int
    model_guidance: tuple[str, ...]
    runtime_notes: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _require_text("id", self.id))
        object.__setattr__(self, "label", _require_text("label", self.label))
        object.__setattr__(
            self,
            "hardware_class",
            _require_text("hardware_class", self.hardware_class),
        )
        if self.minimum_vram_gb < 0:
            raise ValueError("minimum_vram_gb must be non-negative")
        if self.suggested_context_tokens <= 0:
            raise ValueError("suggested_context_tokens must be positive")
        object.__setattr__(self, "model_guidance", _text_tuple(self.model_guidance))
        object.__setattr__(self, "runtime_notes", _text_tuple(self.runtime_notes))

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "label": self.label,
            "hardware_class": self.hardware_class,
            "minimum_vram_gb": self.minimum_vram_gb,
            "suggested_context_tokens": self.suggested_context_tokens,
            "model_guidance": list(self.model_guidance),
            "runtime_notes": list(self.runtime_notes),
        }


def default_local_model_profiles() -> tuple[LocalModelProfile, ...]:
    return (
        LocalModelProfile(
            id="small",
            label="Small local profile",
            hardware_class="CPU or 8-12 GB VRAM",
            minimum_vram_gb=0,
            suggested_context_tokens=8_192,
            model_guidance=("Prefer 7B-9B instruct models in Q4-class quantization.",),
            runtime_notes=(
                "Use smaller context windows and keep retrieval snippets short.",
                "Good for UI smoke tests and light extraction, not deep multi-document synthesis.",
            ),
        ),
        LocalModelProfile(
            id="medium",
            label="Medium local profile",
            hardware_class="12-24 GB VRAM",
            minimum_vram_gb=12,
            suggested_context_tokens=16_384,
            model_guidance=("Prefer 12B-14B instruct models in Q4-class quantization.",),
            runtime_notes=(
                "Suitable for ordinary chat, cited answers, and single-company research flows.",
                "Keep background concurrency at one unless the endpoint is proven stable.",
            ),
        ),
        LocalModelProfile(
            id="strong",
            label="Strong local profile",
            hardware_class="24+ GB VRAM",
            minimum_vram_gb=24,
            suggested_context_tokens=32_768,
            model_guidance=("Prefer 24B-32B instruct models in Q4-class quantization.",),
            runtime_notes=(
                "Suitable for larger filing snippets and longer synthesis prompts.",
                "Tool calling and JSON quality still depend on model/template/runtime support.",
            ),
        ),
    )


def _require_text(name: str, value: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    text = value.strip()
    if text == "":
        raise ValueError(f"{name} is required")
    return text


def _text_tuple(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(_require_text(f"value[{index}]", value) for index, value in enumerate(values))
