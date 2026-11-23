"""Model adapters used by the paper: Whisper and OWSM-CTC v4."""

from typing import Any, List

import torch


class ASRAdapter:
    def load_model(self, model_name: str, device: str) -> Any:
        raise NotImplementedError

    def transcribe(self, model: Any, audio_path: str, *, language: str, task: str, initial_prompt: str | None) -> dict:
        raise NotImplementedError

    def decoder_blocks(self, model: Any) -> List[Any]:
        raise NotImplementedError

    def encoder_blocks(self, model: Any) -> List[Any]:
        raise NotImplementedError

    def hidden_size(self, model: Any) -> int:
        raise NotImplementedError


class WhisperAdapter(ASRAdapter):
    def load_model(self, model_name: str, device: str) -> Any:
        import whisper

        # The paper's "large" column is Whisper large-v1. Avoid the moving
        # upstream `large` alias, which may point at a newer checkpoint.
        checkpoint = "large-v1" if model_name == "large" else model_name
        return whisper.load_model(checkpoint).to(device)

    def transcribe(self, model: Any, audio_path: str, *, language: str, task: str, initial_prompt: str | None) -> dict:
        return model.transcribe(
            audio_path,
            language=language,
            task=task,
            initial_prompt=initial_prompt,
        )

    def decoder_blocks(self, model: Any) -> List[Any]:
        return list(model.decoder.blocks)

    def encoder_blocks(self, model: Any) -> List[Any]:
        return list(model.encoder.blocks)

    def hidden_size(self, model: Any) -> int:
        return int(model.decoder.token_embedding.weight.shape[1])


_WHISPER_TO_ISO3 = {
    "zh": "cmn",
    "sr": "srp",
}


class OWSMCTCAdapter(ASRAdapter):
    """OWSM-CTC v4 adapter used for the paper's Table 4."""

    def load_model(self, model_name: str, device: str) -> Any:
        from espnet2.bin.s2t_inference_ctc import Speech2TextGreedySearch

        return Speech2TextGreedySearch.from_pretrained(
            model_name,
            device=device,
            lang_sym="<eng>",
            task_sym="<asr>",
        )

    def transcribe(self, model: Any, audio_path: str, *, language: str, task: str, initial_prompt: str | None) -> dict:
        import librosa
        import soundfile as sf

        speech, sample_rate = sf.read(audio_path)
        if sample_rate != 16000:
            speech = librosa.resample(speech, orig_sr=sample_rate, target_sr=16000)

        iso3 = _WHISPER_TO_ISO3.get(language, language)
        model.lang_sym = initial_prompt or f"<{iso3}>"
        result = model(speech)
        return {"text": result[0][0]}

    def decoder_blocks(self, model: Any) -> List[Any]:
        return []

    def encoder_blocks(self, model: Any) -> List[Any]:
        return list(model.s2t_model.encoder.encoders)

    def hidden_size(self, model: Any) -> int:
        return int(model.s2t_model.encoder.output_size())


def get_adapter(backend: str) -> ASRAdapter:
    adapters = {
        "whisper": WhisperAdapter,
        "owsm-ctc": OWSMCTCAdapter,
    }
    try:
        return adapters[backend.lower()]()
    except KeyError as exc:
        raise ValueError(f"Unsupported paper backend: {backend}") from exc
