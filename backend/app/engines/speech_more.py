"""Backends for the newer speech engines (Supertonic, Kyutai, Parler-TTS, MeloTTS...).

Same contract as the ones in ``speech_worker``: ``speak`` returns float samples and a rate.
Each library is imported only inside its backend, in that runtime's own virtualenv.

Some libraries fetch helper models by name the first time they load (a text encoder, a voice
pack). Workers run offline, so ``prepare`` fetches them right after the model downloads, into
the shared Hugging Face cache (``HF_HUB_CACHE``), where the library then finds them.
"""

import re
from pathlib import Path

from speech_base import (
    DEFAULT,
    DESIGNS,
    STARTERS,
    VOICE_REPO,
    Backend,
    device,
    hub_file,
    numpy,
    read_json,
)


def snapshot(repo: str, patterns: list[str] | None = None, offline: bool = False) -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo, allow_patterns=patterns, local_files_only=offline))


def fetch_starters() -> None:
    snapshot(VOICE_REPO, [path for path, _, _ in STARTERS.values()])


def words(name: str) -> str:
    """ "WeatherPresenterCheerful" / "en-Emma_woman" -> readable words."""
    name = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name)
    return re.sub(r"[_\-]+", " ", name).strip()


def bundled_espeak() -> None:
    """Point phonemizer at the espeak-ng library bundled by espeakng-loader (the container
    has no system espeak)."""
    import os

    import espeakng_loader
    from phonemizer.backend.espeak.wrapper import EspeakWrapper

    EspeakWrapper.set_library(espeakng_loader.get_library_path())
    os.environ["ESPEAK_DATA_PATH"] = espeakng_loader.get_data_path()


class Supertonic(Backend):
    """Supertone's Supertonic: tiny ONNX models, very fast on CPU, 10 voices, up to 31
    languages."""

    def __init__(self, model_dir, config):
        super().__init__(model_dir, config)
        from supertonic import TTS
        from supertonic.config import AVAILABLE_MODELS

        name = config.get("source", {}).get("repo", "").split("/")[-1].lower()
        self.tts = TTS(
            model=name if name in AVAILABLE_MODELS else "supertonic-3",
            model_dir=model_dir,
            auto_download=False,
        )

    def voices(self):
        return list(self.tts.voice_style_names)

    def details(self):
        gender = {"F": "Female", "M": "Male"}
        return {
            v: {
                "name": f"{gender.get(v[0], 'Voice')} {v[1:]}",
                "language": "Voices",
                "gender": gender.get(v[0], ""),
            }
            for v in self.voices()
        }

    def speak(self, text, voice, speed, profile=None):
        voices = self.voices()
        style = self.tts.get_voice_style(voice if voice in voices else voices[0])
        wav, _ = self.tts.synthesize(text, voice_style=style, speed=1.05 * speed)
        return numpy(wav), self.tts.sample_rate


class Kitten(Backend):
    """KittenTTS: the small ONNX models (8 voices, CPU), or KittenTTS 2 (47 voices with
    styles, and copies a voice from a clip and its transcript)."""

    def __init__(self, model_dir, config):
        super().__init__(model_dir, config)
        from kittenml import KittenTTS

        kwargs = {"device": device()} if self._kitten2(model_dir) else {}
        self.model = KittenTTS(str(model_dir), **kwargs)
        if self._kitten2(model_dir):
            self.accepts = ("clone",)
            self.needs_text = True  # otherwise it downloads Whisper to transcribe the clip

    @staticmethod
    def _kitten2(model_dir: Path) -> bool:
        return read_json(model_dir / "config.json").get("type") == "KITTEN2"

    @classmethod
    def prepare(cls, dest: Path, config: dict) -> None:
        if cls._kitten2(dest):  # its vocoder comes from Chatterbox Turbo, by name
            snapshot("ResembleAI/chatterbox-turbo", ["s3gen_meanflow.safetensors"])

    def voices(self):
        return list(self.model.available_voices)

    def details(self):
        out = {}
        for v in self.voices():
            m = re.match(r"expr-voice-(\d+)-([mf])$", v)
            if m:
                g = "Female" if m[2] == "f" else "Male"
                out[v] = {"name": f"Voice {m[1]}", "language": "Voices", "gender": g}
            else:
                out[v] = {"name": words(v), "language": "Voices", "gender": ""}
        return out

    def speak(self, text, voice, speed, profile=None):
        profile = self.check(profile)
        if profile:
            wav = self.model.generate(
                text, reference=profile["audio"], reference_text=profile["text"]
            )
        else:
            voices = self.voices()
            v = voice if voice in voices else voices[0]
            if self.accepts:
                wav = self.model.generate(text, voice=v)
            else:
                wav = self.model.generate(text, voice=v, speed=speed)
        return numpy(wav), self.model.sample_rate


# Kyutai TTS 1.6B speaks in voices from kyutai/tts-voices, as precomputed embeddings: about
# 120 of them (the donated voices, 228 more, can be added from the voice library).
KYUTAI_VOICES = [
    "alba-mackenna/*.safetensors",
    "unmute-prod-website/*.safetensors",
    "vctk/*_023.wav.*.safetensors",
]
VCTK_INFO = re.compile(r"vctk/p(\d+)_023")


class Kyutai(Backend):
    """Kyutai TTS (moshi). The 1.6B English/French model speaks in Kyutai's voice library;
    the 0.75B public model copies a voice from a short clip."""

    def __init__(self, model_dir, config):
        super().__init__(model_dir, config)
        from moshi.models.loaders import CheckpointInfo
        from moshi.models.tts import TTSModel

        cfg = read_json(model_dir / "config.json")
        info = CheckpointInfo.from_hf_repo(
            config.get("source", {}).get("repo", "kyutai/tts-1.6b-en_fr"),
            moshi_weights=model_dir / cfg["moshi_name"],
            mimi_weights=model_dir / cfg["mimi_name"],
            tokenizer=model_dir / cfg["tokenizer_name"],
            config_path=model_dir / "config.json",
        )
        # The 1.6B model takes 32 codebooks; the 0.75B one only has 16.
        n_q = min(32, (info.lm_config or {}).get("n_q", 32))
        self.tts = TTSModel.from_checkpoint_info(info, n_q=n_q, temp=0.6, device=device())
        if self.tts.multi_speaker:
            folder = snapshot(VOICE_REPO, KYUTAI_VOICES, offline=True)
            suffix = self.tts.voice_suffix
            self.embeddings = {
                p.relative_to(folder).as_posix().removesuffix(suffix): p
                for p in folder.rglob(f"*{suffix}")
            }
        else:
            self.embeddings = {}
            self.accepts = ("clone",)
            self.starters = True

    @staticmethod
    def prepare(dest: Path, config: dict) -> None:
        # The library tells the two apart only once loaded; both kinds of voice are small.
        snapshot(VOICE_REPO, KYUTAI_VOICES)
        fetch_starters()

    def voices(self):
        return sorted(self.embeddings) if self.embeddings else super().voices()

    def details(self):
        if not self.embeddings:
            return super().details()
        out = {}
        for v in self.embeddings:
            group, _, name = v.partition("/")
            m = VCTK_INFO.match(v)
            if m:
                out[v] = {
                    "name": f"Speaker {m[1]}",
                    "language": "VCTK (English accents)",
                    "gender": "",
                }
            elif group == "voice-donations":
                out[v] = {"name": Path(name).stem, "language": "Donated voices", "gender": ""}
            else:
                label = {"alba-mackenna": "Alba MacKenna", "unmute-prod-website": "Unmute"}
                out[v] = {
                    "name": words(Path(name).stem).title(),
                    "language": label.get(group, group),
                    "gender": "",
                }
        return out

    def speak(self, text, voice, speed, profile=None):
        if self.embeddings:
            path = self.embeddings.get(voice) or self.embeddings.get(
                "alba-mackenna/casual.wav", next(iter(self.embeddings.values()))
            )
            source = str(path)
        else:
            source = "file://" + self.clone(voice, profile)["audio"]
        wav = self.tts.simple_generate(text, source, show_progress=False)[0]
        return numpy(wav), self.tts.mimi.sample_rate


class Pocket(Backend):
    """Kyutai's Pocket TTS: 100M parameters, fast on CPU, about 20 voices. The gated
    ``kyutai/pocket-tts`` weights also copy a voice from a clip."""

    LANG = "english"
    NO_CLONING = "kyutai/pocket-tts-without-voice-cloning"

    def __init__(self, model_dir, config):
        super().__init__(model_dir, config)
        import pocket_tts
        from pocket_tts import TTSModel

        lang = model_dir / "languages" / self.LANG

        def local(m: re.Match) -> str:
            path = model_dir / m[1]
            return str(path if path.exists() else hub_file(self.NO_CLONING, m[1]))

        text = (Path(pocket_tts.__file__).parent / "config" / f"{self.LANG}.yaml").read_text()
        text = re.sub(r"hf://kyutai/pocket-tts[\w-]*/([^@\s]+)@\w+", local, text)
        text = re.sub(r"(?m)^(weights_path\w*): .*$", rf"\1: {lang / 'model.safetensors'}", text)
        cfg = model_dir / ".amy-pocket.yaml"
        cfg.write_text(text)
        self.model = TTSModel.load_model(config=str(cfg))
        self.embeddings = {p.stem: p for p in sorted((lang / "embeddings").glob("*.safetensors"))}
        if config.get("source", {}).get("repo") != self.NO_CLONING:
            self.accepts = ("clone",)
        self.states: dict[str, object] = {}

    @classmethod
    def prepare(cls, dest: Path, config: dict) -> None:
        if not (dest / "languages" / cls.LANG / "tokenizer.json").exists():
            snapshot(cls.NO_CLONING, [f"languages/{cls.LANG}/tokenizer.json"])

    def voices(self):
        return list(self.embeddings) or [DEFAULT]

    def speak(self, text, voice, speed, profile=None):
        profile = self.check(profile)
        if profile:
            source = profile["audio"]
        else:
            source = str(self.embeddings.get(voice) or next(iter(self.embeddings.values())))
        if source not in self.states:
            self.states = {source: self.model.get_state_for_audio_prompt(source)}
        return numpy(self.model.generate_audio(self.states[source], text)), self.model.sample_rate


class Parler(Backend):
    """Parler-TTS (and AI4Bharat's Indic Parler-TTS): speaks in a voice described in words."""

    accepts = ("design",)

    def __init__(self, model_dir, config):
        super().__init__(model_dir, config)
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer

        self.model = ParlerTTSForConditionalGeneration.from_pretrained(str(model_dir)).to(device())
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.describer = self._describer(model_dir) or self.tokenizer

    @staticmethod
    def _encoder(model_dir: Path) -> str | None:
        return read_json(model_dir / "config.json").get("text_encoder", {}).get("_name_or_path")

    @classmethod
    def _describer(cls, model_dir: Path):
        """The description tokenizer, when it isn't the model's own (Indic Parler-TTS)."""
        from transformers import AutoTokenizer

        name = cls._encoder(model_dir)
        if not name:
            return None
        try:
            return AutoTokenizer.from_pretrained(name, local_files_only=True)
        except OSError:
            return None

    @classmethod
    def prepare(cls, dest: Path, config: dict) -> None:
        name = cls._encoder(dest)
        if name:
            snapshot(name, ["*.json", "*.model", "*.txt"])

    def voices(self):
        return list(DESIGNS)

    def speak(self, text, voice, speed, profile=None):
        description = self.described(voice, self.check(profile)) or next(iter(DESIGNS.values()))
        description = description.rstrip(". ") + ". The recording is very clear and close up."
        dev = device()
        audio = self.model.generate(
            input_ids=self.describer(description, return_tensors="pt").input_ids.to(dev),
            prompt_input_ids=self.tokenizer(text, return_tensors="pt").input_ids.to(dev),
        )
        return numpy(audio), self.model.config.sampling_rate


# MeloTTS: model repo -> (language code, the BERT it loads by name).
MELO = {
    "english": ("EN", "bert-base-uncased"),
    "english-v2": ("EN_V2", "bert-base-uncased"),
    "english-v3": ("EN_NEWEST", "bert-base-uncased"),
    "chinese": ("ZH", "hfl/chinese-roberta-wwm-ext-large"),
    "japanese": ("JP", "tohoku-nlp/bert-base-japanese-v3"),
    "spanish": ("ES", "dccuchile/bert-base-spanish-wwm-uncased"),
    "french": ("FR", "dbmdz/bert-base-french-europeana-cased"),
    "korean": ("KR", "kykim/bert-kor-base"),
}
# MeloTTS loads every language's tokenizer when it's imported, whatever the model's language.
MELO_TOKENIZERS = (
    "bert-base-uncased",
    "bert-base-multilingual-uncased",
    "tohoku-nlp/bert-base-japanese-v3",
    "dccuchile/bert-base-spanish-wwm-uncased",
    "dbmdz/bert-base-french-europeana-cased",
    "kykim/bert-kor-base",
    "hfl/chinese-roberta-wwm-ext-large",
)
MELO_SPEAKERS = {
    "EN-US": "American",
    "EN-BR": "British",
    "EN_INDIA": "Indian",
    "EN-AU": "Australian",
    "EN-Default": "Default",
}


class Melo(Backend):
    """MyShell's MeloTTS: light, real-time on CPU; English accents, Chinese, Japanese,
    Spanish, French and Korean."""

    def __init__(self, model_dir, config):
        super().__init__(model_dir, config)
        self.dictionary()
        from melo.api import TTS

        lang, _ = self.language(config)
        self.tts = TTS(
            language=lang,
            device=device(),
            config_path=str(model_dir / "config.json"),
            ckpt_path=str(model_dir / "checkpoint.pth"),
        )
        self.speakers = dict(self.tts.hps.data.spk2id)

    @staticmethod
    def language(config: dict) -> tuple[str, str]:
        repo = config.get("source", {}).get("repo", "").lower()
        return MELO.get(repo.split("melotts-")[-1], MELO["english"])

    @staticmethod
    def dictionary() -> None:
        """MeloTTS imports a Japanese tokenizer that wants the full 500 MB UniDic; point it
        at the small one (unidic-lite) instead."""
        import unidic
        import unidic_lite

        target = Path(unidic.DICDIR)
        if not (target / "mecabrc").exists():
            if target.is_symlink() or target.exists():
                target.unlink() if target.is_symlink() else target.rmdir()
            target.symlink_to(unidic_lite.DICDIR, target_is_directory=True)

    @classmethod
    def prepare(cls, dest: Path, config: dict) -> None:
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        _, bert = cls.language(config)
        # Loading them once with the network on puts exactly the files they need in the cache.
        for name in MELO_TOKENIZERS:
            AutoTokenizer.from_pretrained(name)
        AutoModelForMaskedLM.from_pretrained(bert)
        try:
            import nltk

            for pkg in ("averaged_perceptron_tagger_eng", "cmudict"):
                nltk.download(pkg, quiet=True)
        except Exception:  # noqa: BLE001, S110 - English text normalisation fetches these too
            pass

    def voices(self):
        return list(self.speakers)

    def details(self):
        return {
            s: {"name": MELO_SPEAKERS.get(s, s), "language": "Voices", "gender": ""}
            for s in self.speakers
        }

    def speak(self, text, voice, speed, profile=None):
        speaker = self.speakers.get(voice, next(iter(self.speakers.values())))
        wav = self.tts.tts_to_file(text, speaker, None, speed=speed, quiet=True)
        return numpy(wav), self.tts.hps.data.sampling_rate


class Sopro(Backend):
    """Sopro: small, fast, streams; speaks in a voice copied from a short clip."""

    accepts = ("clone",)
    starters = True

    def __init__(self, model_dir, config):
        super().__init__(model_dir, config)
        from sopro import SoproTTS

        self.tts = SoproTTS.from_pretrained(str(model_dir), device=device())

    @staticmethod
    def prepare(dest: Path, config: dict) -> None:
        fetch_starters()

    def speak(self, text, voice, speed, profile=None):
        clip = self.clone(voice, profile)
        wav = self.tts.synthesize(text, ref_audio_path=clip["audio"])
        return numpy(wav), getattr(self.tts, "sample_rate", 24000)


class Soprano(Backend):
    """Soprano: an 80M-parameter English model, many times faster than real time."""

    def __init__(self, model_dir, config):
        super().__init__(model_dir, config)
        from soprano import SopranoTTS

        self.tts = SopranoTTS(backend="transformers", device=device(), model_path=str(model_dir))

    def speak(self, text, voice, speed, profile=None):
        return numpy(self.tts.infer(text)), 32000


class Zonos(Backend):
    """Zyphra's Zonos: copies a voice from a short clip, in English and several other
    languages."""

    accepts = ("clone",)
    starters = True

    def __init__(self, model_dir, config):
        super().__init__(model_dir, config)
        bundled_espeak()
        from zonos.model import Zonos as Model

        self.model = Model.from_local(
            str(model_dir / "config.json"), str(model_dir / "model.safetensors"), device=device()
        )
        self.speakers: dict[str, object] = {}

    @staticmethod
    def prepare(dest: Path, config: dict) -> None:
        fetch_starters()
        snapshot("Zyphra/Zonos-v0.1-speaker-embedding")
        snapshot("descript/dac_44khz")

    def speak(self, text, voice, speed, profile=None):
        import soundfile as sf
        import torch
        from zonos.conditioning import make_cond_dict

        clip = self.clone(voice, profile)["audio"]
        if clip not in self.speakers:
            wav, sr = sf.read(clip, dtype="float32", always_2d=True)
            tensor = torch.from_numpy(wav.mean(axis=1)).unsqueeze(0)
            self.speakers = {clip: self.model.make_speaker_embedding(tensor, sr)}
        cond = make_cond_dict(text=text, speaker=self.speakers[clip], language="en-us")
        codes = self.model.generate(self.model.prepare_conditioning(cond))
        return numpy(self.model.autoencoder.decode(codes)[0]), self.model.autoencoder.sampling_rate


class NeuTts(Backend):
    """Neuphonic's NeuTTS Air and Nano: on-device English; copies a voice from a clip and
    its transcript."""

    accepts = ("clone",)
    needs_text = True
    starters = True
    CODEC = "neuphonic/neucodec"

    def __init__(self, model_dir, config):
        super().__init__(model_dir, config)
        bundled_espeak()
        from neutts import NeuTTS

        dev = device()
        self.tts = NeuTTS(
            backbone_repo=str(model_dir),
            backbone_device=dev,
            codec_repo=str(snapshot(self.CODEC, offline=True)),
            codec_device=dev,
        )
        self.refs: dict[str, object] = {}

    @classmethod
    def prepare(cls, dest: Path, config: dict) -> None:
        fetch_starters()
        snapshot(cls.CODEC)

    def speak(self, text, voice, speed, profile=None):
        clip = self.clone(voice, profile)
        if clip["audio"] not in self.refs:
            self.refs = {clip["audio"]: self.tts.encode_reference(clip["audio"])}
        return numpy(self.tts.infer(text, self.refs[clip["audio"]], clip["text"])), 24000


class OuteTts(Backend):
    """OuteAI's OuteTTS 1.0 (Llama 1B or Qwen 0.6B based), with its default English
    speaker."""

    DAC = ("ibm-research/DAC.speech.v1.0", "weights_24khz_1.5kbps_v1.0.pth")

    def __init__(self, model_dir, config):
        super().__init__(model_dir, config)
        import outetts
        from outetts.dac.interface import DacInterface

        DacInterface._get_model = lambda _self: hub_file(*self.DAC)
        self.iface = outetts.Interface(
            config=outetts.ModelConfig(
                model_path=str(model_dir),
                tokenizer_path=str(model_dir),
                interface_version=outetts.InterfaceVersion.V3,
                backend=outetts.Backend.HF,
                device=device(),
            )
        )
        self.speaker = self.iface.load_default_speaker("EN-FEMALE-1-NEUTRAL")

    @classmethod
    def prepare(cls, dest: Path, config: dict) -> None:
        snapshot(cls.DAC[0], [cls.DAC[1]])

    def details(self):
        return {DEFAULT: {"name": "English female", "language": "Voices", "gender": "Female"}}

    def speak(self, text, voice, speed, profile=None):
        import outetts

        out = self.iface.generate(config=outetts.GenerationConfig(text=text, speaker=self.speaker))
        return numpy(out.audio), out.sr


class IndexTts(Backend):
    """Bilibili's IndexTTS 2: expressive Chinese and English; copies a voice from a clip."""

    accepts = ("clone",)
    starters = True

    def __init__(self, model_dir, config):
        super().__init__(model_dir, config)
        from indextts.infer_v2 import IndexTTS2

        self.tts = IndexTTS2(
            cfg_path=str(model_dir / "config.yaml"),
            model_dir=str(model_dir),
            use_fp16=device() == "cuda",
            use_cuda_kernel=False,
            use_deepspeed=False,
        )

    @staticmethod
    def prepare(dest: Path, config: dict) -> None:
        from indextts.utils.model_download import ensure_models_available

        fetch_starters()
        ensure_models_available(str(dest))

    def speak(self, text, voice, speed, profile=None):
        import tempfile

        import soundfile as sf

        clip = self.clone(voice, profile)
        with tempfile.NamedTemporaryFile(suffix=".wav") as out:
            self.tts.infer(spk_audio_prompt=clip["audio"], text=text, output_path=out.name)
            wav, sr = sf.read(out.name, dtype="float32")
        return wav, sr


VIBEVOICE_LANGS = {
    "de": "German", "en": "English", "fr": "French", "in": "English (India)", "it": "Italian",
    "jp": "Japanese", "kr": "Korean", "nl": "Dutch", "pl": "Polish", "pt": "Portuguese",
    "sp": "Spanish",
}  # fmt: skip


class VibeVoice(Backend):
    """Microsoft's VibeVoice Realtime 0.5B: streaming, long-form speech in 25 preset voices
    (Microsoft ships no voice cloning for it)."""

    def __init__(self, model_dir, config):
        super().__init__(model_dir, config)
        import torch
        from vibevoice.modular.modeling_vibevoice_streaming_inference import (
            VibeVoiceStreamingForConditionalGenerationInference as Model,
        )
        from vibevoice.processor.vibevoice_streaming_processor import (
            VibeVoiceStreamingProcessor,
        )

        dev = device()
        self.processor = VibeVoiceStreamingProcessor.from_pretrained(str(model_dir))
        self.model = Model.from_pretrained(
            str(model_dir),
            torch_dtype=torch.bfloat16 if dev == "cuda" else torch.float32,
            device_map=dev,
            attn_implementation="sdpa",
        )
        self.model.eval()
        self.model.set_ddpm_inference_steps(num_steps=5)
        self.presets = {p.stem: p for p in sorted((model_dir / "voices").glob("*.pt"))}

    @staticmethod
    def prepare(dest: Path, config: dict) -> None:
        config_file = dest / "preprocessor_config.json"
        settings = read_json(config_file) if config_file.exists() else {}
        name = settings.get("language_model_pretrained_name", "Qwen/Qwen2.5-0.5B")
        snapshot(name, ["*.json", "*.txt"])

    def voices(self):
        return list(self.presets)

    def details(self):
        out = {}
        for v in self.presets:
            lang, _, rest = v.partition("-")
            name, _, gender = rest.partition("_")
            out[v] = {
                "name": name.replace("Spk", "Speaker "),
                "language": VIBEVOICE_LANGS.get(lang, lang),
                "gender": {"man": "Male", "woman": "Female"}.get(gender, ""),
            }
        return out

    def speak(self, text, voice, speed, profile=None):
        import copy

        import torch

        path = (
            self.presets.get(voice)
            or self.presets.get("en-Emma_woman")
            or next(iter(self.presets.values()))
        )
        # Microsoft's own preset files: prompt caches pickled with transformers classes.
        cached = torch.load(path, map_location=device(), weights_only=False)  # noqa: S614
        inputs = self.processor.process_input_with_cached_prompt(
            text=text.replace("’", "'"),
            cached_prompt=cached,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        inputs = {k: v.to(device()) if torch.is_tensor(v) else v for k, v in inputs.items()}
        out = self.model.generate(
            **inputs,
            max_new_tokens=None,
            cfg_scale=1.5,
            tokenizer=self.processor.tokenizer,
            generation_config={"do_sample": False},
            verbose=False,
            all_prefilled_outputs=copy.deepcopy(cached),
        )
        return numpy(out.speech_outputs[0]), 24000


class VieNeu(Backend):
    """VieNeu-TTS v3 Turbo: Vietnamese (and code-switched English) at 48 kHz, with preset
    voices; copies a voice from a clip."""

    accepts = ("clone",)
    MOSS = "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX"

    def __init__(self, model_dir, config):
        super().__init__(model_dir, config)
        from vieneu import Vieneu

        repo = config.get("source", {}).get("repo", "pnnbao-ump/VieNeu-TTS-v3-Turbo")
        self.tts = Vieneu(backbone_repo=repo)
        self.presets = {label: vid for label, vid in self.tts.list_preset_voices()}

    @classmethod
    def prepare(cls, dest: Path, config: dict) -> None:
        snapshot(config.get("source", {}).get("repo", "pnnbao-ump/VieNeu-TTS-v3-Turbo"))
        snapshot(cls.MOSS)

    def voices(self):
        return list(self.presets.values())

    def details(self):
        out = {}
        for label, vid in self.presets.items():
            gender = "Female" if " Nữ" in label else "Male" if " Nam" in label else ""
            out[vid] = {"name": vid, "language": "Vietnamese", "gender": gender}
        return out

    def speak(self, text, voice, speed, profile=None):
        profile = self.check(profile)
        if profile:
            wav = self.tts.infer(text, ref_audio=profile["audio"], ref_text=profile.get("text"))
        else:
            ids = self.voices()
            wav = self.tts.infer(text, voice=voice if voice in ids else ids[0])
        return numpy(wav), 48000


def read_clip(path: str, rate: int):
    """A mono float clip at ``rate`` (linear resampling: fine for a voice prompt)."""
    import numpy as np
    import soundfile as sf

    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if sr != rate:
        n = int(len(wav) * rate / sr)
        wav = np.interp(np.linspace(0, len(wav) - 1, n), np.arange(len(wav)), wav)
    return wav.astype(np.float32)


class Csm(Backend):
    """Sesame CSM 1B (transformers): conversational English. It has no fixed voices, so it
    speaks in the voice of a sample and its transcript (a starter voice or the user's)."""

    accepts = ("clone",)
    needs_text = True
    starters = True

    def __init__(self, model_dir, config):
        super().__init__(model_dir, config)
        from transformers import AutoProcessor, CsmForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(str(model_dir))
        self.model = CsmForConditionalGeneration.from_pretrained(
            str(model_dir), device_map=device()
        )

    @staticmethod
    def prepare(dest: Path, config: dict) -> None:
        fetch_starters()

    def speak(self, text, voice, speed, profile=None):
        clip = self.clone(voice, profile)
        conversation = [
            {"role": "0", "content": [
                {"type": "text", "text": clip["text"]},
                {"type": "audio", "path": read_clip(clip["audio"], 24000)},
            ]},
            {"role": "0", "content": [{"type": "text", "text": text}]},
        ]  # fmt: skip
        inputs = self.processor.apply_chat_template(
            conversation, tokenize=True, return_dict=True
        ).to(device())
        audio = self.model.generate(**inputs, output_audio=True)
        return numpy(audio[0]), 24000


# transformers model types with their own class (see speech_worker.Transformers).
TRANSFORMERS: dict[str, type[Backend]] = {"csm": Csm}

BACKENDS: dict[str, type[Backend]] = {
    "supertonic": Supertonic,
    "kitten": Kitten,
    "kyutai": Kyutai,
    "pocket": Pocket,
    "parler": Parler,
    "melo": Melo,
    "sopro": Sopro,
    "soprano": Soprano,
    "zonos": Zonos,
    "neutts": NeuTts,
    "outetts": OuteTts,
    "indextts": IndexTts,
    "vibevoice": VibeVoice,
    "vieneu": VieNeu,
}
