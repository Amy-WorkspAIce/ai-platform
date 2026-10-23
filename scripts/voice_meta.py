"""Make backend/app/voice_meta.json: labels for the voices in kyutai/tts-voices.

    pip install numpy soundfile
    python scripts/voice_meta.py path/to/VCTK/speaker-info.txt [--cache DIR]

Downloads each voice sample the app shows (the voice library and Kyutai TTS's built-in
voices), measures its pitch, and writes gender, age, accent, region, language and style
wherever they're known. ``speaker-info.txt`` comes with the VCTK corpus
(https://datashare.ed.ac.uk/handle/10283/3443).

Gender from pitch is only an estimate. Checked against the VCTK speakers (whose gender is
known), the cut-off at 155 Hz got 104 of 105 right; voices between 145 and 165 Hz are left
without a gender.
"""

import argparse
import concurrent.futures as cf
import json
import sys
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.voice_library import COLLECTIONS, REPO, entries, file_url  # noqa: E402
from app.voice_meta import DATA, key  # noqa: E402

MALE_BELOW, FEMALE_ABOVE = 145.0, 165.0

# Kyutai TTS 1.6B's own voices that aren't in the voice library.
UNMUTE = [
    "unmute-prod-website/default_voice.wav",
    "unmute-prod-website/degaulle-2.wav",
    "unmute-prod-website/developer-1.mp3",
    "unmute-prod-website/developpeuse-3.wav",
    "unmute-prod-website/ex04_narration_longform_00001.wav",
    "unmute-prod-website/fabieng-enhanced-v2.wav",
    "unmute-prod-website/freesound/440565_why-is-there-educationwav.mp3",
    "unmute-prod-website/freesound/519189_request-42---hmm-i-dont-knowwav.mp3",
]

# What the recordings are known to be.
KNOWN = {
    "alba-mackenna/a-moment-by": {"gender": "Female", "style": "Storytelling"},
    "alba-mackenna/announcer": {"gender": "Female", "style": "Announcer"},
    "alba-mackenna/casual": {"gender": "Female", "style": "Casual"},
    "alba-mackenna/merchant": {"gender": "Female", "style": "Character"},
    "voice-zero/bill_boerst": {"gender": "Male", "style": "Audiobook"},
    "voice-zero/caro_davy": {"gender": "Female", "style": "Audiobook"},
    "voice-zero/peter_yearsley": {"gender": "Male", "style": "Audiobook"},
    "voice-zero/stuart_bell": {"gender": "Male", "style": "Audiobook"},
    "unmute-prod-website/degaulle-2": {"gender": "Male", "language": "French"},
    "unmute-prod-website/developpeuse-3": {"gender": "Female", "language": "French"},
    "unmute-prod-website/fabieng-enhanced-v2": {"gender": "Male", "language": "French"},
    "unmute-prod-website/ex04_narration_longform_00001": {"style": "Narration"},
}

ACCENTS = {"NorthernIrish": "Northern Irish", "SouthAfrican": "South African"}
ACCENTS |= {"NewZealand": "New Zealand"}
NOT_REGIONS = {"English", "India"}  # leftovers in the speaker list, not places


def vctk_speakers(path: Path) -> dict[str, dict]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:
        parts = line.split()
        if len(parts) < 4:
            continue
        sid = parts[0].lstrip("p")
        accent = ACCENTS.get(parts[3], parts[3])
        rest = parts[4:]
        if rest[:1] == ["English"] and accent != "English":  # "Australian English Sydney"
            rest = rest[1:]
        region = " ".join(rest)
        gender = {"f": "Female", "m": "Male"}[parts[2][0].lower()]
        info = {"gender": gender, "age": int(parts[1]), "accent": accent, "language": "English"}
        if region and region not in NOT_REGIONS:
            info["region"] = region
        out[f"vctk/p{sid}"] = info
    return out


def pitch(path: Path) -> float | None:
    """Median pitch (Hz) of the voiced, louder half of a recording, by autocorrelation."""
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    hop, win = int(sr * 0.01), int(sr * 0.04)
    lo, hi = int(sr / 400), int(sr / 60)
    starts = range(0, len(x) - win, hop)
    rms = np.array([np.sqrt(np.mean(x[i : i + win] ** 2)) for i in starts])
    gate = np.percentile(rms, 50)
    found = []
    for k, i in enumerate(starts):
        if rms[k] < gate:
            continue
        frame = x[i : i + win] * np.hanning(win)
        ac = np.correlate(frame, frame, "full")[win - 1 :]
        if ac[0] <= 0:
            continue
        ac = ac / ac[0]
        lag = lo + int(np.argmax(ac[lo:hi]))
        if ac[lag] > 0.45:
            found.append(sr / lag)
    return float(np.median(found)) if len(found) > 20 else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("speaker_info", type=Path)
    parser.add_argument("--cache", type=Path, default=Path(tempfile.gettempdir()) / "amy-voices")
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)

    with urllib.request.urlopen(f"https://huggingface.co/api/models/{REPO}") as resp:  # noqa: S310
        files = [s["rfilename"] for s in json.load(resp)["siblings"]]
    library = entries(files)
    paths = [e["id"] for e in library] + UNMUTE

    def fetch(path: str) -> Path:
        local = args.cache / path.replace("/", "__")
        if not local.exists():
            urllib.request.urlretrieve(file_url(path), local)  # noqa: S310
        return local

    with cf.ThreadPoolExecutor(8) as pool:
        local = dict(zip(paths, pool.map(fetch, paths), strict=True))

    languages = {folder: lang for folder, (_, lang, _, _) in COLLECTIONS.items()}
    languages["voice-donations"] = ""  # volunteers from anywhere: not known per voice
    vctk = vctk_speakers(args.speaker_info)
    table: dict[str, dict] = {}
    for path in paths:
        k = key(path)
        info = dict(vctk.get(k) or {})
        folder = next((f for f in languages if path.startswith(f + "/")), "")
        if languages.get(folder) and "language" not in info:
            info["language"] = languages[folder]
        info |= KNOWN.get(k, {})
        if "gender" not in info:
            hz = pitch(local[path])
            if hz and hz < MALE_BELOW:
                info |= {"gender": "Male", "estimated": True}
            elif hz and hz > FEMALE_ABOVE:
                info |= {"gender": "Female", "estimated": True}
        table[k] = info
    DATA.write_text(json.dumps(table, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {len(table)} voices to {DATA}")


if __name__ == "__main__":
    main()
