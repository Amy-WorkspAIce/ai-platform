import { useCallback, useEffect, useState } from "react";
import { getPref, setPref } from "./api";

/**
 * Which microphone and speakers this browser uses. Saved per browser (a phone and a desktop
 * have different devices); empty means the system default.
 */
export const MIC_PREF = "mic-device";
export const SPEAKER_PREF = "speaker-device";

export function micConstraints(): MediaStreamConstraints {
  const id = getPref(MIC_PREF);
  // "ideal" so a device that's been unplugged falls back to the default instead of failing.
  return { audio: id ? { deviceId: { ideal: id } } : true };
}

type SinkAudio = HTMLAudioElement & { setSinkId?: (id: string) => Promise<void> };

/** Can this browser choose the output device? (Chrome and Edge can; Safari and Firefox can't yet.) */
export const canChooseSpeaker = () =>
  typeof (HTMLAudioElement.prototype as SinkAudio).setSinkId === "function";

/** Send an audio element to the chosen speakers. */
export function routeAudio(el: HTMLAudioElement | null): void {
  const id = getPref(SPEAKER_PREF);
  const a = el as SinkAudio | null;
  if (a && id && a.setSinkId) a.setSinkId(id).catch(() => undefined);
}

export type DeviceLists = { mics: MediaDeviceInfo[]; speakers: MediaDeviceInfo[]; labelled: boolean };

/**
 * The browser's audio devices. Browsers hide device names until the page has been given
 * microphone access once, so ``ask()`` requests it to fill in the names.
 */
export function useAudioDevices() {
  const [lists, setLists] = useState<DeviceLists>({ mics: [], speakers: [], labelled: false });
  const supported = !!navigator.mediaDevices?.enumerateDevices;

  const refresh = useCallback(async () => {
    if (!supported) return;
    const all = await navigator.mediaDevices.enumerateDevices();
    setLists({
      mics: all.filter((d) => d.kind === "audioinput"),
      speakers: all.filter((d) => d.kind === "audiooutput"),
      labelled: all.some((d) => d.label),
    });
  }, [supported]);

  const ask = useCallback(async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    stream.getTracks().forEach((t) => t.stop());
    await refresh();
  }, [refresh]);

  useEffect(() => {
    void refresh();
    if (!supported) return;
    navigator.mediaDevices.addEventListener("devicechange", refresh);
    return () => navigator.mediaDevices.removeEventListener("devicechange", refresh);
  }, [refresh, supported]);

  return { ...lists, supported, refresh, ask };
}

export function useDevicePref(name: string): [string, (id: string) => void] {
  const [value, setValue] = useState(() => getPref(name) ?? "");
  return [
    value,
    (id: string) => {
      setPref(name, id);
      setValue(id);
    },
  ];
}

/** Kokoro voice ids look like "af_heart": language, gender, then the name. */
const VOICE_LANG: Record<string, string> = {
  a: "American English",
  b: "British English",
  e: "Spanish",
  f: "French",
  h: "Hindi",
  i: "Italian",
  j: "Japanese",
  p: "Brazilian Portuguese",
  z: "Mandarin Chinese",
};

export function describeVoice(id: string): { name: string; language: string; gender: string } {
  const m = id.match(/^([a-z])([fm])_(.+)$/);
  if (!m || !VOICE_LANG[m[1]]) {
    // Other engines name voices freely ("Ana Florence", "ryan", "v2/en_speaker_3").
    const name = id.split("/").pop()!.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
    return { name, language: "Voices", gender: "" };
  }
  const name = m[3].replace(/(^|_)(\w)/g, (_, sep: string, c: string) => (sep ? " " : "") + c.toUpperCase());
  return { name, language: VOICE_LANG[m[1]] ?? "Other", gender: m[2] === "f" ? "Female" : "Male" };
}

/** A short line in the voice's own language, for previews. */
export function sampleText(id: string): string {
  const samples: Record<string, string> = {
    e: "Hola, soy una de las voces de Amy.",
    f: "Bonjour, je suis l'une des voix d'Amy.",
    h: "नमस्ते, मैं एमी की आवाज़ों में से एक हूँ।",
    i: "Ciao, sono una delle voci di Amy.",
    j: "こんにちは、エイミーの声のひとつです。",
    p: "Olá, eu sou uma das vozes da Amy.",
    z: "你好，我是艾米的声音之一。",
  };
  return samples[id[0]] ?? `Hi, I'm ${describeVoice(id).name}, one of Amy's voices.`;
}
