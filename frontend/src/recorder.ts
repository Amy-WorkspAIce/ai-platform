import { useCallback, useRef, useState } from "react";

/** Microphone access needs a secure context: HTTPS or localhost. */
export function canRecord(): boolean {
  return (
    window.isSecureContext &&
    !!navigator.mediaDevices?.getUserMedia &&
    typeof MediaRecorder !== "undefined"
  );
}

export function useRecorder() {
  const [recording, setRecording] = useState(false);
  const recorder = useRef<MediaRecorder | null>(null);
  const chunks = useRef<Blob[]>([]);

  const start = useCallback(async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const rec = new MediaRecorder(stream);
    chunks.current = [];
    rec.ondataavailable = (e) => e.data.size && chunks.current.push(e.data);
    rec.start();
    recorder.current = rec;
    setRecording(true);
  }, []);

  /** Stop recording and resolve with the audio, or null if nothing was recorded. */
  const stop = useCallback(
    () =>
      new Promise<Blob | null>((resolve) => {
        const rec = recorder.current;
        if (!rec) return resolve(null);
        rec.onstop = () => {
          rec.stream.getTracks().forEach((t) => t.stop());
          recorder.current = null;
          setRecording(false);
          const blob = new Blob(chunks.current, { type: rec.mimeType || "audio/webm" });
          resolve(blob.size ? blob : null);
        };
        rec.stop();
      }),
    [],
  );

  return { recording, start, stop };
}

export function extensionFor(blob: Blob): string {
  if (blob.type.includes("ogg")) return "ogg";
  if (blob.type.includes("mp4")) return "m4a";
  return "webm";
}
