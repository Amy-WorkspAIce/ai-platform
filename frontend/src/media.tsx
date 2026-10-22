import { useEffect, useState } from "react";
import type { Api, Attachment, VideoJob } from "./api";
import { routeAudio } from "./devices";
import { Icon } from "./icons";

/** Fetch one auth-protected file as an object URL (released when unmounted). */
export function useBlobUrl(fetchUrl: (() => Promise<string>) | null, key: string): string | null {
  const [url, setUrl] = useState<string | null>(null);
  useEffect(() => {
    if (!fetchUrl) return;
    let alive = true;
    let made: string | null = null;
    fetchUrl()
      .then((u) => {
        made = u;
        if (alive) setUrl(u);
        else URL.revokeObjectURL(u);
      })
      .catch(() => undefined);
    return () => {
      alive = false;
      if (made) URL.revokeObjectURL(made);
    };
  }, [key]);
  return url;
}

export function ImageView({ api, id, alt }: { api: Api; id: string; alt?: string }) {
  const url = useBlobUrl(() => api.imageUrl(id), id);
  if (!url) return <div className="media-placeholder">…</div>;
  return (
    <figure className="media image">
      <a href={url} target="_blank" rel="noreferrer" title="Open full size">
        <img src={url} alt={alt ?? "Generated image"} />
      </a>
      <figcaption>
        <a className="small" href={url} download={`amy-${id}.png`}>
          Download
        </a>
      </figcaption>
    </figure>
  );
}

const PHASES: Record<string, string> = {
  queued: "Waiting",
  optimizing: "Optimizing model files (first run only)",
  loading: "Loading the model",
  generating: "Generating",
  encoding: "Encoding video",
};

/** A video job: shows progress while rendering, then the clip. */
export function VideoJobView({ api, jobId }: { api: Api; jobId: string }) {
  const [job, setJob] = useState<VideoJob | null>(null);
  const [done, setDone] = useState(false);
  const [missing, setMissing] = useState(false);

  useEffect(() => {
    let stop = false;
    async function poll() {
      try {
        const jobs = await api.videoJobs();
        const j = jobs.find((x) => x.id === jobId) ?? null;
        if (!j) {
          // Finished a while ago (or the server restarted): is it in the library?
          const videos = await api.videos();
          if (videos.some((v) => v.id === jobId)) setDone(true);
          else setMissing(true);
          return;
        }
        setJob(j);
        if (j.status === "done") setDone(true);
        else if (j.status !== "error" && !stop) setTimeout(poll, 3000);
      } catch {
        if (!stop) setTimeout(poll, 5000);
      }
    }
    void poll();
    return () => {
      stop = true;
    };
  }, [api, jobId]);

  const url = useBlobUrl(done ? () => api.videoUrl(jobId) : null, `${jobId}-${done}`);
  if (done && url) {
    return (
      <figure className="media video">
        <video src={url} controls loop playsInline />
        <figcaption>
          <a className="small" href={url} download={`amy-${jobId}.mp4`}>
            Download
          </a>
        </figcaption>
      </figure>
    );
  }
  if (missing) return <p className="muted small">This video is no longer available.</p>;
  if (job?.status === "error") return <p className="error small">Video failed: {job.error}</p>;
  const pct = job?.phase === "encoding" ? 100 : job?.steps ? Math.round((job.step / job.steps) * 100) : 3;
  return (
    <div className="media-progress">
      <span className="muted small">
        <Icon name="video" size={14} /> {PHASES[job?.phase ?? "queued"] ?? job?.phase}
        {job?.phase === "generating" && job.steps ? ` · step ${job.step}/${job.steps}` : ""}
        {job ? ` · ${Math.round(Date.now() / 1000 - job.started_at)}s` : ""}
      </span>
      <div className="progress">
        <div style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

export function AudioView({ api, id }: { api: Api; id: string }) {
  const url = useBlobUrl(() => api.audioUrl(id), id);
  if (!url) return <div className="media-placeholder small">…</div>;
  return <audio className="media audio" src={url} controls ref={routeAudio} />;
}

export function AttachmentView({ api, attachment }: { api: Api; attachment: Attachment }) {
  if (attachment.type === "image") return <ImageView api={api} id={attachment.id} />;
  if (attachment.type === "video") return <VideoJobView api={api} jobId={attachment.job_id} />;
  return <AudioView api={api} id={attachment.id} />;
}
