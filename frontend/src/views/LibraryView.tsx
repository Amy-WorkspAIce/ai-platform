import { useCallback, useEffect, useMemo, useState } from "react";
import { Api, type GalleryAudio, type GalleryImage, type GalleryVideo, type VideoJob } from "../api";
import { useBlobUrls } from "../blobs";
import { AudioView, VideoJobView } from "../media";
import { describeVoice } from "../devices";

type Tab = "images" | "videos" | "audio";

/** Everything you've created (or attached), in one place. Creation happens in chat. */
export function LibraryView({ api, onNewChat }: { api: Api; onNewChat: () => void }) {
  const [tab, setTab] = useState<Tab>("images");
  const [images, setImages] = useState<GalleryImage[]>([]);
  const [videos, setVideos] = useState<GalleryVideo[]>([]);
  const [jobs, setJobs] = useState<VideoJob[]>([]);
  const [audio, setAudio] = useState<GalleryAudio[]>([]);
  const [selected, setSelected] = useState<GalleryImage | null>(null);
  const [error, setError] = useState("");

  const refresh = useCallback(async () => {
    try {
      const [i, v, j, a] = await Promise.all([api.images(), api.videos(), api.videoJobs(), api.audioList()]);
      setImages(i);
      setVideos(v);
      setJobs(j.filter((x) => x.status === "queued" || x.status === "running"));
      setAudio(a);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [api]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const imageUrl = useCallback((id: string) => api.imageUrl(id), [api]);
  const imageIds = useMemo(() => images.map((i) => i.id), [images]);
  const urls = useBlobUrls(imageUrl, imageIds);
  const videoUrl = useCallback((id: string) => api.videoUrl(id), [api]);
  const videoIds = useMemo(() => videos.map((v) => v.id), [videos]);
  const vurls = useBlobUrls(videoUrl, videoIds);

  async function remove(kind: Tab, id: string) {
    if (!confirm("Delete this from your library?")) return;
    try {
      if (kind === "images") await api.deleteImage(id);
      else if (kind === "videos") await api.deleteVideo(id);
      else await api.deleteAudio(id);
      setSelected(null);
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  const counts = { images: images.length, videos: videos.length + jobs.length, audio: audio.length };
  const empty = counts[tab] === 0;

  return (
    <div className="page">
      <h1>Library</h1>
      <p className="muted small">
        Everything you've created in chat. To make something new, just ask, e.g. "generate a photo of a cabin in the
        snow" or "read this aloud".
      </p>
      <div className="segmented library-tabs">
        {(["images", "videos", "audio"] as Tab[]).map((t) => (
          <button key={t} className={tab === t ? "active" : ""} onClick={() => setTab(t)}>
            {t[0].toUpperCase() + t.slice(1)} {counts[t] > 0 && <span className="muted">({counts[t]})</span>}
          </button>
        ))}
      </div>
      {error && <p className="error">{error}</p>}

      {empty && (
        <div className="welcome-card left">
          <p>Nothing here yet. Ask for {tab === "images" ? "an image" : tab === "videos" ? "a video" : "something to be read aloud"} in chat and it'll show up here.</p>
          <button className="primary" onClick={onNewChat}>
            Start a chat
          </button>
        </div>
      )}

      {tab === "images" && (
        <div className="gallery">
          {images.map((img) => (
            <button key={img.id} className="thumb" onClick={() => setSelected(img)} title={img.prompt}>
              {urls[img.id] ? <img src={urls[img.id]} alt={img.prompt} loading="lazy" /> : <span className="muted small">…</span>}
            </button>
          ))}
        </div>
      )}

      {tab === "videos" && (
        <>
          {jobs.map((j) => (
            <div key={j.id} className="card video-job">
              <p className="clip-prompt">{j.prompt}</p>
              <VideoJobView api={api} jobId={j.id} />
            </div>
          ))}
          <div className="video-gallery">
            {videos.map((v) => (
              <figure key={v.id} className="card video-card">
                {vurls[v.id] ? <video src={vurls[v.id]} controls loop playsInline preload="metadata" /> : <div className="video-placeholder muted">…</div>}
                <figcaption>
                  <p className="clip-prompt">{v.prompt}</p>
                  <p className="muted small">
                    {v.model.replace("image/", "").split("/").pop()} · {v.params.width}×{v.params.height} ·{" "}
                    {(v.params.frames / v.params.fps).toFixed(1)}s
                  </p>
                  <div className="row-actions start">
                    {vurls[v.id] && (
                      <a className="button secondary" href={vurls[v.id]} download={`amy-${v.id}.mp4`}>
                        Download
                      </a>
                    )}
                    <button className="link danger" onClick={() => remove("videos", v.id)}>
                      Delete
                    </button>
                  </div>
                </figcaption>
              </figure>
            ))}
          </div>
        </>
      )}

      {tab === "audio" && (
        <div className="audio-list">
          {audio.map((a) => (
            <div key={a.id} className="card audio-item">
              <p className="clip-prompt">{a.text}</p>
              <div className="audio-row">
                <AudioView api={api} id={a.id} />
                <span className="muted small">{a.params.voice_name ?? describeVoice(a.params.voice).name}</span>
                <button className="link danger" onClick={() => remove("audio", a.id)}>
                  Delete
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      {selected && (
        <div className="lightbox" onClick={() => setSelected(null)}>
          <div className="lightbox-body" onClick={(e) => e.stopPropagation()}>
            {urls[selected.id] && <img src={urls[selected.id]} alt={selected.prompt} />}
            <div className="lightbox-info">
              <p>{selected.prompt}</p>
              <p className="muted small">
                {selected.model === "upload"
                  ? "Uploaded"
                  : `${selected.model.replace("image/", "").split("/").pop()} · ${selected.params.width}×${selected.params.height} · seed ${selected.params.seed}`}
              </p>
              <div className="row-actions start">
                <a className="button secondary" href={urls[selected.id]} download={`amy-${selected.id}.png`}>
                  Download
                </a>
                <button className="link danger" onClick={() => remove("images", selected.id)}>
                  Delete
                </button>
                <button className="link" onClick={() => setSelected(null)}>
                  Close
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
