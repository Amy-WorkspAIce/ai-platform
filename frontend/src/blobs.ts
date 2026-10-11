import { useEffect, useRef, useState } from "react";

/**
 * Media that needs the API key (images, videos) can't be loaded by <img src> directly, so
 * it's fetched as a blob and shown through an object URL.
 */
export function useBlobUrls(fetchUrl: (id: string) => Promise<string>, ids: string[]): Record<string, string> {
  const [urls, setUrls] = useState<Record<string, string>>({});
  const requested = useRef(new Set<string>());
  const live = useRef(urls);
  live.current = urls;

  useEffect(() => {
    for (const id of ids) {
      if (requested.current.has(id)) continue;
      requested.current.add(id);
      fetchUrl(id)
        .then((url) => setUrls((u) => ({ ...u, [id]: url })))
        .catch(() => requested.current.delete(id));
    }
  }, [fetchUrl, ids]);

  // Release blob memory when leaving the page.
  useEffect(() => () => Object.values(live.current).forEach((u) => URL.revokeObjectURL(u)), []);

  return urls;
}
