import type { Api, Fit, ModelRow, ModelTable, SortKey } from "./api";

/**
 * The model table is fetched once (prefetched when the app starts) and then filtered and
 * sorted in the browser, so changing filters or typing in search is instant.
 */
let cached: Promise<ModelTable> | null = null;

export function loadModelTable(api: Api, refresh = false): Promise<ModelTable> {
  if (!cached || refresh) {
    const request = api.catalogModels();
    cached = request;
    request.catch(() => {
      if (cached === request) cached = null; // let the next call retry
    });
  }
  return cached;
}

// Lower is better. "Runs fully on the GPU" and "small enough for the CPU" are equally good.
const FIT_TIER: Record<Fit, number> = { gpu: 0, light: 0, partial: 1, cpu: 2, unknown: 3, too_large: 4 };

function tier(r: ModelRow): number {
  if (r.unsupported) return 9;
  return FIT_TIER[r.fit] + (r.requires_token ? 5 : 0);
}

/** Memory used while running: comparable across chat, image and video models. */
const capability = (r: ModelRow) => r.needed_gb ?? r.size_gb ?? 0;
export const downloads = (r: ModelRow) => r.downloads ?? r.popularity ?? 0;

function compare(sort: SortKey): (a: ModelRow, b: ModelRow) => number {
  const by = (...keys: ((r: ModelRow) => number | string)[]) => (a: ModelRow, b: ModelRow) => {
    for (const key of keys) {
      const x = key(a);
      const y = key(b);
      if (x < y) return -1;
      if (x > y) return 1;
    }
    return 0;
  };
  const sizeOr = (fallback: number) => (r: ModelRow) => r.size_gb ?? fallback;
  switch (sort) {
    case "name":
      return by((r) => r.name.toLowerCase(), sizeOr(0));
    case "popular":
      return by((r) => -downloads(r), (r) => r.name.toLowerCase());
    case "size_asc":
      return by(sizeOr(Infinity));
    case "size_desc":
      return by((r) => -(r.size_gb ?? -1));
    case "fit":
      // Most headroom first.
      return by(tier, (r) => r.needed_gb ?? Infinity);
    default:
      // Perfect match: runs well first; then the model that uses the most of your hardware.
      // A 0.6B model fits, but if an 8B also fits, it's far more capable.
      return by(tier, (r) => -capability(r), (r) => -downloads(r));
  }
}

export type Filters = { category: string; query: string; sort: SortKey; runnable: boolean };

export function filterModels(rows: ModelRow[], f: Filters): ModelRow[] {
  const q = f.query.trim().toLowerCase();
  return rows
    .filter((r) => f.category === "all" || r.use_cases.includes(f.category))
    .filter(
      (r) =>
        !q ||
        r.name.toLowerCase().includes(q) ||
        r.tag.toLowerCase().includes(q) ||
        (r.description ?? "").toLowerCase().includes(q) ||
        (r.publisher ?? "").toLowerCase().includes(q),
    )
    .filter((r) => !f.runnable || (!r.unsupported && ["gpu", "light", "partial"].includes(r.fit)))
    .sort(compare(f.sort));
}
