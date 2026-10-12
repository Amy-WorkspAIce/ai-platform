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

// Perfect-match score weights. Keep in sync with backend/app/catalog/__init__.py.
const POPULARITY_WEIGHT = 0.12; // per order of magnitude of downloads
const POPULARITY_CAP = 100_000_000;
const CURATED_BONUS = 0.25;
const AGE_GRACE_DAYS = 180;
const AGE_MAX_PENALTY = 1;
const AGE_FULL_DAYS = 730;

/**
 * How good a pick this is among models that fit equally well. Capability dominates
 * (doubling the size is worth 1 point); popularity and recency separate models of similar
 * size, e.g. a current 12B over a two-year-old 14B.
 */
export function matchScore(r: ModelRow, now = Date.now() / 1000): number {
  let score = Math.log2(1 + capability(r));
  score += POPULARITY_WEIGHT * Math.log10(1 + Math.min(downloads(r), POPULARITY_CAP));
  if ((r.source ?? "curated") === "curated") score += CURATED_BONUS;
  if (r.updated) {
    const ageDays = Math.max(0, (now - r.updated) / 86400 - AGE_GRACE_DAYS);
    score -= AGE_MAX_PENALTY * Math.min(1, ageDays / (AGE_FULL_DAYS - AGE_GRACE_DAYS));
  }
  return score;
}

/** "3mo ago", "2y ago" */
export function ago(epoch: number | null | undefined, now = Date.now() / 1000): string {
  if (!epoch) return "";
  const days = (now - epoch) / 86400;
  if (days < 1) return "today";
  if (days < 30) return `${Math.round(days)}d ago`;
  if (days < 365) return `${Math.round(days / 30.4)}mo ago`;
  return `${Math.round(days / 365.25)}y ago`;
}

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
    default: {
      // Perfect match: runs well first; then the best overall pick (capability, then
      // popularity and recency). A 0.6B model fits, but an 8B that also fits is far better.
      const now = Date.now() / 1000;
      return by(tier, (r) => -matchScore(r, now), (r) => -downloads(r));
    }
  }
}

/** Which models to show by how they fit this machine. */
export type FitFilter = "any" | "runs" | "gpu" | "gpu_partial";

const FIT_FILTERS: Record<FitFilter, Fit[] | null> = {
  any: null,
  runs: ["gpu", "light", "partial"],
  gpu: ["gpu", "light"],
  gpu_partial: ["gpu", "light", "partial"],
};

export type Filters = { category: string; query: string; sort: SortKey; reverse?: boolean; fit: FitFilter };

export function filterModels(rows: ModelRow[], f: Filters): ModelRow[] {
  const q = f.query.trim().toLowerCase();
  const fits = FIT_FILTERS[f.fit];
  const cmp = compare(f.sort);
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
    .filter((r) => !fits || (!r.unsupported && fits.includes(r.fit)))
    .sort(f.reverse ? (a, b) => cmp(b, a) : cmp);
}
