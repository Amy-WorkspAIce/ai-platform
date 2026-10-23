import { useEffect, useMemo, useState } from "react";
import type { VoiceLabels } from "../api";

export const UNCATEGORIZED = "Uncategorized";

/** One way to narrow a list of voices: the values an item has (none means Uncategorized). */
export type Facet<T> = { key: string; label: string; values: (item: T) => string[]; order?: string[] };
const AGES = ["Teens", "20s", "30s", "40s", "50 and over"];
const FEW = 12; // values shown before "more"
export type Sort<T> = { label: string; key: (item: T) => string | number };

const one = (v: string | number | undefined | null) => (v === undefined || v === null || v === "" ? [] : [String(v)]);

function ageGroup(age: number | undefined): string[] {
  if (!age) return [];
  if (age < 20) return ["Teens"];
  if (age >= 50) return ["50 and over"];
  return [`${Math.floor(age / 10) * 10}s`];
}

/** The facets every voice list shares, read from a voice's labels. */
export function labelFacets<T>(labels: (item: T) => VoiceLabels): Facet<T>[] {
  return [
    { key: "language", label: "Language", values: (i) => one(labels(i).language) },
    { key: "gender", label: "Gender", values: (i) => one(labels(i).gender) },
    {
      key: "gender-source",
      label: "Gender from",
      values: (i) => (labels(i).gender ? [labels(i).estimated ? "Estimated from pitch" : "Known"] : []),
    },
    { key: "age", label: "Age", values: (i) => ageGroup(labels(i).age), order: AGES },
    { key: "accent", label: "Accent", values: (i) => one(labels(i).accent) },
    { key: "region", label: "Region", values: (i) => one(labels(i).region) },
    { key: "style", label: "Style", values: (i) => one(labels(i).style) },
  ];
}

/** Gender and age for a voice card, e.g. "Female (estimated) · 23". */
export function labelLine(l: VoiceLabels): string {
  const parts = [l.gender ? `${l.gender}${l.estimated ? " (est.)" : ""}` : "", l.age ? String(l.age) : "", l.accent ?? ""];
  return parts.filter(Boolean).join(" · ");
}

type Selected = Record<string, string[]>;

/**
 * Search, sort and narrow a list of voices by facets. Values within a facet widen the list
 * (Female or Male); different facets narrow it (Female and Scottish). Each value shows how many
 * voices it would leave, given the other filters.
 */
export function useVoiceFilters<T>(
  items: T[],
  facets: Facet<T>[],
  sorts: Sort<T>[],
  text: (item: T) => string,
  resetKey?: string | null,
) {
  const [selected, setSelected] = useState<Selected>({});
  const [query, setQuery] = useState("");
  const [sort, setSort] = useState(0);
  useEffect(() => setSelected({}), [resetKey]); // another model's voices: start over
  const q = query.trim().toLowerCase();

  const valuesOf = (facet: Facet<T>, item: T) => {
    const v = facet.values(item);
    return v.length ? v : [UNCATEGORIZED];
  };
  const passes = (item: T, skip?: string) =>
    (!q || text(item).toLowerCase().includes(q)) &&
    facets.every((f) => f.key === skip || !selected[f.key]?.length || valuesOf(f, item).some((v) => selected[f.key].includes(v)));

  const matching = useMemo(() => {
    const by = sorts[sort] ?? sorts[0];
    return items
      .filter((i) => passes(i))
      .sort((a, b) => {
        const x = by.key(a);
        const y = by.key(b);
        return typeof x === "number" && typeof y === "number" ? x - y : String(x).localeCompare(String(y), undefined, { numeric: true });
      });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [items, selected, q, sort]);

  // Every facet with at least one labelled voice, its values (most common first), and counts.
  const groups = useMemo(
    () =>
      facets
        .map((f) => {
          const all = new Map<string, number>();
          const shown = new Map<string, number>();
          for (const item of items) {
            const counted = passes(item, f.key);
            for (const v of valuesOf(f, item)) {
              all.set(v, (all.get(v) ?? 0) + 1);
              if (counted) shown.set(v, (shown.get(v) ?? 0) + 1);
            }
          }
          const rank = (v: string) => (f.order ? f.order.indexOf(v) : -(all.get(v) ?? 0));
          const values = [...all.keys()]
            .filter((v) => v !== UNCATEGORIZED)
            .sort((a, b) => rank(a) - rank(b) || a.localeCompare(b, undefined, { numeric: true }));
          if (all.has(UNCATEGORIZED)) values.push(UNCATEGORIZED);
          return { facet: f, values, counts: shown, labelled: values.some((v) => v !== UNCATEGORIZED) };
        })
        .filter((g) => g.labelled),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [items, selected, q],
  );

  const toggle = (key: string, value: string) =>
    setSelected((s) => {
      const now = s[key] ?? [];
      return { ...s, [key]: now.includes(value) ? now.filter((v) => v !== value) : [...now, value] };
    });
  const active = Object.values(selected).reduce((n, v) => n + v.length, 0);
  const clear = () => setSelected({});

  return { matching, groups, selected, toggle, active, clear, query, setQuery, sort, setSort, sorts };
}

type FilterState<T> = ReturnType<typeof useVoiceFilters<T>>;

export function VoiceFilterBar<T>({ state, total, label }: { state: FilterState<T>; total: number; label: string }) {
  const { groups, selected, toggle, active, clear, query, setQuery, sort, setSort, sorts, matching } = state;
  const [expanded, setExpanded] = useState<string[]>([]);
  return (
    <div className="voice-filters">
      <div className="library-filters">
        <input type="search" placeholder={`Search ${label}`} value={query} onChange={(e) => setQuery(e.target.value)} aria-label={`Search ${label}`} />
        {sorts.length > 1 && (
          <label className="voice-sort">
            <span className="muted small">Sort</span>
            <select value={sort} onChange={(e) => setSort(Number(e.target.value))}>
              {sorts.map((s, i) => (
                <option key={s.label} value={i}>
                  {s.label}
                </option>
              ))}
            </select>
          </label>
        )}
      </div>
      {groups.length > 0 && (
        <details className="facets">
          <summary>
            Filters{active ? ` · ${active} on` : ""}
            <span className="muted small"> · showing {matching.length} of {total}</span>
          </summary>
          {groups.map(({ facet, values, counts }) => {
            // Long lists (regions) show the most common values, any picked ones, and Uncategorized.
            const open = expanded.includes(facet.key) || values.length <= FEW + 2;
            const shown = open
              ? values
              : values.filter((v, i) => i < FEW || v === UNCATEGORIZED || selected[facet.key]?.includes(v));
            return (
              <div key={facet.key} className="facet">
                <span className="facet-label small">{facet.label}</span>
                <div className="facet-values">
                  {shown.map((v) => {
                    const on = selected[facet.key]?.includes(v) ?? false;
                    const n = counts.get(v) ?? 0;
                    return (
                      <button
                        key={v}
                        className={`chip small ${on ? "active" : ""} ${v === UNCATEGORIZED ? "uncategorized" : ""}`}
                        aria-pressed={on}
                        disabled={!on && n === 0}
                        onClick={() => toggle(facet.key, v)}
                      >
                        {v} <span className="count">{n}</span>
                      </button>
                    );
                  })}
                  {!open && (
                    <button className="link small" onClick={() => setExpanded((e) => [...e, facet.key])}>
                      {values.length - shown.length} more
                    </button>
                  )}
                </div>
              </div>
            );
          })}
          {active > 0 && (
            <button className="link small" onClick={clear}>
              Clear filters
            </button>
          )}
        </details>
      )}
    </div>
  );
}
