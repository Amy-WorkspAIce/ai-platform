/**
 * Interface icons, drawn to the brand's icon rules: outline, 1.5px stroke, square caps,
 * 24px grid, shown at 16 or 20px in currentColor. The brand has no emoji in the interface.
 */
const PATHS = {
  menu: "M4 6h16M4 12h16M4 18h16",
  move: "M7 4L3 8l4 4M3 8h14M17 20l4-4-4-4M21 16H7",
  edit: "M4 20h4L19 9l-4-4L4 16zM13.5 6.5l4 4",
  close: "M6 6l12 12M18 6L6 18",
  image: "M3 5h18v14H3zM3 16l5-5 4 4 3-3 6 6M15.5 9.5h.01",
  attach: "M15 7l-6.5 6.5a1.6 1.6 0 0 0 3 2.2L18 9.5a3.2 3.2 0 0 0-5.2-3.8L6.3 12.2a4.8 4.8 0 0 0 7.4 6.1L20 12",
  folder: "M3 6h6l2 2h10v11H3z",
  volume: "M4 9h4l5-4v14l-5-4H4zM16 9a4 4 0 0 1 0 6M18.5 6.5a7.5 7.5 0 0 1 0 11",
  mic: "M9 6a3 3 0 0 1 6 0v5a3 3 0 0 1-6 0zM5 11a7 7 0 0 0 14 0M12 18v3",
  stop: "M7 7h10v10H7z",
  video: "M3 6h13v12H3zM16 10l5-3v10l-5-3",
  external: "M14 4h6v6M20 4l-9 9M18 14v6H4V6h6",
  // The Orbit, simplified: Amy's presence (memory notices).
  orbit: "M12 8.5a3.5 3.5 0 1 1 0 7 3.5 3.5 0 0 1 0-7zM2.5 14.6c-.6-2.2 3.3-5 8.7-6.3s10.3-.6 10.9 1.6-3.3 5-8.7 6.3-10.3.6-10.9-1.6z",
} as const;

export type IconName = keyof typeof PATHS;

export function Icon({ name, size = 16, label }: { name: IconName; size?: number; label?: string }) {
  return (
    <svg
      className="icon"
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.5}
      strokeLinecap="square"
      strokeLinejoin="miter"
      role={label ? "img" : undefined}
      aria-label={label}
      aria-hidden={label ? undefined : true}
    >
      <path d={PATHS[name]} />
    </svg>
  );
}
