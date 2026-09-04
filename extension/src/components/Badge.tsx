import clsx from "clsx";

export type Tone = "safe" | "warn" | "danger";

export function Badge({ tone, children }: { tone: Tone; children: React.ReactNode }) {
  const cls = {
    safe: "bg-emerald-100 text-emerald-700",
    warn: "bg-amber-100 text-amber-800",
    danger: "bg-red-100 text-red-700",
  }[tone];
  return <span className={clsx("inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium", cls)}>{children}</span>;
}
