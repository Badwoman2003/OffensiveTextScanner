import clsx from "clsx";
import type { ScanResultRow } from "../lib/types";
import { Badge, type Tone } from "./Badge";

function tone(prob: number): Tone {
  if (prob >= 0.8) return "danger";
  if (prob >= 0.5) return "warn";
  return "safe";
}

export function ResultCard({ row, onFeedback }: { row: ScanResultRow; onFeedback?: () => void }) {
  const t = tone(row.prob_offensive);
  return (
    <div
      className={clsx(
        "rounded-lg border p-3 space-y-1",
        t === "danger"
          ? "border-red-200 bg-red-50"
          : t === "warn"
          ? "border-amber-200 bg-amber-50"
          : "border-emerald-200 bg-emerald-50"
      )}
    >
      <div className="flex items-center gap-2">
        <Badge tone={t}>{(row.prob_offensive * 100).toFixed(0)}%</Badge>
        <span className="text-xs text-slate-500">{row.modality_used}</span>
        {onFeedback && (
          <button className="ml-auto text-xs underline text-slate-500 hover:text-slate-800" onClick={onFeedback}>
            mark as FP
          </button>
        )}
      </div>
      <p className="text-sm text-slate-900 line-clamp-3">{row.text || row.ocr_text}</p>
      {row.ocr_text && row.text && row.ocr_text !== row.text && (
        <p className="text-xs text-slate-500">OCR: {row.ocr_text}</p>
      )}
    </div>
  );
}
