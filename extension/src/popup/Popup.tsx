import { useCallback, useEffect, useState } from "react";
import clsx from "clsx";
import { getHistory, getSettings } from "../lib/storage";
import type { HistoryEntry, Settings } from "../lib/types";

type Status = "idle" | "scanning" | "done" | "error";

export function Popup() {
  const [status, setStatus] = useState<Status>("idle");
  const [message, setMessage] = useState<string>("");
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [settings, setSettings] = useState<Settings | null>(null);

  useEffect(() => {
    void getSettings().then(setSettings);
    void getHistory().then(setHistory);
  }, []);

  const handleScan = useCallback(async () => {
    setStatus("scanning");
    setMessage("Collecting page content...");
    try {
      await chrome.runtime.sendMessage({ type: "SCAN_PAGE" });
      setStatus("done");
      setMessage("Scan dispatched. See the floating card on the page.");
      void getHistory().then(setHistory);
    } catch (e) {
      setStatus("error");
      setMessage(e instanceof Error ? e.message : String(e));
    }
  }, []);

  const openOptions = () => chrome.runtime.openOptionsPage();

  const latest = history[0];

  return (
    <div className="p-4 space-y-3 bg-white dark:bg-slate-950 text-slate-900 dark:text-slate-100">
      <header className="flex items-center justify-between">
        <h1 className="text-base font-semibold">OTS Scanner</h1>
        <button className="text-xs text-slate-500 hover:text-slate-900 dark:hover:text-white" onClick={openOptions}>
          Settings
        </button>
      </header>

      <button
        className={clsx(
          "w-full py-2 rounded-lg font-medium transition shadow-sm",
          status === "scanning"
            ? "bg-slate-200 text-slate-600 dark:bg-slate-800 dark:text-slate-400 cursor-wait"
            : "bg-emerald-600 hover:bg-emerald-700 text-white"
        )}
        disabled={status === "scanning"}
        onClick={handleScan}
      >
        {status === "scanning" ? "Scanning..." : "Scan this page"}
      </button>

      {message && (
        <p className={clsx("text-xs", status === "error" ? "text-red-600" : "text-slate-500")}>{message}</p>
      )}

      <section className="border-t border-slate-200 dark:border-slate-800 pt-3">
        <h2 className="text-xs uppercase tracking-wide text-slate-500 mb-2">Latest</h2>
        {latest ? (
          <div className="text-xs">
            <div className="truncate" title={latest.url}>
              {latest.url}
            </div>
            <div className="flex items-center gap-2 mt-1">
              <span
                className={clsx(
                  "inline-block h-2 w-2 rounded-full",
                  latest.offensiveCount > 0 ? "bg-red-500" : "bg-emerald-500"
                )}
              />
              <span>
                {latest.offensiveCount} / {latest.totalCount} flagged
              </span>
              <span className="ml-auto text-slate-400">{new Date(latest.scannedAt).toLocaleTimeString()}</span>
            </div>
          </div>
        ) : (
          <p className="text-xs text-slate-400">No scans yet.</p>
        )}
      </section>

      {settings && (
        <footer className="text-[10px] text-slate-400 flex justify-between pt-2 border-t border-slate-200 dark:border-slate-800">
          <span className="truncate max-w-[180px]" title={settings.apiBase}>
            API: {settings.apiBase}
          </span>
          <span>thr {(settings.threshold * 100).toFixed(0)}%</span>
        </footer>
      )}
    </div>
  );
}
