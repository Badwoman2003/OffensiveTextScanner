import { useEffect, useState } from "react";
import { clearHistory, getHistory, getSettings, setSettings } from "../lib/storage";
import type { HistoryEntry, Settings } from "../lib/types";

export function Options() {
  const [settings, setLocal] = useState<Settings | null>(null);
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [savedAt, setSavedAt] = useState<string>("");

  useEffect(() => {
    void getSettings().then(setLocal);
    void getHistory().then(setHistory);
  }, []);

  if (!settings) return <div className="p-6">Loading...</div>;

  const save = async (patch: Partial<Settings>) => {
    const next = await setSettings(patch);
    setLocal(next);
    setSavedAt(new Date().toLocaleTimeString());
  };

  const requestHost = async () => {
    try {
      const url = new URL(settings.apiBase);
      await chrome.permissions.request({ origins: [`${url.origin}/*`] });
    } catch {
      /* not allowed */
    }
  };

  return (
    <div className="max-w-2xl mx-auto p-6 space-y-6 text-slate-900 dark:text-slate-100">
      <header>
        <h1 className="text-2xl font-semibold">OTS Settings</h1>
        <p className="text-sm text-slate-500">
          Configure where the extension sends scanned content and how aggressive the detector should be.
        </p>
      </header>

      <section className="space-y-4 bg-white dark:bg-slate-900 rounded-xl p-5 shadow-sm">
        <label className="block">
          <span className="text-sm font-medium">Backend URL</span>
          <input
            type="url"
            className="mt-1 w-full rounded border border-slate-300 dark:border-slate-700 bg-transparent px-3 py-2 text-sm"
            value={settings.apiBase}
            onChange={(e) => save({ apiBase: e.target.value })}
            onBlur={() => void requestHost()}
          />
        </label>

        <label className="block">
          <span className="text-sm font-medium">API Token</span>
          <input
            type="password"
            className="mt-1 w-full rounded border border-slate-300 dark:border-slate-700 bg-transparent px-3 py-2 text-sm font-mono"
            value={settings.apiToken}
            onChange={(e) => save({ apiToken: e.target.value })}
          />
        </label>

        <label className="block">
          <span className="text-sm font-medium">
            Offensive threshold ({(settings.threshold * 100).toFixed(0)}%)
          </span>
          <input
            type="range"
            min={0.1}
            max={0.95}
            step={0.05}
            className="mt-2 w-full"
            value={settings.threshold}
            onChange={(e) => save({ threshold: Number(e.target.value) })}
          />
        </label>

        <div className="flex items-center gap-3">
          <input
            id="allowImages"
            type="checkbox"
            checked={settings.allowImages}
            onChange={(e) => save({ allowImages: e.target.checked })}
          />
          <label htmlFor="allowImages" className="text-sm">
            Allow sending image URLs to the backend
          </label>
        </div>

        <div className="flex items-center gap-3">
          <input
            id="autoScan"
            type="checkbox"
            checked={settings.scanAutomatically}
            onChange={(e) => save({ scanAutomatically: e.target.checked })}
          />
          <label htmlFor="autoScan" className="text-sm">
            Scan automatically when a page loads
          </label>
        </div>

        <p className="text-xs text-slate-400">Last saved: {savedAt || "not yet"}</p>
      </section>

      <section className="bg-white dark:bg-slate-900 rounded-xl p-5 shadow-sm">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold">History</h2>
          <button
            className="text-xs text-red-600 hover:underline"
            onClick={async () => {
              await clearHistory();
              setHistory([]);
            }}
          >
            Clear all
          </button>
        </div>
        <ul className="mt-3 divide-y divide-slate-200 dark:divide-slate-800 text-sm">
          {history.length === 0 && <li className="py-3 text-slate-400">No scans yet.</li>}
          {history.map((h) => (
            <li key={h.jobId} className="py-2 flex items-center gap-2">
              <span
                className={
                  "inline-block h-2 w-2 rounded-full " + (h.offensiveCount > 0 ? "bg-red-500" : "bg-emerald-500")
                }
              />
              <span className="truncate flex-1" title={h.url}>
                {h.url}
              </span>
              <span className="text-slate-500">
                {h.offensiveCount}/{h.totalCount}
              </span>
              <span className="text-slate-400 text-xs">{new Date(h.scannedAt).toLocaleString()}</span>
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
}
