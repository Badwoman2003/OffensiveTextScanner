import { defaultSettings, type HistoryEntry, type Settings } from "./types";

const SETTINGS_KEY = "ots.settings";
const HISTORY_KEY = "ots.history";
const MAX_HISTORY = 200;

export async function getSettings(): Promise<Settings> {
  const obj = await chrome.storage.local.get(SETTINGS_KEY);
  return { ...defaultSettings, ...(obj[SETTINGS_KEY] ?? {}) };
}

export async function setSettings(patch: Partial<Settings>): Promise<Settings> {
  const cur = await getSettings();
  const next = { ...cur, ...patch };
  await chrome.storage.local.set({ [SETTINGS_KEY]: next });
  return next;
}

export async function pushHistory(entry: HistoryEntry): Promise<void> {
  const obj = await chrome.storage.local.get(HISTORY_KEY);
  const cur: HistoryEntry[] = obj[HISTORY_KEY] ?? [];
  cur.unshift(entry);
  await chrome.storage.local.set({ [HISTORY_KEY]: cur.slice(0, MAX_HISTORY) });
}

export async function getHistory(): Promise<HistoryEntry[]> {
  const obj = await chrome.storage.local.get(HISTORY_KEY);
  return obj[HISTORY_KEY] ?? [];
}

export async function clearHistory(): Promise<void> {
  await chrome.storage.local.set({ [HISTORY_KEY]: [] });
}
