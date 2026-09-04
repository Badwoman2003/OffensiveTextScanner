/**
 * Background service worker. Drives the request/response flow between:
 *   - the popup / extension button (user intent)
 *   - the content script (page data + in-page feedback)
 *   - the OTS backend (API calls)
 */
import { createScan, pollUntilDone, submitFeedback } from "../lib/api";
import { getSettings, pushHistory } from "../lib/storage";
import type { RuntimeMessage, ScanResponse } from "../lib/types";

function ensureClientId(): Promise<string> {
  return new Promise((resolve) => {
    chrome.storage.local.get("ots.clientId", (res) => {
      let id: string = res["ots.clientId"];
      if (!id) {
        id = crypto.randomUUID();
        chrome.storage.local.set({ "ots.clientId": id });
      }
      resolve(id);
    });
  });
}

async function scanActiveTab(): Promise<void> {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab?.id || !tab.url || tab.url.startsWith("chrome://")) return;

  const settings = await getSettings();
  const clientId = await ensureClientId();

  const page = await chrome.tabs.sendMessage<RuntimeMessage, { textBlocks: string[]; imageUrls: string[]; url: string }>(
    tab.id,
    { type: "SCAN_PAGE" }
  );

  try {
    const { job_id } = await createScan(settings, {
      text_blocks: page.textBlocks,
      image_urls: settings.allowImages ? page.imageUrls : [],
      threshold: settings.threshold,
      client_id: clientId,
    });

    const response = (await pollUntilDone(settings, job_id, clientId, {
      intervalMs: 500,
      timeoutMs: 60_000,
    })) as ScanResponse;

    await pushHistory({
      url: page.url,
      scannedAt: new Date().toISOString(),
      offensiveCount: response.offensive_count,
      totalCount: response.results.length,
      jobId: response.job_id,
    });

    await chrome.tabs.sendMessage(tab.id, { type: "SCAN_RESULT", response } satisfies RuntimeMessage);

    const badge = response.offensive_count.toString();
    chrome.action.setBadgeText({ tabId: tab.id, text: response.offensive_count ? badge : "" });
    chrome.action.setBadgeBackgroundColor({ color: response.offensive_count ? "#dc2626" : "#16a34a" });
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    await chrome.tabs.sendMessage(tab.id, { type: "SCAN_ERROR", error: msg } satisfies RuntimeMessage);
  }
}

chrome.action.onClicked.addListener(() => {
  void scanActiveTab();
});

chrome.runtime.onMessage.addListener((msg: RuntimeMessage, _sender, sendResponse) => {
  if (msg.type === "SCAN_PAGE") {
    void scanActiveTab().then(() => sendResponse({ ok: true }));
    return true;
  }
  if (msg.type === "FEEDBACK") {
    void (async () => {
      const s = await getSettings();
      const id = await ensureClientId();
      await submitFeedback(s, id, {
        job_id: msg.jobId,
        item_index: msg.itemIndex,
        user_label: msg.userLabel,
        comment: msg.comment,
      });
      sendResponse({ ok: true });
    })();
    return true;
  }
  return false;
});
