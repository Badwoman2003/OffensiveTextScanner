import type { ScanResponse, Settings } from "./types";

export class ApiError extends Error {
  constructor(message: string, public status: number) {
    super(message);
  }
}

function headers(settings: Settings, clientId: string) {
  return {
    "Content-Type": "application/json",
    Authorization: `Bearer ${settings.apiToken}`,
    "X-Client-Id": clientId,
  };
}

export async function createScan(
  settings: Settings,
  body: { text_blocks: string[]; image_urls: string[]; threshold: number; client_id: string }
): Promise<{ job_id: string }> {
  const r = await fetch(`${settings.apiBase}/api/v1/scan`, {
    method: "POST",
    headers: headers(settings, body.client_id),
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new ApiError(await r.text(), r.status);
  return r.json();
}

export async function fetchResult(settings: Settings, jobId: string, clientId: string): Promise<ScanResponse> {
  const r = await fetch(`${settings.apiBase}/api/v1/scan/${jobId}`, {
    headers: headers(settings, clientId),
  });
  if (!r.ok) throw new ApiError(await r.text(), r.status);
  return r.json();
}

export async function pollUntilDone(
  settings: Settings,
  jobId: string,
  clientId: string,
  opts: { intervalMs?: number; timeoutMs?: number; onProgress?: (p: number, status: string) => void } = {}
): Promise<ScanResponse> {
  const interval = opts.intervalMs ?? 500;
  const timeout = opts.timeoutMs ?? 60_000;
  const start = Date.now();

  while (true) {
    if (Date.now() - start > timeout) throw new ApiError("timeout waiting for scan", 408);
    const statusR = await fetch(`${settings.apiBase}/api/v1/scan/${jobId}/status`, {
      headers: headers(settings, clientId),
    });
    if (!statusR.ok) throw new ApiError(await statusR.text(), statusR.status);
    const { status, progress } = (await statusR.json()) as { status: string; progress: number };
    opts.onProgress?.(progress, status);
    if (status === "success" || status === "failed") break;
    await new Promise((res) => setTimeout(res, interval));
  }
  return fetchResult(settings, jobId, clientId);
}

export async function submitFeedback(
  settings: Settings,
  clientId: string,
  payload: { job_id: string; item_index: number; user_label: 0 | 1; comment?: string }
): Promise<void> {
  await fetch(`${settings.apiBase}/api/v1/feedback`, {
    method: "POST",
    headers: headers(settings, clientId),
    body: JSON.stringify(payload),
  });
}
