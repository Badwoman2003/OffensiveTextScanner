// Shared types mirroring backend/schemas/scan.py

export type Modality = "text_only" | "image_only" | "both";

export interface ScanResultRow {
  text: string;
  ocr_text: string;
  label: 0 | 1;
  prob_offensive: number;
  modality_used: Modality;
}

export interface ScanResponse {
  job_id: string;
  status: "success" | "failed";
  results: ScanResultRow[];
  offensive_count: number;
  summary: Record<string, number | string>;
}

export interface Settings {
  apiBase: string;
  apiToken: string;
  threshold: number;
  allowImages: boolean;
  scanAutomatically: boolean;
}

export const defaultSettings: Settings = {
  apiBase: "http://127.0.0.1:8080",
  apiToken: "dev-token-change-me",
  threshold: 0.5,
  allowImages: true,
  scanAutomatically: false,
};

export interface HistoryEntry {
  url: string;
  scannedAt: string;
  offensiveCount: number;
  totalCount: number;
  jobId: string;
}

export type RuntimeMessage =
  | { type: "SCAN_PAGE" }
  | { type: "SCAN_RESULT"; response: ScanResponse }
  | { type: "SCAN_ERROR"; error: string }
  | { type: "FEEDBACK"; jobId: string; itemIndex: number; userLabel: 0 | 1; comment?: string };
