# OTS Chrome Extension

Manifest V3 extension written in React 18 + TypeScript + Vite + TailwindCSS.

## Development

```bash
npm install
npm run dev        # Vite dev server with HMR for popup/options
npm run build      # produces extension/dist ready for "Load unpacked"
```

Load `extension/dist` into Chrome via `chrome://extensions` → *Load unpacked*.

## Architecture

- `src/background/` — service worker: orchestrates scans and proxies to the OTS backend.
- `src/content/`    — injected script: collects DOM text + image URLs via `TreeWalker` +
                        `MutationObserver` + `IntersectionObserver`, renders the floating card.
- `src/popup/`      — React panel shown by the toolbar button.
- `src/options/`    — settings & history dashboard.
- `src/lib/`        — shared API client, storage helpers, types.
- `src/components/` — reusable React bits (badges, result cards).

The host for the backend is configurable in the options page; the extension only requests
host permissions once you've set a URL.
