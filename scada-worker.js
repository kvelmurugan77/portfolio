/**
 * scada-worker.js — Web Worker for non-blocking SCADA CSV parsing
 *
 * Receives: { type: "parse", file: ArrayBuffer, fileName: string, delimiter: string, skipHeader: int, skipFooter: int }
 * Sends: { type: "progress", pct, msg } during parse, then { type: "done", rows, headers, fileName } on completion.
 *
 * The worker uses PapaParse (loaded via importScripts) to parse the file in chunks.
 * All row objects are built here so the main thread receives ready-to-use data.
 * This keeps the UI responsive even for multi-MB SCADA exports with hundreds
 * of thousands of rows.
 *
 * Worker URL resolution: when loaded from file:// or https://, PapaParse is
 * imported from the jsdelivr CDN. If the page is loaded from file://, the
 * worker falls back to an inline parser (see parseInline below) so the tool
 * works fully offline once cached by the browser.
 */

// Try CDN first; if importScripts fails (e.g. file:// origin in some browsers),
// fall back to the inline parser.
let PAPA_AVAILABLE = false;
try {
  importScripts("https://cdn.jsdelivr.net/npm/papaparse@5.4.1/papaparse.min.js");
  PAPA_AVAILABLE = (typeof Papa !== "undefined");
} catch (e) {
  // Will use inline parser
  PAPA_AVAILABLE = false;
}

/**
 * Inline CSV parser — handles standard quoted-CSV (with " escapes) and
 * configurable delimiters. Less feature-rich than PapaParse but enough
 * for typical SCADA exports. Used as a fallback when PapaParse can't be
 * loaded (e.g. offline file:// usage).
 */
function parseInline(text, delimiter) {
  const rows = [];
  let cur = [];
  let field = "";
  let inQuotes = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (inQuotes) {
      if (c === '"') {
        if (text[i + 1] === '"') { field += '"'; i++; }
        else inQuotes = false;
      } else field += c;
    } else {
      if (c === '"') inQuotes = true;
      else if (c === delimiter) { cur.push(field); field = ""; }
      else if (c === '\r') { /* skip */ }
      else if (c === '\n') {
        cur.push(field);
        rows.push(cur);
        cur = []; field = "";
      } else field += c;
    }
  }
  // Last field
  if (field !== "" || cur.length > 0) {
    cur.push(field);
    rows.push(cur);
  }
  return rows;
}

/**
 * Auto-detect delimiter: try comma, semicolon, tab, pipe; pick the one
 * producing the most columns on the first non-empty line.
 */
function autoDetectDelimiter(text) {
  const firstLine = (text.split(/\r?\n/).find(l => l.trim().length > 0) || "");
  const candidates = [",", "\t", ";", "|"];
  let best = ",", bestCount = -1;
  candidates.forEach(d => {
    const count = firstLine.split(d).length;
    if (count > bestCount) { bestCount = count; best = d; }
  });
  return best;
}

self.onmessage = function(e) {
  const msg = e.data;
  if (msg.type !== "parse") return;

  const { fileBuffer, fileName, delimiter, skipHeader, skipFooter } = msg;
  // Decode the ArrayBuffer as UTF-8 text
  const text = new TextDecoder("utf-8").decode(fileBuffer);

  // Resolve delimiter
  let delim = delimiter;
  if (delim === "auto" || !delim) {
    delim = autoDetectDelimiter(text);
    self.postMessage({ type: "progress", pct: 10, msg: `Auto-detected delimiter: ${delim === "\t" ? "tab" : delim}` });
  } else if (delim === "\\t") {
    delim = "\t";
  }

  // Parse rows
  let rows;
  if (PAPA_AVAILABLE) {
    const result = Papa.parse(text, {
      delimiter: delim,
      skipEmptyLines: true,
      header: false,
      worker: false,  // workers can't nest workers; we ARE the worker
      chunk: undefined  // process all at once in worker; main thread is free
    });
    rows = result.data;
  } else {
    rows = parseInline(text, delim);
  }
  self.postMessage({ type: "progress", pct: 50, msg: `Parsed ${rows.length.toLocaleString()} rows from ${fileName}` });

  // Apply skip-header / skip-footer
  const sh = skipHeader || 0;
  const sf = skipFooter || 0;
  if (rows.length <= sh) {
    self.postMessage({ type: "done", rows: [], headers: [], fileName, msg: "File has no data after skipping header rows" });
    return;
  }
  const headers = (rows[sh] || []).map(h => String(h).trim());
  let dataRows = rows.slice(sh + 1);
  if (sf > 0 && dataRows.length > sf) dataRows = dataRows.slice(0, dataRows.length - sf);

  // Build row objects (one per data row)
  const objRows = new Array(dataRows.length);
  for (let i = 0; i < dataRows.length; i++) {
    const r = dataRows[i];
    const obj = {};
    for (let j = 0; j < headers.length; j++) obj[headers[j]] = r[j];
    objRows[i] = obj;
    // Yield progress periodically
    if (i % 20000 === 0 && i > 0) {
      self.postMessage({ type: "progress", pct: 50 + Math.floor((i / dataRows.length) * 45), msg: `Building row objects: ${i.toLocaleString()} / ${dataRows.length.toLocaleString()}` });
    }
  }

  self.postMessage({
    type: "done",
    rows: objRows,
    headers: headers,
    fileName: fileName,
    msg: `Loaded ${objRows.length.toLocaleString()} rows from ${fileName}`
  });
};
