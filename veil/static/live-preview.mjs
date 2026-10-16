"use strict";

const MAX_IMAGE_BYTES = 2 * 1024 * 1024;

export async function jpegDataURL(blob) {
  if (!blob.size || blob.size > MAX_IMAGE_BYTES) throw new Error("Invalid snapshot");
  const bytes = new Uint8Array(await blob.arrayBuffer());
  let binary = "";
  for (let offset = 0; offset < bytes.length; offset += 32768) {
    binary += String.fromCharCode(...bytes.subarray(offset, offset + 32768));
  }
  return `data:image/jpeg;base64,${btoa(binary)}`;
}

// A slower snapshot request finishes before the newest pending frame is loaded.
// Progress notifications never repeatedly cancel work already in flight.
export class LivePreview {
  constructor({ endpoint, onImage, onStatus, onUnauthorized, visible = () => !document.hidden,
    request = (...args) => fetch(...args), createURL = jpegDataURL,
    revokeURL = () => {}, now = () => Date.now(), timeoutMs = 15000, retryMs = 3000 }) {
    Object.assign(this, { endpoint, onImage, onStatus, onUnauthorized, visible, request, createURL, revokeURL, now, timeoutMs, retryMs });
    this.enabled = true;
    this.active = false;
    this.disposed = false;
    this.latest = null;
    this.loadedFrame = null;
    this.currentURL = null;
    this.inFlight = null;
    this.retryAt = 0;
    this.retryTimer = null;
    this.retriedFrame = null;
  }

  update(job) {
    this.active = job.status === "processing";
    if (!this.active || !Number.isSafeInteger(job.preview_frame) || job.preview_frame < 0) {
      this.latest = null;
      this.stop(true);
      return;
    }
    this.latest = { frame: job.preview_frame, time: Number.isFinite(job.preview_time) ? job.preview_time : null };
    this.refresh();
  }

  setEnabled(enabled) {
    this.enabled = enabled;
    if (enabled) { this.retryAt = 0; this.refresh(); }
    else this.stop(true);
  }

  visibilityChanged() {
    if (this.visible()) { this.retryAt = 0; this.refresh(); }
    else this.stop(false);
  }

  clearImage() {
    this.onImage(null, null);
    if (this.currentURL) this.revokeURL(this.currentURL);
    this.currentURL = null;
    this.loadedFrame = null;
  }

  stop(clear) {
    clearTimeout(this.retryTimer);
    this.retryTimer = null;
    if (this.inFlight && !this.inFlight.cancelled) {
      this.inFlight.cancelled = true;
      this.inFlight.controller.abort();
    }
    if (clear) this.clearImage();
  }

  dispose() {
    this.disposed = true;
    this.active = false;
    this.latest = null;
    this.stop(true);
  }

  imageFailed() {
    this.clearImage();
    this.backoff(this.retryMs, "unavailable");
  }

  backoff(delay, status) {
    this.retryAt = this.now() + delay;
    this.onStatus(status);
    clearTimeout(this.retryTimer);
    this.retryTimer = null;
    // One automatic retry per announced frame covers a quiet finalizing stage.
    // Further attempts need fresh job metadata or an explicit Hide/Show action.
    if (this.canLoad() && this.retriedFrame !== this.latest.frame) {
      this.retryTimer = setTimeout(() => {
        this.retryTimer = null;
        this.retriedFrame = this.latest?.frame;
        this.retryAt = 0;
        this.refresh();
      }, delay);
    }
  }

  canLoad() {
    return !this.disposed && this.active && this.enabled && this.visible() && this.latest;
  }

  async refresh() {
    if (!this.canLoad() || this.inFlight || (this.loadedFrame !== null && this.latest.frame <= this.loadedFrame) || this.now() < this.retryAt) return;
    clearTimeout(this.retryTimer);
    this.retryTimer = null;
    const snapshot = this.latest;
    const load = { controller: new AbortController(), cancelled: false };
    this.inFlight = load;
    const timer = setTimeout(() => load.controller.abort(), this.timeoutMs);
    if (!this.currentURL) this.onStatus("loading");
    try {
      const response = await this.request(`${this.endpoint}?frame=${encodeURIComponent(snapshot.frame)}`, {
        credentials: "same-origin", cache: "no-store", signal: load.controller.signal,
      });
      if (!this.canLoad() || load.cancelled) return;
      if (response.status === 401) {
        this.active = false;
        this.latest = null;
        this.stop(true);
        this.onUnauthorized();
        return;
      }
      if (response.status === 204) {
        this.clearImage();
        this.backoff(2000, "waiting");
        return;
      }
      const type = response.headers.get("content-type")?.split(";")[0].trim();
      if (!response.ok || type !== "image/jpeg" || Number(response.headers.get("content-length")) > MAX_IMAGE_BYTES) throw new Error("Snapshot unavailable");
      const blob = await response.blob();
      if (!this.canLoad() || load.cancelled) return;
      if (!blob.size || blob.size > MAX_IMAGE_BYTES) throw new Error("Invalid snapshot");
      const frameHeader = response.headers.get("x-preview-frame");
      const timeHeader = response.headers.get("x-preview-time");
      const actualFrame = frameHeader !== null ? Number(frameHeader) : snapshot.frame;
      const actualTime = timeHeader !== null ? Number(timeHeader) : snapshot.time;
      const frame = Number.isSafeInteger(actualFrame) && actualFrame >= 0 ? actualFrame : snapshot.frame;
      if (this.loadedFrame !== null && frame <= this.loadedFrame) {
        this.backoff(2000, "ready");
        return;
      }
      const url = await this.createURL(blob);
      if (!this.canLoad() || load.cancelled) { this.revokeURL(url); return; }
      if (load.controller.signal.aborted) { this.revokeURL(url); throw new Error("Snapshot timed out"); }
      const previous = this.currentURL;
      this.currentURL = url;
      this.loadedFrame = frame;
      this.onImage(url, { frame, time: Number.isFinite(actualTime) && actualTime >= 0 ? actualTime : null });
      if (previous) this.revokeURL(previous);
      this.onStatus("ready");
    } catch {
      if (this.canLoad() && !load.cancelled) {
        this.backoff(this.retryMs, "unavailable");
      }
    } finally {
      clearTimeout(timer);
      if (this.inFlight === load) this.inFlight = null;
      this.refresh();
    }
  }
}
