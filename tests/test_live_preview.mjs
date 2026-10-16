import assert from "node:assert/strict";
import test from "node:test";
import { jpegDataURL, LivePreview } from "../veil/static/live-preview.mjs";

const flush = () => new Promise((resolve) => setImmediate(resolve));
const job = (frame = 1, time = 0, status = "processing") => ({ status, preview_frame: frame, preview_time: time });

function jpeg(frame = 1, time = 0) {
  return new Response(new Uint8Array([255, 216, 255, 217]), {
    headers: { "Content-Type": "image/jpeg", "X-Preview-Frame": String(frame), "X-Preview-Time": String(time) },
  });
}

function harness(t, { rejectOnAbort = false, timeoutMs = 15000, retryMs = 3000, createURL } = {}) {
  const calls = [], images = [], statuses = [], revoked = [];
  let visible = true, clock = 0, generated = 0, unauthorized = 0;
  const controller = new LivePreview({
    endpoint: "/api/jobs/test-job/live-preview",
    visible: () => visible,
    now: () => clock,
    timeoutMs,
    retryMs,
    request: (url, options) => new Promise((resolve, reject) => {
      const call = { url, options, resolve, reject, aborts: 0 };
      options.signal.addEventListener("abort", () => {
        call.aborts += 1;
        if (rejectOnAbort) reject(new DOMException("Aborted", "AbortError"));
      });
      calls.push(call);
    }),
    createURL: createURL || (() => `blob:test-${++generated}`),
    revokeURL: (url) => revoked.push(url),
    onImage: (url, snapshot) => images.push({ url, snapshot }),
    onStatus: (status) => statuses.push(status),
    onUnauthorized: () => { unauthorized += 1; },
  });
  t.after(async () => {
    controller.dispose();
    for (const call of calls) call.reject(new DOMException("Test cleanup", "AbortError"));
    await flush();
  });
  return {
    controller, calls, images, statuses, revoked,
    visible: (value) => { visible = value; },
    advance: (ms) => { clock += ms; },
    unauthorized: () => unauthorized,
    generated: () => generated,
  };
}

test("slow requests finish and coalesce notifications into the newest pending frame", async (t) => {
  const h = harness(t);
  h.controller.update(job(1));
  h.controller.update(job(2));
  h.controller.update(job(3));
  assert.equal(h.calls.length, 1);
  assert.equal(h.calls[0].options.signal.aborted, false);
  assert.equal(h.calls[0].options.credentials, "same-origin");
  assert.equal(h.calls[0].options.cache, "no-store");
  h.calls[0].resolve(jpeg(1, 12.4));
  await flush();
  assert.equal(h.calls.length, 2);
  assert.equal(h.calls[1].url, "/api/jobs/test-job/live-preview?frame=3");
  assert.equal(h.images.at(-1).snapshot.time, 12.4);
  h.calls[1].resolve(jpeg(3, 12.5));
  await flush();
  assert.equal(h.calls.length, 2);
  assert.deepEqual(h.revoked, ["blob:test-1"]);
  h.controller.dispose();
  assert.deepEqual(h.revoked, ["blob:test-1", "blob:test-2"]);
  assert.equal(h.images.at(-1).url, null);
});

test("default data URLs preserve bounded JPEG bytes without object URL state", async () => {
  const bytes = new Uint8Array(65537);
  for (let index = 0; index < bytes.length; index += 1) bytes[index] = index % 256;
  const url = await jpegDataURL(new Blob([bytes], { type: "image/jpeg" }));
  assert.ok(url.startsWith("data:image/jpeg;base64,"));
  assert.deepEqual(Buffer.from(url.split(",")[1], "base64"), Buffer.from(bytes));
  await assert.rejects(jpegDataURL(new Blob([])), /Invalid snapshot/);
  await assert.rejects(jpegDataURL(new Blob([new Uint8Array(2 * 1024 * 1024 + 1)])), /Invalid snapshot/);
});

for (const action of ["hide", "dispose", "hidden-page"]) {
  test(`${action} during async conversion cannot restore a late image`, async (t) => {
    let finishConversion;
    const h = harness(t, { createURL: () => new Promise((resolve) => { finishConversion = resolve; }) });
    h.controller.update(job(1));
    h.calls[0].resolve(jpeg(1));
    await flush();
    assert.equal(typeof finishConversion, "function");
    if (action === "hide") h.controller.setEnabled(false);
    else if (action === "dispose") h.controller.dispose();
    else { h.visible(false); h.controller.visibilityChanged(); }
    finishConversion("blob:late-conversion");
    await flush();
    assert.equal(h.images.some((image) => image.url !== null), false);
    assert.deepEqual(h.revoked, ["blob:late-conversion"]);
    assert.equal(h.calls.length, 1);
  });
}

test("one automatic retry handles unchanged metadata and stops after another failure", async (t) => {
  const h = harness(t, { retryMs: 5 });
  h.controller.update(job(1));
  h.calls[0].resolve(new Response(null, { status: 503 }));
  await new Promise((resolve) => setTimeout(resolve, 12));
  assert.equal(h.calls.length, 2);
  h.calls[1].resolve(new Response(null, { status: 503 }));
  await new Promise((resolve) => setTimeout(resolve, 12));
  assert.equal(h.calls.length, 2);
  assert.equal(h.controller.retryTimer, null);
});

test("hide and dispose clear a pending automatic retry", async (t) => {
  const h = harness(t, { retryMs: 10 });
  h.controller.update(job(1));
  h.calls[0].resolve(new Response(null, { status: 503 }));
  await flush();
  assert.notEqual(h.controller.retryTimer, null);
  h.controller.setEnabled(false);
  assert.equal(h.controller.retryTimer, null);
  await new Promise((resolve) => setTimeout(resolve, 20));
  assert.equal(h.calls.length, 1);
  h.controller.setEnabled(true);
  h.calls[1].resolve(new Response(null, { status: 503 }));
  await flush();
  assert.notEqual(h.controller.retryTimer, null);
  h.controller.dispose();
  assert.equal(h.controller.retryTimer, null);
});

test("a newer server snapshot does not trigger redundant older-frame requests", async (t) => {
  const h = harness(t);
  h.controller.update(job(1));
  h.calls[0].resolve(jpeg(4, 0.1));
  await flush();
  assert.equal(h.controller.loadedFrame, 4);
  assert.equal(h.calls.length, 1);
  h.controller.update(job(3));
  assert.equal(h.calls.length, 1);
  h.controller.update(job(5));
  assert.equal(h.calls.length, 2);
  assert.ok(h.calls[1].url.endsWith("frame=5"));
});

test("hidden pages and Hide preview stop loads; resuming requests only the latest frame", async (t) => {
  const h = harness(t);
  h.visible(false);
  h.controller.update(job(1));
  assert.equal(h.calls.length, 0);
  h.visible(true);
  h.controller.visibilityChanged();
  assert.equal(h.calls.length, 1);
  h.controller.update(job(2));
  h.visible(false);
  h.controller.visibilityChanged();
  h.controller.visibilityChanged();
  assert.equal(h.calls[0].aborts, 1);
  h.calls[0].resolve(jpeg(1)); // A late response cannot restore a hidden preview.
  await flush();
  assert.equal(h.generated(), 0);
  h.visible(true);
  h.controller.visibilityChanged();
  assert.equal(h.calls.length, 2);
  assert.ok(h.calls[1].url.endsWith("frame=2"));
  h.calls[1].resolve(jpeg(2));
  await flush();
  h.controller.setEnabled(false);
  assert.equal(h.images.at(-1).url, null);
  assert.deepEqual(h.revoked, ["blob:test-1"]);
  h.controller.update(job(3));
  assert.equal(h.calls.length, 2);
  h.controller.setEnabled(true);
  assert.equal(h.calls.length, 3);
  assert.ok(h.calls[2].url.endsWith("frame=3"));
});

for (const status of ["completed", "failed", "cancelled"]) {
  test(`${status} clears the image and ignores a late in-flight snapshot`, async (t) => {
    const h = harness(t);
    h.controller.update(job(1));
    h.calls[0].resolve(jpeg(1));
    await flush();
    h.controller.update(job(2));
    h.controller.update(job(2, 0, status));
    assert.equal(h.calls[1].aborts, 1);
    assert.equal(h.images.at(-1).url, null);
    h.calls[1].resolve(jpeg(2));
    await flush();
    assert.equal(h.generated(), 1);
    assert.deepEqual(h.revoked, ["blob:test-1"]);
    assert.equal(h.calls.length, 2);
  });
}

test("disposal clears object URLs and stops removed-job or logout requests", async (t) => {
  const h = harness(t);
  h.controller.update(job(1));
  h.calls[0].resolve(jpeg(1));
  await flush();
  h.controller.update(job(2));
  h.controller.dispose();
  h.calls[1].resolve(jpeg(2));
  await flush();
  h.controller.update(job(3));
  assert.equal(h.calls.length, 2);
  assert.equal(h.generated(), 1);
  assert.equal(h.images.at(-1).url, null);
  assert.deepEqual(h.revoked, ["blob:test-1"]);
});

test("errors retain the last edited image and retry with bounded backoff", async (t) => {
  const h = harness(t);
  h.controller.update(job(1));
  h.calls[0].resolve(jpeg(1));
  await flush();
  h.controller.update(job(2));
  h.calls[1].resolve(new Response("Unavailable", { status: 503 }));
  await flush();
  assert.equal(h.statuses.at(-1), "unavailable");
  assert.equal(h.controller.currentURL, "blob:test-1");
  assert.equal(h.revoked.length, 0);
  h.controller.update(job(3));
  assert.equal(h.calls.length, 2);
  h.advance(3000);
  h.controller.update(job(4));
  assert.equal(h.calls.length, 3);
});

test("204 is a soft wait and 401 deactivates before invoking the sign-in hook", async (t) => {
  const h = harness(t);
  h.controller.update(job(1));
  h.calls[0].resolve(new Response(null, { status: 204 }));
  await flush();
  assert.equal(h.statuses.at(-1), "waiting");
  assert.equal(h.generated(), 0);
  h.advance(2000);
  h.controller.update(job(2));
  h.calls[1].resolve(new Response(null, { status: 401 }));
  await flush();
  assert.equal(h.unauthorized(), 1);
  assert.equal(h.controller.active, false);
  assert.equal(h.calls.length, 2);
  h.controller.refresh();
  assert.equal(h.calls.length, 2);
});

test("missing preview metadata and invalid payloads never create image URLs", async (t) => {
  const h = harness(t);
  for (const value of [null, undefined, -1, NaN, 1.5]) h.controller.update({ ...job(), preview_frame: value });
  assert.equal(h.calls.length, 0);
  h.controller.update(job(1));
  h.calls[0].resolve(new Response("<html>error</html>", { headers: { "Content-Type": "text/html" } }));
  await flush();
  assert.equal(h.generated(), 0);
  h.advance(3000);
  h.controller.update(job(2));
  h.calls[1].resolve(new Response(new Uint8Array(2 * 1024 * 1024 + 1), { headers: { "Content-Type": "image/jpeg" } }));
  await flush();
  assert.equal(h.generated(), 0);
  assert.equal(h.statuses.at(-1), "unavailable");
});

test("a hung fetch times out without racing another request", async (t) => {
  const h = harness(t, { rejectOnAbort: true, timeoutMs: 5 });
  h.controller.update(job(1));
  h.controller.update(job(2));
  await new Promise((resolve) => setTimeout(resolve, 15));
  assert.equal(h.calls.length, 1);
  assert.equal(h.calls[0].aborts, 1);
  assert.equal(h.controller.inFlight, null);
  assert.equal(h.statuses.at(-1), "unavailable");
});
