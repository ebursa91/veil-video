import assert from "node:assert/strict";
import test from "node:test";
import { releaseVideo, retentionLabel, retentionNotice } from "../veil/static/retention.mjs";

const iso = (date) => date.toISOString();

test("the retention notice reads the API policy rather than assuming seven days", () => {
  assert.equal(retentionNotice(7), "Finished video files are removed after 7 days. Job history is kept.");
  assert.equal(retentionNotice(1), "Finished video files are removed after 1 day. Job history is kept.");
  assert.equal(retentionNotice(30), "Finished video files are removed after 30 days. Job history is kept.");
});

test("disabled and unavailable policies are distinct", () => {
  assert.equal(retentionNotice(0), "Automatic file removal is off. Delete videos to free disk space.");
  for (const value of [null, undefined, "7", false, NaN, Infinity, -1]) assert.equal(retentionNotice(value), "");
});

test("finished jobs show the supplied expiration instant, including its UTC offset", () => {
  for (const status of ["completed", "failed", "cancelled"]) {
    assert.equal(retentionLabel({ status, expires_at: "2026-10-23T12:30:00+02:00" }, iso), "Files expire 2026-10-23T10:30:00.000Z");
  }
});

test("active jobs do not show a finished-file deadline", () => {
  for (const status of ["queued", "processing", "unknown"]) {
    assert.equal(retentionLabel({ status, expires_at: "2026-10-23T10:30:00Z" }, iso), "");
  }
});

test("archived history shows the actual archival date rather than an old expiration", () => {
  assert.equal(retentionLabel({
    status: "archived", previous_status: "completed",
    expires_at: "2026-10-23T10:30:00Z", archived_at: "2026-10-23T10:31:02Z",
  }, iso), "Archived 2026-10-23T10:31:02.000Z");
});

test("legacy and invalid dates do not become invented deadlines or Invalid Date labels", () => {
  for (const value of [undefined, null, "", "not-a-date", "2026-99-99T12:00:00Z", 0]) {
    assert.equal(retentionLabel({ status: "completed", expires_at: value }, iso), "");
    assert.equal(retentionLabel({ status: "archived", archived_at: value }, iso), "");
  }
  assert.equal(retentionLabel({ status: "completed", created_at: "2026-10-16T12:00:00Z" }, iso), "");
});

test("releasing an opened video pauses, removes its source, and unloads the media", () => {
  const calls = [];
  let source = "/api/jobs/example/preview";
  const video = {
    hasAttribute: (name) => name === "src" && source !== null,
    pause: () => calls.push("pause"),
    removeAttribute: (name) => { calls.push(`remove:${name}`); source = null; },
    load: () => calls.push("load"),
  };
  releaseVideo(video);
  assert.equal(source, null);
  assert.deepEqual(calls, ["pause", "remove:src", "load"]);
  releaseVideo(video);
  assert.deepEqual(calls, ["pause", "remove:src", "load"]);
});
