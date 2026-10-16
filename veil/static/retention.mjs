const finished = new Set(["completed", "failed", "cancelled"]);
const localDate = new Intl.DateTimeFormat(undefined, {
  year: "numeric", month: "short", day: "numeric", hour: "numeric", minute: "2-digit", timeZoneName: "short",
});

export function retentionNotice(days) {
  if (typeof days !== "number" || !Number.isFinite(days) || days < 0) return "";
  if (days === 0) return "Automatic file removal is off. Delete videos to free disk space.";
  return `Finished video files are removed after ${days} ${days === 1 ? "day" : "days"}. Job history is kept.`;
}

export function retentionLabel(job, formatDate = (date) => localDate.format(date)) {
  const archived = job.status === "archived";
  if (!archived && !finished.has(job.status)) return "";
  const value = archived ? job.archived_at : job.expires_at;
  if (typeof value !== "string" || !value.trim()) return "";
  const date = new Date(value);
  if (!Number.isFinite(date.getTime())) return "";
  return `${archived ? "Archived" : "Files expire"} ${formatDate(date)}`;
}

export function releaseVideo(video) {
  if (!video.hasAttribute("src")) return;
  video.pause();
  video.removeAttribute("src");
  video.load();
}
