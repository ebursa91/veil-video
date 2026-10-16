"""Original procedural emoji artwork and bounded, per-video face tracking.

Overlays are decorative. The caller must apply the normal expanded privacy blur
first, including during detection dropouts. Five landmarks provide approximate
2D pose, not a reconstructed 3D head or a reliable reading of human emotion.
"""

import math
import random
from dataclasses import dataclass, field
from functools import lru_cache

import cv2
import numpy as np

EXPRESSIONS = ("angry", "disgust", "fearful", "happy", "neutral", "sad", "surprised")
SPRITE_SIZE = 256
VARIANTS = 3


@dataclass(frozen=True, slots=True)
class EmojiOverlay:
    track_id: int
    box: tuple[float, float, float, float]
    center: tuple[float, float]
    size: tuple[float, float]
    roll: float
    yaw: float
    pitch: float
    class_id: int
    variant: int


@dataclass(slots=True)
class _Track:
    track_id: int
    box: np.ndarray
    landmarks: np.ndarray
    geometry: np.ndarray
    probabilities: np.ndarray
    class_id: int
    variants: dict[int, int] = field(default_factory=dict)
    missed: int = 0
    candidate: int = -1
    candidate_frames: int = 0


def _probabilities(expression) -> np.ndarray:
    values = np.asarray(getattr(expression, "probabilities", expression), dtype=np.float64)
    if values.shape != (7,) or not np.isfinite(values).all():
        return np.full(7, 1 / 7)
    values = np.clip(values, 0, 1)
    return values / values.sum() if values.sum() > 0 else np.full(7, 1 / 7)


def _face_data(face):
    box = np.asarray(face.box, dtype=np.float64)
    if box.shape != (4,) or not np.isfinite(box).all() or min(box[2:]) <= 1:
        return None
    x, y, width, height = box
    landmarks = np.asarray(face.landmarks, dtype=np.float64)
    if landmarks.shape != (5, 2) or not np.isfinite(landmarks).all():
        landmarks = np.array(
            [
                [x + width * 0.3, y + height * 0.35],
                [x + width * 0.7, y + height * 0.35],
                [x + width * 0.5, y + height * 0.55],
                [x + width * 0.35, y + height * 0.75],
                [x + width * 0.65, y + height * 0.75],
            ]
        )
    return box, landmarks


def _geometry(box, landmarks) -> np.ndarray:
    x, y, width, height = box
    # YuNet's semantic eye order is stable even when the head is upside down.
    eye_a, eye_b = landmarks[:2]
    eyes = (eye_a + eye_b) * 0.5
    mouth = landmarks[3:5].mean(axis=0)
    eye_vector = eye_b - eye_a
    distance = float(np.linalg.norm(eye_vector))
    roll = yaw = pitch = 0.0
    if distance > max(1, width * 0.05):
        axis = eye_vector / distance
        vertical = np.array([-axis[1], axis[0]])
        roll = math.degrees(math.atan2(axis[1], axis[0]))
        yaw = float(np.clip(np.dot(landmarks[2] - eyes, axis) / distance * 1.6, -0.8, 0.8))
        eye_to_mouth = float(np.dot(mouth - eyes, vertical))
        if eye_to_mouth > max(1, height * 0.05):
            ratio = np.dot(landmarks[2] - eyes, vertical) / eye_to_mouth
            pitch = float(np.clip((ratio - 0.5) * 1.5, -0.65, 0.65))
    center = np.array([x + width * 0.5, y + height * 0.5]) * 0.7 + (eyes + mouth) * 0.15
    return np.array([*center, width * 1.75, height * 1.85, roll, yaw, pitch])


def _association(previous, current) -> float | None:
    px, py, pw, ph = previous
    x, y, width, height = current
    scale = max(width, height, pw, ph)
    distance = math.hypot(x + width / 2 - px - pw / 2, y + height / 2 - py - ph / 2) / scale
    size_ratio = max(width / pw, pw / width, height / ph, ph / height)
    if distance > 0.9 or size_ratio > 3:
        return None
    intersection = max(0, min(x + width, px + pw) - max(x, px)) * max(
        0, min(y + height, py + ph) - max(y, py)
    )
    union = width * height + pw * ph - intersection
    iou = intersection / union if union > 0 else 0
    return distance + (1 - iou) * 0.35 + abs(math.log(size_ratio)) * 0.1


class EmojiTracker:
    """Stable decoration state, with no retained frames and at most 64 tracks.

    Expressions use an exponential average plus a confidence margin and a short
    sustained-change requirement. A style variant is chosen once per track and
    expression, so a smile does not randomly change artwork between frames.
    """

    def __init__(self, fps: float, *, seed: int | str | None = None, max_tracks: int = 64):
        if not math.isfinite(fps) or fps <= 0 or not 1 <= max_tracks <= 64:
            raise ValueError("A positive frame rate and 1..64 tracks are required.")
        self.hold = max(2, round(fps * 0.25))
        self.max_tracks = max_tracks
        self._pose_alpha = 1 - math.exp(-1 / (fps * 0.07))
        self._expression_alpha = 1 - math.exp(-1 / (fps * 0.22))
        self._change_frames = max(2, round(fps * 0.12))
        self._random = random.Random(seed)
        self._tracks: dict[int, _Track] = {}
        self._next_id = 1

    def _expression(self, track, probabilities):
        if probabilities.max() >= 0.35:
            track.probabilities += self._expression_alpha * (probabilities - track.probabilities)
        candidate = int(track.probabilities.argmax())
        confident = (
            candidate != track.class_id
            and track.probabilities[candidate] >= 0.43
            and track.probabilities[candidate] - track.probabilities[track.class_id] >= 0.15
        )
        if not confident:
            track.candidate, track.candidate_frames = -1, 0
        elif candidate == track.candidate:
            track.candidate_frames += 1
        else:
            track.candidate, track.candidate_frames = candidate, 1
        if track.candidate_frames >= self._change_frames:
            track.class_id = candidate
            track.candidate, track.candidate_frames = -1, 0

    def update(self, faces, expressions) -> list[EmojiOverlay]:
        faces, expressions = list(faces), list(expressions)
        if len(faces) != len(expressions):
            raise ValueError("Each detected face requires one expression prediction.")
        detections = []
        for face, expression in zip(faces, expressions, strict=True):
            data = _face_data(face)
            if data is not None:
                detections.append((*data, _probabilities(expression), float(face.score)))
        detections.sort(key=lambda item: item[3], reverse=True)
        detections = detections[: self.max_tracks]
        pairs = []
        for track_id, track in self._tracks.items():
            for index, (box, _, _, _) in enumerate(detections):
                cost = _association(track.box, box)
                if cost is not None:
                    pairs.append((cost, track_id, index))
        matched_tracks, matched_faces = set(), set()
        for _, track_id, index in sorted(pairs):
            if track_id in matched_tracks or index in matched_faces:
                continue
            track = self._tracks[track_id]
            box, landmarks, probabilities, _ = detections[index]
            geometry = _geometry(box, landmarks)
            movement = np.linalg.norm(geometry[:2] - track.geometry[:2])
            alpha = (
                max(self._pose_alpha, 0.75) if movement > max(box[2:]) * 0.35 else self._pose_alpha
            )
            # Keep full roll while crossing the -180/180 degree boundary smoothly.
            delta = (geometry[4] - track.geometry[4] + 180) % 360 - 180
            geometry[4] = track.geometry[4] + delta
            track.geometry += alpha * (geometry - track.geometry)
            track.landmarks += alpha * (landmarks - track.landmarks)
            track.box, track.missed = box.copy(), 0
            self._expression(track, probabilities)
            matched_tracks.add(track_id)
            matched_faces.add(index)
        for track_id in list(self._tracks):
            if track_id not in matched_tracks:
                self._tracks[track_id].missed += 1
                if self._tracks[track_id].missed > self.hold:
                    del self._tracks[track_id]
        for index, (box, landmarks, probabilities, _) in enumerate(detections):
            if index in matched_faces or len(self._tracks) >= self.max_tracks:
                continue
            class_id = int(probabilities.argmax()) if probabilities.max() >= 0.4 else 4
            track_id = self._next_id
            self._next_id += 1
            self._tracks[track_id] = _Track(
                track_id,
                box.copy(),
                landmarks.copy(),
                _geometry(box, landmarks),
                probabilities.copy(),
                class_id,
            )
        overlays = []
        for track in self._tracks.values():
            if track.class_id not in track.variants:
                track.variants[track.class_id] = self._random.randrange(VARIANTS)
            variant = track.variants[track.class_id]
            cx, cy, width, height, roll, yaw, pitch = track.geometry
            overlays.append(
                EmojiOverlay(
                    track.track_id,
                    tuple(track.box),
                    (cx, cy),
                    (width, height),
                    float(roll),
                    float(yaw),
                    float(pitch),
                    track.class_id,
                    variant,
                )
            )
        return overlays


def _curve(canvas, points, color, width=5):
    a, b, c = (np.asarray(point, np.float64) for point in points)
    t = np.linspace(0, 1, 28)[:, None]
    curve = (1 - t) ** 2 * a + 2 * (1 - t) * t * b + t**2 * c
    cv2.polylines(canvas, [np.rint(curve).astype(np.int32)], False, color, width, cv2.LINE_AA)


@lru_cache(maxsize=21)
def render_emoji(class_id: int, variant: int = 0) -> np.ndarray:
    """Return a read-only 256px premultiplied BGRA sprite drawn by this project."""
    if class_id not in range(7) or variant not in range(VARIANTS):
        raise ValueError("Unknown expression or emoji variant.")
    yy, xx = np.mgrid[:SPRITE_SIZE, :SPRITE_SIZE]
    distance = np.hypot(xx - 127.5, yy - 127.5)
    highlight = np.clip(1 - np.hypot(xx - 89, yy - 63) / 250, 0, 1)
    canvas = np.zeros((SPRITE_SIZE, SPRITE_SIZE, 4), np.uint8)
    canvas[:, :, 0] = 15 + highlight * (42 + variant * 5)
    canvas[:, :, 1] = 174 + highlight * 52
    canvas[:, :, 2] = 245 + highlight * 10
    canvas[:, :, 3] = np.rint(np.clip(116.5 - distance, 0, 1) * 255)
    dark, white, pink = (43, 59, 82, 255), (236, 250, 255, 255), (107, 104, 236, 255)
    cv2.circle(canvas, (128, 128), 115, (24, 151, 211, 255), 2, cv2.LINE_AA)
    shift = (-2, 0, 2)[variant]

    def eye(x, y=106, wide=False):
        cv2.ellipse(
            canvas,
            (x, y),
            (14 if wide else 9, 19 if wide else 13),
            0,
            0,
            360,
            dark,
            -1,
            cv2.LINE_AA,
        )
        cv2.circle(canvas, (x - 3, y - 5), 3 if wide else 2, white, -1, cv2.LINE_AA)

    def brow(points):
        _curve(canvas, points, dark, 6)

    if class_id == 3:  # Happy: crescent eyes and an unmistakable upward smile.
        for x in (84, 172):
            _curve(canvas, [(x - 15, 108), (x, 86 + shift), (x + 15, 108)], dark, 6)
        if variant == 1:
            _curve(canvas, [(77, 151), (128, 191), (179, 151)], dark, 7)
        else:
            cv2.ellipse(canvas, (128, 151), (49, 39 + shift), 0, 0, 180, dark, -1, cv2.LINE_AA)
            cv2.rectangle(canvas, (94, 151), (162, 161), white, -1)
            cv2.ellipse(canvas, (128, 181), (23, 8), 0, 0, 360, pink, -1, cv2.LINE_AA)
        for x in (57, 199):
            cv2.ellipse(canvas, (x, 139), (13, 7), 0, 0, 360, (103, 160, 248, 255), -1, cv2.LINE_AA)
    elif class_id == 6:  # Surprised.
        eye(84, wide=True)
        eye(172, wide=True)
        brow([(65, 74), (84, 61 - shift), (103, 74)])
        brow([(153, 74), (172, 61 - shift), (191, 74)])
        cv2.ellipse(canvas, (128, 170), (19 + variant, 27), 0, 0, 360, dark, -1, cv2.LINE_AA)
        cv2.ellipse(canvas, (128, 184), (10, 5), 0, 0, 360, pink, -1, cv2.LINE_AA)
    elif class_id == 2:  # Fearful.
        eye(84, wide=True)
        eye(172, wide=True)
        brow([(64, 81), (88, 60), (105, 77)])
        brow([(151, 77), (168, 60), (192, 81)])
        cv2.ellipse(canvas, (128, 169), (32, 20 + variant), 0, 0, 360, dark, -1, cv2.LINE_AA)
        cv2.rectangle(canvas, (108, 151), (148, 158), white, -1)
        if variant != 1:
            cv2.ellipse(
                canvas, (204, 120), (7, 13), -25, 0, 360, (239, 198, 91, 255), -1, cv2.LINE_AA
            )
    elif class_id == 0:  # Angry.
        eye(84, 111)
        eye(172, 111)
        brow([(62, 79), (84, 88), (106, 99)])
        brow([(150, 99), (172, 88), (194, 79)])
        if variant == 2:
            cv2.rectangle(canvas, (92, 160), (164, 178), dark, -1)
            cv2.rectangle(canvas, (99, 164), (157, 174), white, -1)
        else:
            _curve(canvas, [(87, 178), (128, 147 - shift), (169, 178)], dark, 7)
    elif class_id == 5:  # Sad.
        eye(84, 111)
        eye(172, 111)
        brow([(63, 87), (85, 85), (105, 74)])
        brow([(151, 74), (171, 85), (193, 87)])
        _curve(canvas, [(90, 178), (128, 150 + shift), (166, 178)], dark, 6)
        if variant != 2:
            cv2.ellipse(
                canvas,
                (77 if variant == 0 else 179, 143),
                (7, 14),
                0,
                0,
                360,
                (235, 183, 73, 255),
                -1,
                cv2.LINE_AA,
            )
    elif class_id == 1:  # Disgust: asymmetric squint, wrinkled nose, uneven mouth.
        _curve(canvas, [(68, 108), (84, 118), (101, 105)], dark, 6)
        eye(172, 109)
        brow([(63, 82), (84, 89), (106, 82)])
        brow([(151, 88), (172, 76), (193, 84)])
        _curve(canvas, [(114, 132), (128, 124), (140, 134)], dark, 4)
        _curve(canvas, [(87, 169), (120, 147 + shift), (169, 176)], dark, 7)
        if variant == 2:
            cv2.ellipse(canvas, (145, 176), (13, 7), 15, 0, 360, pink, -1, cv2.LINE_AA)
    else:  # Neutral.
        eye(84)
        eye(172)
        if variant == 2:
            brow([(66, 80), (84, 77), (102, 80)])
            brow([(154, 80), (172, 77), (190, 80)])
        _curve(canvas, [(97, 164), (128, 164 + shift), (159, 164)], dark, 6)
    # Premultiplication avoids dark fringes when the antialiased sprite is warped.
    alpha = canvas[:, :, 3:4].astype(np.uint16)
    canvas[:, :, :3] = ((canvas[:, :, :3].astype(np.uint16) * alpha + 127) // 255).astype(np.uint8)
    canvas.setflags(write=False)
    return canvas


def draw_emojis(frame: np.ndarray, overlays: list[EmojiOverlay]) -> np.ndarray:
    """Composite into clipped ROIs; memory is bounded by the visible frame area."""
    height, width = frame.shape[:2]
    source = np.array([[0, 0], [255, 0], [255, 255], [0, 255]], np.float32)
    corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], np.float64)
    for overlay in overlays:
        values = np.array(
            [*overlay.center, *overlay.size, overlay.roll, overlay.yaw, overlay.pitch]
        )
        if not np.isfinite(values).all() or min(overlay.size) <= 0:
            continue
        size = np.minimum(overlay.size, max(width, height) * 4)
        yaw, pitch = np.clip([overlay.yaw, overlay.pitch], -0.8, 0.8)
        points = corners * size * 0.5
        points[:, 0] *= 1 - abs(yaw) * 0.12
        points /= (1 + 0.12 * (corners[:, 0] * yaw + corners[:, 1] * pitch))[:, None]
        angle = math.radians(overlay.roll)
        rotation = np.array(
            [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
        )
        points = points @ rotation.T + overlay.center
        left = max(0, min(width, math.floor(points[:, 0].min()) - 2))
        right = max(0, min(width, math.ceil(points[:, 0].max()) + 2))
        top = max(0, min(height, math.floor(points[:, 1].min()) - 2))
        bottom = max(0, min(height, math.ceil(points[:, 1].max()) + 2))
        if right <= left or bottom <= top:
            continue
        transform = cv2.getPerspectiveTransform(source, (points - [left, top]).astype(np.float32))
        sprite = render_emoji(overlay.class_id, overlay.variant)
        warped = cv2.warpPerspective(
            sprite, transform, (right - left, bottom - top), flags=cv2.INTER_LINEAR
        )
        target = frame[top:bottom, left:right]
        # Small row blocks bound blending scratch space even for a close-up in 4K.
        for row in range(0, len(target), 128):
            pixels, base = warped[row : row + 128], target[row : row + 128]
            alpha = pixels[:, :, 3:4].astype(np.float32) / 255
            base[:] = np.clip(
                pixels[:, :, :3] + base.astype(np.float32) * (1 - alpha), 0, 255
            ).astype(np.uint8)
    return frame
