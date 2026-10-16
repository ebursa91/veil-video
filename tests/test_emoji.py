import gc
import math
import weakref
from dataclasses import replace
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from veil.emoji import EmojiOverlay, EmojiTracker, draw_emojis, render_emoji


def face(x=100, y=100, width=100, height=110, roll=0, nose_offset=(0, 0)):
    box = (x, y, width, height)
    points = np.array([[0.3, 0.35], [0.7, 0.35], [0.5, 0.55], [0.35, 0.75], [0.65, 0.75]])
    points[2] += nose_offset
    points = points * [width, height] + [x, y]
    angle = math.radians(roll)
    rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    center = np.array([x + width / 2, y + height / 2])
    points = (points - center) @ rotation.T + center
    return SimpleNamespace(box=box, landmarks=points, score=0.9)


def expression(class_id):
    probabilities = np.full(7, 0.01)
    probabilities[class_id] = 0.94
    return SimpleNamespace(probabilities=tuple(probabilities))


def overlay(**changes):
    item = EmojiOverlay(1, (65, 65, 70, 70), (100, 100), (140, 140), 0, 0, 0, 4, 0)
    return replace(item, **changes)


@pytest.mark.parametrize("roll", [32, 125, -155, 180])
def test_landmark_pose_matches_position_size_and_roll(roll):
    current = face(100, 80, 100, 110, roll=roll)
    item = EmojiTracker(30, seed=1).update([current], [expression(4)])[0]
    assert item.roll == pytest.approx(roll)
    assert item.center == pytest.approx((150, 135), abs=3)
    assert item.size == pytest.approx((175, 203.5))
    assert item.yaw == pytest.approx(0, abs=1e-10)
    assert item.pitch == pytest.approx(0, abs=1e-10)
    turned = EmojiTracker(30).update([face(nose_offset=(0.1, 0.05))], [expression(4)])[0]
    assert turned.yaw > 0.2
    assert turned.pitch > 0.1


def test_roll_crosses_the_angle_boundary_without_spinning():
    tracker = EmojiTracker(30)
    first = tracker.update([face(roll=179)], [expression(4)])[0]
    changed = tracker.update([face(roll=-179)], [expression(4)])[0]
    assert 0 < changed.roll - first.roll < 2


def test_pose_smoothing_reduces_jitter_but_follows_large_motion():
    tracker = EmojiTracker(30, seed=1)
    first = tracker.update([face()], [expression(4)])[0]
    small = tracker.update([face(x=102)], [expression(4)])[0]
    assert first.track_id == small.track_id
    assert 0 < small.center[0] - first.center[0] < 2
    moved = tracker.update([face(x=152)], [expression(4)])[0]
    assert moved.track_id == first.track_id
    assert moved.center[0] - small.center[0] > 30


def test_expression_hysteresis_and_variants_stay_stable():
    tracker = EmojiTracker(30, seed="test-video")
    neutral = tracker.update([face()], [expression(4)])[0]
    for index in range(20):
        item = tracker.update([face()], [expression(3 if index % 2 else 4)])[0]
        assert item.class_id == 4
        assert item.variant == neutral.variant
    for _ in range(25):
        happy = tracker.update([face()], [expression(3)])[0]
    assert happy.class_id == 3
    for _ in range(25):
        neutral_again = tracker.update([face()], [expression(4)])[0]
    assert neutral_again.class_id == 4
    assert neutral_again.variant == neutral.variant
    for _ in range(25):
        happy_again = tracker.update([face()], [expression(3)])[0]
    assert happy_again.class_id == 3
    assert happy_again.variant == happy.variant
    assert happy_again.track_id == neutral.track_id


def test_track_identity_survives_detection_order_changes_and_short_dropouts():
    tracker = EmojiTracker(30, seed=5)
    initial = tracker.update([face(x=40), face(x=400)], [expression(3), expression(4)])
    changed = tracker.update([face(x=402), face(x=42)], [expression(4), expression(3)])
    assert [item.track_id for item in changed] == [item.track_id for item in initial]
    assert [item.class_id for item in changed] == [3, 4]
    for _ in range(tracker.hold):
        held = tracker.update([], [])
        assert len(held) == 2
        assert held[0].variant == initial[0].variant
    assert tracker.update([], []) == []


def test_seed_produces_repeatable_per_track_styles():
    a, b = EmojiTracker(30, seed="one-job"), EmojiTracker(30, seed="one-job")
    for _ in range(10):
        assert a.update([face()], [expression(3)]) == b.update([face()], [expression(3)])


def test_uncertain_expression_and_invalid_landmarks_keep_a_neutral_box_overlay():
    current = face()
    current.landmarks[:] = np.nan
    item = EmojiTracker(30).update([current], [SimpleNamespace(probabilities=(1 / 7,) * 7)])[0]
    assert item.class_id == 4
    assert np.isfinite([*item.center, *item.size, item.roll]).all()
    assert item.box == current.box
    invalid = face(width=-1)
    assert EmojiTracker(30).update([invalid], [expression(4)]) == []
    with pytest.raises(ValueError, match="one expression"):
        EmojiTracker(30).update([face()], [])


def test_all_expression_classes_have_original_distinct_bounded_sprites():
    hashes = set()
    for class_id in range(7):
        for variant in range(3):
            sprite = render_emoji(class_id, variant)
            assert sprite.shape == (256, 256, 4)
            assert sprite.dtype == np.uint8
            assert not sprite.flags.writeable
            assert sprite[128, 128, 3] == 255
            assert sprite[0, 0, 3] == 0
            hashes.add(sprite.tobytes())
    assert len(hashes) == 21
    assert render_emoji.cache_info().maxsize == 21
    with pytest.raises(ValueError):
        render_emoji(7)


def test_happy_mouth_is_visibly_different_from_neutral():
    happy = render_emoji(3)[145:195, 70:185, :3]
    neutral = render_emoji(4)[145:195, 70:185, :3]
    dark = np.array([85, 95, 115])
    assert np.count_nonzero(np.all(happy < dark, axis=2)) > 2 * np.count_nonzero(
        np.all(neutral < dark, axis=2)
    )


def test_rotation_moves_features_and_preserves_background():
    original = np.full((200, 200, 3), 35, np.uint8)
    upright = draw_emojis(original.copy(), [overlay()])
    rotated = draw_emojis(original.copy(), [overlay(roll=90)])
    np.testing.assert_array_equal(rotated[:20], original[:20])
    assert rotated[100, 80, 2] < 130  # The mouth rotates to the image-left cheek.
    assert upright[100, 80, 2] > 180
    assert np.abs(rotated.astype(float) - upright).mean() > 2


def test_offscreen_and_oversized_overlays_allocate_only_visible_roi(monkeypatch):
    frame = np.zeros((80, 120, 3), np.uint8)
    calls = []
    real_warp = cv2.warpPerspective

    def limited_warp(source, transform, dimensions, **kwargs):
        assert dimensions[0] <= frame.shape[1]
        assert dimensions[1] <= frame.shape[0]
        calls.append(dimensions)
        return real_warp(source, transform, dimensions, **kwargs)

    monkeypatch.setattr(cv2, "warpPerspective", limited_warp)
    assert draw_emojis(frame, [overlay(center=(-1000, -1000), size=(30, 30))]) is frame
    assert not calls
    draw_emojis(frame, [overlay(center=(0, 0), size=(100, 100))])
    draw_emojis(frame, [overlay(center=(40, 40), size=(1e12, 1e12))])
    draw_emojis(frame, [overlay(center=(float("nan"), 20))])
    assert len(calls) == 2
    assert frame.any()


def test_tracker_memory_is_bounded_and_renderer_retains_no_frames():
    tracker = EmojiTracker(30, max_tracks=8)
    many = [face(x=index * 200) for index in range(100)]
    assert len(tracker.update(many, [expression(4)] * len(many))) == 8
    assert len(tracker._tracks) == 8
    for track in tracker._tracks.values():
        assert (
            sum(
                array.nbytes
                for array in (track.box, track.landmarks, track.geometry, track.probabilities)
            )
            < 256
        )
    frame = np.zeros((200, 200, 3), np.uint8)
    reference = weakref.ref(frame)
    draw_emojis(frame, [overlay()])
    del frame
    gc.collect()
    assert reference() is None
    for _ in range(tracker.hold + 1):
        tracker.update([], [])
    assert not tracker._tracks
    with pytest.raises(ValueError):
        EmojiTracker(0)
    with pytest.raises(ValueError):
        EmojiTracker(30, max_tracks=65)
