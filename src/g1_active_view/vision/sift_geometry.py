"""The exact SIFT correspondence feature used by the retained checkpoint."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import cv2


def _gray(image):
    if isinstance(image, (str, Path)):
        im = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
        if im is None: raise ValueError(f"unreadable image: {image}")
    else:
        arr = np.asarray(image)
        if arr.ndim != 3 or arr.shape[2] < 3: raise ValueError(f"expected HxWx3 image, got {arr.shape}")
        arr = arr[:, :, :3]
        if arr.dtype != np.uint8:
            if np.issubdtype(arr.dtype, np.floating) and arr.max(initial=0) <= 1.0: arr = arr * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        im = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    return cv2.resize(im, (640, 480), interpolation=cv2.INTER_AREA)


def geometry(current, target):
    cv2.setRNGSeed(20260927)
    im, tm = _gray(current), _gray(target)
    detector = cv2.SIFT_create(nfeatures=1500, contrastThreshold=.02)
    c, dc = detector.detectAndCompute(im, None); t, dt = detector.detectAndCompute(tm, None)
    c = np.array([x.pt for x in c], dtype=np.float32).reshape(-1, 2)
    t = np.array([x.pt for x in t], dtype=np.float32).reshape(-1, 2)
    pairs = []
    if dc is not None and dt is not None and len(dc) > 2 and len(dt) > 2:
        forward = cv2.BFMatcher().knnMatch(dc, dt, k=2); reverse = cv2.BFMatcher().knnMatch(dt, dc, k=2)
        rev = {a.queryIdx: a.trainIdx for a, b in reverse if a.distance < .8 * b.distance}
        pairs = [(a.queryIdx, a.trainIdx) for a, b in forward if a.distance < .8 * b.distance and rev.get(a.trainIdx) == a.queryIdx]
    x = c[[a for a, b in pairs]] if pairs else np.zeros((0, 2), np.float32)
    y = t[[b for a, b in pairs]] if pairs else np.zeros((0, 2), np.float32)
    mask = np.ones(len(x), bool); H = None
    if len(x) >= 8:
        F, m = cv2.findFundamentalMat(x, y, cv2.FM_RANSAC, 2., .995)
        if m is not None and m.sum() >= 8: mask = m[:, 0].astype(bool)
        H, _ = cv2.findHomography(x, y, cv2.RANSAC, 3.)
    x, y = x[mask], y[mask]
    delta = (y - x) / np.array([640, 480]); features = []
    features.extend([min(len(pairs) / 500, 2), min(len(x) / 500, 2), len(x) / max(1, len(pairs)), float(np.mean(np.abs(im.astype(float) - tm)) / 255)])
    if len(x): features.extend(np.quantile(delta, [.1, .25, .5, .75, .9], axis=0).flatten()); features.extend([float(np.median(np.linalg.norm(delta, axis=1))), float(np.quantile(np.linalg.norm(delta, axis=1), .95))])
    else: features.extend([0.] * 12)
    for j in range(3):
        for i in range(3):
            q = (x[:, 0] >= i * 640 / 3) & (x[:, 0] < (i + 1) * 640 / 3) & (x[:, 1] >= j * 160) & (x[:, 1] < (j + 1) * 160)
            features.extend(np.median(delta[q], axis=0) if q.any() else [0, 0]); features.append(min(float(q.sum()) / 50, 2))
    pts = np.array([[xx, yy] for yy in [80, 240, 400] for xx in [100, 320, 540]], np.float32).reshape(1, -1, 2)
    flow = (cv2.perspectiveTransform(pts, H)[0] - pts[0]) / np.array([640, 480]) if H is not None else np.zeros((9, 2))
    features.extend(flow.flatten())
    return np.nan_to_num(np.array(features, np.float32), nan=0, posinf=2, neginf=-2).clip(-2, 2)

GEO_DIM = 61
FLOW_INDICES = [i for i in range(GEO_DIM) if i not in [0, 1, 2, *range(18, 43, 3)]]
