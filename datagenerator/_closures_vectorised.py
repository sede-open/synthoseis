"""Task 7 — pure-function vectorised helpers for Closures hot paths.

Pulled out of `datagenerator.Closures` so they can be unit-tested without
instantiating the full `Closures` object hierarchy. Each helper preserves
byte-for-byte parity with the Python-loop code it replaces on valid
inputs (non-negative integer label arrays with 0 = background).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def bincount_label_sizes(labels: np.ndarray) -> np.ndarray:
    """Return per-label voxel counts as a 1D int64 array of length ``max+1``.

    Equivalent to ``[(labels == x).sum() for x in range(labels.max() + 1)]``
    but one linear pass instead of ``max`` full-cube scans.

    Parameters
    ----------
    labels : ndarray
        Non-negative integer label volume (skimage/ndimage convention, 0 = bg).
    """
    flat = np.ascontiguousarray(labels).ravel()
    # bincount requires non-negative int input. skimage.measure.label outputs
    # int32/int64 non-negative; cast explicitly to be robust against float
    # inputs arriving from legacy `labels.astype('float32')` paths.
    if flat.dtype.kind == "f":
        flat = flat.astype(np.int64, copy=False)
    elif flat.dtype.kind in ("u", "i"):
        flat = flat.astype(np.int64, copy=False)
    else:
        raise TypeError(f"bincount_label_sizes: unsupported dtype {flat.dtype}")
    return np.bincount(flat)


def closure_size_filter_sizes(labels_before: np.ndarray, labels_after: np.ndarray):
    """Vectorised equivalent of the ``closure_size_filter`` size lists.

    Original code:
        s = [labels[labels == x].size for x in range(1, 1 + np.max(labels_before))]
        t = [labels[labels == x].size for x in range(1, 1 + np.max(labels_after))]

    Returns ``(s, t)`` as plain lists matching the original shapes.
    """
    s = bincount_label_sizes(labels_before)[1:].tolist()
    t = bincount_label_sizes(labels_after)[1:].tolist()
    return s, t


def relabel_consecutive(labels_clean: np.ndarray):
    """Drop-in replacement for ``Closures.parse_label_values_and_counts``.

    Replaces the Python-level ``for i in range(...): next_label = ... .min()``
    loop followed by per-label relabel scans with one ``np.unique``
    ``return_inverse=True`` pass.

    Parity contract:
    - Input must contain 0 (background); skimage label output always does.
    - Output labels are consecutive ints 0, 1, ..., N with N = number of
      distinct non-zero labels in the input.
    - Output ``label_values`` is ``[1, 2, ..., N]`` matching the original
      (which rewrites ``label_values[i] = i`` then removes 0).
    - Output dtype is preserved.
    """
    unique_vals, inverse = np.unique(labels_clean, return_inverse=True)
    new_labels = inverse.reshape(labels_clean.shape).astype(labels_clean.dtype)
    # Original emits [1, 2, ..., N]; if 0 is present len(unique_vals) = N+1.
    if unique_vals.size and unique_vals[0] == 0:
        label_values = list(range(1, unique_vals.size))
    else:
        # Background absent: original ``label_values.remove(0)`` would raise,
        # but empirically this path never fires (segment_closures always
        # produces background). Preserve safe default for synthetic inputs.
        label_values = list(range(0, unique_vals.size))
    return label_values, new_labels


def parse_closure_codes_vectorised(
    hc_closure_codes: np.ndarray,
    labels: np.ndarray,
    num: int,
    code: float = 0.1,
) -> np.ndarray:
    """Vectorised equivalent of ``Closures.parse_closure_codes``.

    Original loop:
        labels = labels.astype('float32')
        if num > 0:
            for x in range(1, num + 1):
                y = code + labels[labels == x].size
                labels[labels == x] = y
            hc_closure_codes += labels

    Replacement: one ``bincount`` + one LUT scatter. The original mutates
    ``labels`` in place; we produce an equivalent scaled volume and add it
    to ``hc_closure_codes`` without touching the input.
    """
    if num <= 0:
        return hc_closure_codes

    counts = bincount_label_sizes(labels)
    # The original only touches labels in [1..num]; higher label IDs are
    # left untouched (would never be zero-set to their old value + code),
    # but the original loop only wrote labels in that range, so any label
    # > num retained its original float value. Preserve that behaviour by
    # keeping the identity mapping above num.
    lut_size = max(counts.size, int(num) + 1)
    remap = np.zeros(lut_size, dtype=np.float32)
    # Labels 1..num get code + size; empty bins (size 0) get 0, matching
    # the original (loop body ``labels[labels == x] = y`` only fires when
    # the mask is non-empty — but ``.size`` returns 0 and ``y = code``,
    # then ``labels[labels == x] = code`` would no-op on the empty mask.
    # So our remap[empty] = 0 matches the empty-mask no-op.
    upto = min(int(num) + 1, counts.size)
    nonzero = counts[1:upto] > 0
    remap[1:upto] = np.where(nonzero, code + counts[1:upto].astype(np.float32), 0.0)
    # Labels > num: original did not touch, left as-is (float label id).
    # Our behaviour differs there. In production this branch never fires
    # because `num = measure.label(..., return_num=True)` = max label.
    if lut_size > int(num) + 1:
        upper = np.arange(int(num) + 1, lut_size, dtype=np.float32)
        remap[int(num) + 1 : lut_size] = upper

    labels_idx = labels.astype(np.int64, copy=False) if labels.dtype.kind == "f" else labels.astype(np.int64, copy=False)
    hc_closure_codes += remap[labels_idx]
    return hc_closure_codes


def assign_fluid_types_vectorised(
    labels_clean: np.ndarray,
    closure_segments: np.ndarray,
    fluid_type_code: np.ndarray,
):
    """Vectorised equivalent of ``Closures.assign_fluid_types`` inner loop.

    Replaces a per-label ``labels_clean == i`` scan with one LUT lookup.

    Parameters
    ----------
    labels_clean : ndarray
        Consecutive-integer label volume (0 = background).
    closure_segments : ndarray
        Original closure_segments volume, used to gate brine assignment
        matching the original's ``np.logical_and(labels_clean == i, _closure_segments > 0)``.
    fluid_type_code : ndarray
        Length ``labels_clean.max() + 1`` int array, values in {0, 1, 2}.

    Returns
    -------
    (oil, gas, brine) : tuple of uint8 arrays
    """
    labels_idx = labels_clean.astype(np.int64, copy=False)
    # Gate: background voxels must not be assigned any fluid even if
    # fluid_type_code[0] happens to match a fluid code.
    fg = labels_idx > 0
    # LUT lookup per voxel; only valid at fg voxels.
    fluid_per_voxel = fluid_type_code[labels_idx]
    brine = ((fluid_per_voxel == 0) & fg & (closure_segments > 0)).astype(np.uint8)
    oil = ((fluid_per_voxel == 1) & fg).astype(np.uint8)
    gas = ((fluid_per_voxel == 2) & fg).astype(np.uint8)
    return oil, gas, brine


def bbox_for_label_and_fault(
    labels_clean: np.ndarray,
    label_id,
    fault_throw: np.ndarray,
    fault_block_val: float,
    fault_tol: float = 0.25,
    pad: int = 32,
):
    """Tight bounding box around ``(labels_clean == label_id) & |fault_throw - v| < tol``,
    padded isotropically by ``pad`` voxels and clipped to the cube shape.

    Returns a tuple of ``slice`` objects (one per axis) or ``None`` if the
    mask is empty.
    """
    mask = (labels_clean == label_id) & (np.abs(fault_throw - fault_block_val) < fault_tol)
    if not mask.any():
        return None
    # ndimage.find_objects on a bool array via labelling is overkill;
    # np.where + min/max is faster for a single connected(-ish) region.
    idx = np.argwhere(mask)
    mins = idx.min(axis=0)
    maxs = idx.max(axis=0) + 1
    shape = labels_clean.shape
    slices = tuple(
        slice(max(0, int(mn) - pad), min(int(sh), int(mx) + pad))
        for mn, mx, sh in zip(mins, maxs, shape)
    )
    return slices
