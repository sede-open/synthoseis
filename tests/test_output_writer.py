"""Round-trip tests for output_writer.write_volume_to_zarr / read_volume_from_zarr."""
import numpy as np
import pytest
import tempfile, os
import xarray as xr
from datagenerator.output_writer import write_volume_to_zarr, read_volume_from_zarr


def test_roundtrip_values(tmp_path):
    arr = np.random.rand(10, 10, 20).astype("float32")
    path = str(tmp_path / "test.zarr")
    write_volume_to_zarr(arr, path, name="amplitude",
                         dims=("inline", "crossline", "time"))
    result = read_volume_from_zarr(path, name="amplitude")
    np.testing.assert_array_equal(result, arr)


def test_roundtrip_attrs(tmp_path):
    arr = np.zeros((5, 5, 10), dtype="float32")
    path = str(tmp_path / "test_attrs.zarr")
    write_volume_to_zarr(arr, path, name="amplitude",
                         dims=("inline", "crossline", "time"),
                         attrs={"angle_deg": 15, "sample_rate_ms": 4})
    ds = xr.open_zarr(path)
    assert ds.attrs.get("angle_deg") == 15 or \
           ds["amplitude"].attrs.get("angle_deg") == 15


def test_roundtrip_label_volume(tmp_path):
    arr = np.zeros((8, 8, 16), dtype="uint8")
    arr[2:5, 2:5, 4:8] = 1
    path = str(tmp_path / "labels.zarr")
    write_volume_to_zarr(arr, path, name="label",
                         dims=("inline", "crossline", "time"))
    result = read_volume_from_zarr(path, name="label")
    np.testing.assert_array_equal(result, arr)


def test_open_zarr_returns_dataset(tmp_path):
    arr = np.ones((4, 4, 8), dtype="float32")
    path = str(tmp_path / "ds.zarr")
    write_volume_to_zarr(arr, path, name="amplitude",
                         dims=("inline", "crossline", "time"))
    ds = xr.open_zarr(path)
    assert isinstance(ds, xr.Dataset)
    assert "amplitude" in ds


# ---------------------------------------------------------------------------
# OPT-2: Blosc-Zstd compression tests
# ---------------------------------------------------------------------------

def _dir_size(path: str) -> int:
    """Return total bytes of all files under *path*."""
    total = 0
    for dp, _, filenames in os.walk(path):
        for f in filenames:
            total += os.path.getsize(os.path.join(dp, f))
    return total


def test_roundtrip_compressed_float32(tmp_path):
    """Default compressor: compressed store is readable; round-trip exact."""
    arr = np.random.rand(20, 20, 40).astype("float32")
    path = str(tmp_path / "compressed_f32.zarr")
    write_volume_to_zarr(arr, path, name="amplitude",
                         dims=("inline", "crossline", "time"))
    result = read_volume_from_zarr(path, name="amplitude")
    np.testing.assert_array_equal(result, arr)


def test_roundtrip_compressed_uint8(tmp_path):
    """Compression with uint8 label volumes round-trips exactly."""
    arr = np.zeros((16, 16, 32), dtype="uint8")
    arr[2:10, 2:10, 4:20] = 1
    path = str(tmp_path / "labels.zarr")
    write_volume_to_zarr(arr, path, name="label",
                         dims=("inline", "crossline", "time"))
    result = read_volume_from_zarr(path, name="label")
    np.testing.assert_array_equal(result, arr)


def test_compression_reduces_size(tmp_path):
    """Default-compressed store should be smaller than uncompressed store."""
    # Use a large, highly-compressible array (constant slices compress well)
    rng = np.random.default_rng(0)
    # 64×64×64 float32 = 1 MB; random data still compressed by Blosc-Zstd
    arr = rng.standard_normal((64, 64, 64)).astype("float32")
    path_c = str(tmp_path / "compressed.zarr")
    path_u = str(tmp_path / "uncompressed.zarr")

    write_volume_to_zarr(arr, path_c, name="data",
                         dims=("inline", "crossline", "time"))
    write_volume_to_zarr(arr, path_u, name="data",
                         dims=("inline", "crossline", "time"),
                         compressor=None)

    size_c = _dir_size(path_c)
    size_u = _dir_size(path_u)
    assert size_c < size_u, (
        f"Compressed store ({size_c} B) should be smaller than "
        f"uncompressed store ({size_u} B)"
    )


def test_compressor_none_disables_compression(tmp_path):
    """compressor=None: store is still readable with correct values."""
    arr = np.random.rand(8, 8, 16).astype("float32")
    path = str(tmp_path / "no_compress.zarr")
    write_volume_to_zarr(arr, path, name="amplitude",
                         dims=("inline", "crossline", "time"),
                         compressor=None)
    result = read_volume_from_zarr(path, name="amplitude")
    np.testing.assert_array_equal(result, arr)


def test_write_volume_no_chunks_no_compressor(tmp_path):
    """chunks=None, compressor=None: write succeeds."""
    arr = np.ones((4, 4, 8), dtype="float32")
    path = str(tmp_path / "plain.zarr")
    write_volume_to_zarr(arr, path, name="v",
                         dims=("x", "y", "z"),
                         chunks=None, compressor=None)
    result = read_volume_from_zarr(path, name="v")
    np.testing.assert_array_equal(result, arr)


def test_write_volume_1d_array(tmp_path):
    """1-D input: write and read round-trip."""
    arr = np.arange(100, dtype="float32")
    path = str(tmp_path / "onedim.zarr")
    write_volume_to_zarr(arr, path, name="trace", dims=("sample",))
    result = read_volume_from_zarr(path, name="trace")
    np.testing.assert_array_equal(result, arr)


# ---------------------------------------------------------------------------
# Error-case tests
# ---------------------------------------------------------------------------

def test_write_volume_invalid_compressor(tmp_path):
    """Passing a plain string as compressor must raise before any data is written."""
    arr = np.ones((4, 4, 8), dtype="float32")
    path = str(tmp_path / "bad_compressor.zarr")
    with pytest.raises(Exception):
        write_volume_to_zarr(
            arr, path, name="amplitude",
            dims=("inline", "crossline", "time"),
            compressor="zstd",  # invalid: must be a codec object or None
        )


def test_write_volume_path_not_writable(tmp_path):
    """Writing to a read-only directory must raise OSError or PermissionError."""
    import stat
    ro_dir = tmp_path / "readonly"
    ro_dir.mkdir()
    ro_dir.chmod(stat.S_IRUSR | stat.S_IXUSR)  # r-x for owner only
    arr = np.ones((4, 4, 8), dtype="float32")
    path = str(ro_dir / "out.zarr")
    try:
        with pytest.raises((OSError, PermissionError)):
            write_volume_to_zarr(
                arr, path, name="amplitude",
                dims=("inline", "crossline", "time"),
            )
    finally:
        # Restore permissions so tmp_path cleanup doesn't fail
        ro_dir.chmod(stat.S_IRWXU)
