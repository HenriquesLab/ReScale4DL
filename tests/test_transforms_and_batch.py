from __future__ import annotations

import os

import numpy as np
import pytest
from tifffile import imread, imwrite

from rescale4dl import batch, blurring, downscaling, upscaling


def test_gaussian_blur_preserves_shape_and_smooths():
    img = np.zeros((5, 5), dtype=np.float32)
    img[2, 2] = 1.0

    blurred = blurring.gaussian_blur(img, sigma=1.0)

    assert blurred.shape == img.shape
    assert 0 < blurred[2, 2] < 1


def test_downscaling_dispatch_and_label_downscale(monkeypatch):
    calls = {}

    def fake_rebin(img, factor, mode="sum"):
        calls["rebin"] = (img.copy(), factor, mode)
        return np.array([[img.sum()]])

    class FakeConvolution:
        def run(self, img, kernel):
            calls["conv"] = (img.dtype, kernel.copy())
            return img + kernel.sum()

    monkeypatch.setattr(downscaling, "rebin_2d", fake_rebin)
    monkeypatch.setattr(downscaling, "Convolution", FakeConvolution)

    img = np.arange(16, dtype=np.float32).reshape(4, 4)

    assert downscaling.binning_img(img, 2, keep_dims=False, mode="mean")[0, 0] == img.sum()
    assert calls["rebin"][1:] == (2, "mean")

    blurred = downscaling.binning_img(img, 2, keep_dims=True, mode="invalid")
    assert blurred.dtype == np.float32
    assert calls["conv"][0] == np.float32
    np.testing.assert_array_equal(calls["conv"][1], np.ones((2, 2), dtype=np.float32))

    labels = np.array([[0, 1], [2, 3]], dtype=np.uint16)
    assert downscaling.binning_label(labels, 2).dtype == np.uint16


@pytest.mark.parametrize("func_name,keep_dims,runner_args", [
    ("upsample_img", False, (0, 0, 2, 2)),
    ("upsample_img", True, (0, 0, 2, 2, 0)),
    ("upsample_labels", False, (0, 0, 2, 2)),
    ("upsample_labels", True, (0, 0, 2, 2, 0)),
])
def test_upscaling_wrappers_call_expected_nanopyx_runner(monkeypatch, func_name, keep_dims, runner_args):
    calls = []

    class FakeRunner:
        def run(self, arr, *args):
            calls.append((arr.dtype, args))
            return np.expand_dims(arr + 1, axis=0)

    monkeypatch.setattr(upscaling, "interp_cr", FakeRunner)
    monkeypatch.setattr(upscaling, "magnify_cr", FakeRunner)
    monkeypatch.setattr(upscaling, "interp_nn", FakeRunner)
    monkeypatch.setattr(upscaling, "magnify_nn", FakeRunner)

    arr = np.array([[1, 2], [3, 4]], dtype=np.uint16)
    result = getattr(upscaling, func_name)(arr, 2, keep_dims=keep_dims)

    assert calls == [(np.float32, runner_args)]
    assert result.shape == arr.shape
    expected_dtype = np.uint16 if func_name == "upsample_labels" else np.float32
    assert result.dtype == expected_dtype


def test_rescale_image_dispatch(monkeypatch):
    monkeypatch.setattr(batch, "binning_img", lambda image, factor, keep_dims, mode: image - factor)
    monkeypatch.setattr(batch, "upsample_img", lambda image, factor, keep_dims: image + factor)

    image = np.array([[3]], dtype=np.float32)

    assert batch.rescale_image(image, 2, "down")[0, 0] == 1
    assert batch.rescale_image(image, 2, "up")[0, 0] == 5
    with pytest.raises(ValueError, match="Invalid scale mode"):
        batch.rescale_image(image, 2, "sideways")


def test_blur_batch_writes_images_and_labels(tmp_path):
    dataset_root = tmp_path / "Raw" / "Dataset"
    (dataset_root / "Images").mkdir(parents=True)
    (dataset_root / "Labels").mkdir()
    (tmp_path / "Processed").mkdir()
    img = np.arange(16, dtype=np.float32).reshape(4, 4)
    lbl = (img > 4).astype(np.uint16)
    imwrite(dataset_root / "Images" / "a.tif", img)
    imwrite(dataset_root / "Labels" / "a.tif", lbl)

    batch.blur_batch(str(dataset_root), "Dataset", 0.5)

    out = tmp_path / "Processed" / "Dataset_blurred_0.5"
    assert (out / "Images" / "a.tif").exists()
    np.testing.assert_array_equal(imread(out / "Labels" / "a.tif"), lbl)


def test_downsample_and_upsample_batch_with_mocked_transforms(tmp_path, monkeypatch):
    dataset_root = tmp_path / "Raw" / "Dataset"
    (dataset_root / "Images").mkdir(parents=True)
    (dataset_root / "Labels").mkdir()
    (tmp_path / "Processed").mkdir()
    img = np.arange(25, dtype=np.float32).reshape(5, 5)
    lbl = np.arange(25, dtype=np.uint16).reshape(5, 5)
    imwrite(dataset_root / "Images" / "a.tif", img)
    imwrite(dataset_root / "Labels" / "a.tif", lbl)

    monkeypatch.setattr(batch, "binning_img", lambda image, factor, keep_dims, mode: image[:2, :2])
    monkeypatch.setattr(batch, "binning_label", lambda image, factor: image[:2, :2].astype(np.uint16))
    batch.downsample_batch(str(dataset_root), "Dataset", 2, keep_dims=False, mode="mean")
    assert (tmp_path / "Processed" / "Dataset_downsampled_2_mode_mean_diff_dims" / "Images" / "a.tif").exists()

    monkeypatch.setattr(batch, "upsample_img", lambda image, factor, keep_dims: np.repeat(image, 2, axis=0))
    monkeypatch.setattr(batch, "upsample_labels", lambda image, factor, keep_dims: image.astype(np.uint16))
    batch.upsample_batch(str(dataset_root), "Dataset", 2, keep_dims=True)
    assert (tmp_path / "Processed" / "Dataset_upsampled_2_same_dims" / "Labels" / "a.tif").exists()


def test_process_batch_and_all_datasets_dispatch(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(batch, "upsample_batch", lambda *args, **kwargs: calls.append(("up", args, kwargs)))
    monkeypatch.setattr(batch, "downsample_batch", lambda *args, **kwargs: calls.append(("down", args, kwargs)))
    monkeypatch.setattr(batch, "blur_batch", lambda *args, **kwargs: calls.append(("blur", args, kwargs)))

    batch.process_batch("input", "Dataset", [2], [3], [0.5], modes=["sum"])

    assert [call[0] for call in calls] == ["up", "up", "down", "down", "blur"]

    datasets = tmp_path / "Datasets"
    (datasets / "A").mkdir(parents=True)
    batch.process_all_datasets(str(datasets), [2], [], [], modes=["sum"])
    assert any(call[1][1] == "A" for call in calls)


def test_rescale_and_crop_image_writes_scaled_and_final(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    imwrite(input_dir / "img.tif", np.arange(16, dtype=np.float32).reshape(4, 4))
    monkeypatch.setattr(batch, "rescale_image", lambda image, factor, mode: image)

    batch.rescale_and_crop_image(str(input_dir), str(output_dir), 2, "up", (2, 2), True)

    assert os.path.exists(output_dir / "scaled" / "scaled_up_2_img.tif")
    np.testing.assert_array_equal(imread(output_dir / "final" / "processed_up_2_img.tif"), np.array([[5, 6], [9, 10]], dtype=np.float32))
