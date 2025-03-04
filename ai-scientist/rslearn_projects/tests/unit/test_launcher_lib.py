import pathlib
import zipfile

from rslp.launcher_lib import generate_combinations, make_archive


def test_make_archive(tmp_path: pathlib.Path) -> None:
    # Make sure make_archive correctly ignores the passed prefixes.
    # We make sure it works with exactly matching file as well as a subdirectory.
    exclude_prefixes = [
        "ignored_file",
        "dir/ignored_subdir",
    ]
    root_dir = tmp_path / "root"
    (root_dir / "dir" / "ignored_subdir").mkdir(parents=True, exist_ok=True)
    (root_dir / "dir" / "okay_subdir").mkdir(parents=True, exist_ok=True)
    (root_dir / "okay_file1").touch()
    (root_dir / "ignored_file").touch()
    (root_dir / "dir" / "okay_file2").touch()
    (root_dir / "dir" / "ignored_subdir" / "also_ignored").touch()
    (root_dir / "dir" / "okay_subdir" / "okay_file3").touch()

    zip_fname = str(tmp_path / "archive.zip")

    make_archive(zip_fname, str(root_dir), exclude_prefixes=exclude_prefixes)
    with zipfile.ZipFile(zip_fname) as zipf:
        members = zipf.namelist()

    assert "okay_file1" in members
    assert "dir/okay_file2" in members
    assert "dir/okay_subdir/okay_file3" in members
    assert len(members) == 3


def test_generate_combinations() -> None:
    """Test the generate_combinations function."""
    base_config = {
        "param1": 0,
        "nested_param": {
            "param2": 1,
            "param3": "k",
        },
    }
    hparams_config = {
        "param1": [1, 2],
        "nested_param": {
            "param2": [3, 4],
            "param3": "l",
        },
    }
    configs = generate_combinations(base_config, hparams_config)
    # param3 is not a list, so it should not be updated
    expected_configs = [
        {"param1": 1, "nested_param": {"param2": 3, "param3": "k"}},
        {"param1": 1, "nested_param": {"param2": 4, "param3": "k"}},
        {"param1": 2, "nested_param": {"param2": 3, "param3": "k"}},
        {"param1": 2, "nested_param": {"param2": 4, "param3": "k"}},
    ]
    assert configs == expected_configs
