from pathlib import Path

import pytest

from scripts.get_data import validate


def _write_csv(path: Path, header: list[str]) -> None:
    path.write_text(",".join(header) + "\n" + ",".join("0" for _ in header) + "\n")


def test_validate_passes_with_correct_schema(tmp_path: Path) -> None:
    _write_csv(
        tmp_path / "application_train.csv", ["SK_ID_CURR", "TARGET", "AMT_CREDIT"]
    )
    _write_csv(tmp_path / "application_test.csv", ["SK_ID_CURR", "AMT_CREDIT"])

    validate(data_dir=tmp_path)  # should not raise


def test_validate_rejects_train_without_target(tmp_path: Path) -> None:
    _write_csv(tmp_path / "application_train.csv", ["SK_ID_CURR", "AMT_CREDIT"])
    _write_csv(tmp_path / "application_test.csv", ["SK_ID_CURR", "AMT_CREDIT"])

    with pytest.raises(ValueError, match="TARGET"):
        validate(data_dir=tmp_path)


def test_validate_rejects_test_with_target(tmp_path: Path) -> None:
    _write_csv(tmp_path / "application_train.csv", ["SK_ID_CURR", "TARGET"])
    _write_csv(tmp_path / "application_test.csv", ["SK_ID_CURR", "TARGET"])

    with pytest.raises(ValueError, match="TARGET"):
        validate(data_dir=tmp_path)


def test_validate_passes_with_train_only(tmp_path: Path) -> None:
    _write_csv(
        tmp_path / "application_train.csv", ["SK_ID_CURR", "TARGET", "AMT_CREDIT"]
    )

    validate(data_dir=tmp_path)  # application_test.csv is optional


def test_validate_requires_train_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        validate(data_dir=tmp_path)


def test_validate_rejects_header_only_file(tmp_path: Path) -> None:
    (tmp_path / "application_train.csv").write_text("SK_ID_CURR,TARGET\n")
    _write_csv(tmp_path / "application_test.csv", ["SK_ID_CURR"])

    with pytest.raises(ValueError, match="no data rows"):
        validate(data_dir=tmp_path)
