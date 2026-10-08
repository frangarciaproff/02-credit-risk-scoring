"""Fetch the Home Credit Default Risk data from Kaggle.

Downloads application_train.csv and application_test.csv only — the
application-level scope agreed in SPEC.md (not the full multi-table
schema: bureau.csv, previous_application.csv, etc. — see docs/DECISIONS.md,
D1). Files land in data/raw/, which is gitignored: never commit them, and
check Kaggle's competition terms before redistributing anything beyond
code.

Requires:
  - Kaggle API credentials (~/.kaggle/kaggle.json, or the KAGGLE_USERNAME /
    KAGGLE_KEY environment variables).
  - Having accepted the competition rules at
    kaggle.com/c/home-credit-default-risk/rules — the API refuses the
    download otherwise, with an error that looks like an auth failure.

See _meta/COMMANDS_CHEATSHEET.md ("Project 02 — Kaggle data") for the exact
one-time setup commands.
"""

import argparse
import csv
import zipfile
from pathlib import Path

COMPETITION = "home-credit-default-risk"
FILES = ("application_train.csv", "application_test.csv")
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def download(data_dir: Path = DATA_DIR, force: bool = False) -> None:
    """Download the two application-level files via the Kaggle API.

    Idempotent: skips files that already exist unless `force` is True.
    Imports the `kaggle` package lazily (not at module level) because
    importing it can trigger credential checks as a side effect, which
    would break `--help` and any test that imports this module without
    Kaggle credentials configured — e.g. in CI.
    """
    from kaggle.api.kaggle_api_extended import KaggleApi

    data_dir.mkdir(parents=True, exist_ok=True)

    missing = [name for name in FILES if force or not (data_dir / name).exists()]
    if not missing:
        print(f"All files already present in {data_dir} (use --force to redownload).")
        return

    api = KaggleApi()

    for name in missing:
        print(f"Downloading {name} from competition '{COMPETITION}'...")
        try:
            api.authenticate()
            api.competition_download_file(
                competition=COMPETITION,
                file_name=name,
                path=str(data_dir),
                force=force,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download {name}. Two likely causes: Kaggle API "
                "credentials aren't set up, or you haven't accepted this "
                "competition's rules at "
                "kaggle.com/c/home-credit-default-risk/rules yet. See "
                "_meta/COMMANDS_CHEATSHEET.md ('Project 02 — Kaggle data')."
            ) from exc

        zipped = data_dir / f"{name}.zip"
        if zipped.exists():
            with zipfile.ZipFile(zipped) as zf:
                zf.extractall(path=data_dir)
            zipped.unlink()


def validate(data_dir: Path = DATA_DIR) -> None:
    """Check the downloaded file(s) match the contract the rest of this
    project relies on.

    application_train.csv is required and must have a TARGET column.
    application_test.csv is optional — it is not used anywhere in local
    train/val/test splitting (see docs/DECISIONS.md, D6), only for an
    optional Kaggle leaderboard submission — so its absence is fine (e.g.
    when it was fetched by hand rather than via `download()`). If present,
    it must NOT have a TARGET column, since that would mean it isn't
    Kaggle's real, unlabeled test file.
    """
    train_path = data_dir / "application_train.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"{train_path} not found — this file is required.")

    train_header = _read_header(train_path)
    if "TARGET" not in train_header:
        raise ValueError(
            "application_train.csv is missing TARGET — is this really the "
            "training file?"
        )

    test_path = data_dir / "application_test.csv"
    if not test_path.exists():
        print(
            "OK: application_train.csv has TARGET. application_test.csv is "
            "absent — fine for local dev (see docs/DECISIONS.md, D6); only "
            "needed for an optional Kaggle leaderboard submission."
        )
        return

    test_header = _read_header(test_path)
    if "TARGET" in test_header:
        raise ValueError(
            "application_test.csv has a TARGET column — this is not "
            "Kaggle's real test file (see docs/DECISIONS.md, D6); it must "
            "not be used for local evaluation."
        )
    print("OK: application_train.csv has TARGET, application_test.csv does not.")


def _read_header(path: Path) -> list[str]:
    """Read just the header (and confirm at least one data row exists),
    without loading the whole file — these files run ~130k-300k rows.
    """
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"{path.name} is empty.")
        if next(reader, None) is None:
            raise ValueError(f"{path.name} has a header but no data rows.")
        return header


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true", help="Redownload even if files already exist."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Where to store the downloaded files.",
    )
    args = parser.parse_args()

    download(data_dir=args.data_dir, force=args.force)
    validate(data_dir=args.data_dir)


if __name__ == "__main__":
    main()
