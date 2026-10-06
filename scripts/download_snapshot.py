"""Download a dated public NCBI snapshot atomically, recording its byte identity."""

import argparse
import os
import re
from pathlib import Path

from clinvar_classifier.annotation import make_session
from clinvar_classifier.io import sha256, utc_now, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("month", help="YYYY-MM")
    parser.add_argument("--directory", type=Path, default=Path("data/raw"))
    parser.add_argument("--expected-sha256")
    args = parser.parse_args()
    if not re.fullmatch(r"20[0-9]{2}-(0[1-9]|1[0-2])", args.month):
        parser.error("Month must be YYYY-MM.")
    url = f"https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/archive/variant_summary_{args.month}.txt.gz"
    args.directory.mkdir(parents=True, exist_ok=True)
    target = args.directory / f"variant_summary_{args.month}.txt.gz"
    if not target.exists():
        temporary = target.with_suffix(target.suffix + ".part")
        with make_session().get(url, timeout=(10, 120), stream=True) as response:
            response.raise_for_status()
            with temporary.open("wb") as stream:
                for chunk in response.iter_content(1024 * 1024):
                    stream.write(chunk)
        os.replace(temporary, target)
    digest = sha256(target)
    if args.expected_sha256 and digest != args.expected_sha256:
        raise ValueError("Downloaded snapshot hash differs from expected release identity.")
    write_json(
        target.with_suffix(target.suffix + ".json"),
        {
            "source_url": url,
            "filename": target.name,
            "sha256": digest,
            "bytes": target.stat().st_size,
            "recorded_at": utc_now(),
        },
    )
    print(digest)


if __name__ == "__main__":
    main()
