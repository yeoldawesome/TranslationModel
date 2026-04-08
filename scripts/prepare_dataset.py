import argparse
import pathlib
import urllib.request
import zipfile


DATASET_URL = "http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip"


def parse_args():
    parser = argparse.ArgumentParser(description="Download and extract spa-eng dataset for offline HPC training.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/spa-eng",
        help="Directory where spa.txt should be extracted.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    zip_path = output_dir / "spa-eng.zip"
    spa_txt_path = output_dir / "spa.txt"

    if spa_txt_path.exists():
        print("Dataset already prepared at", spa_txt_path)
        return

    print("Downloading dataset to", zip_path)
    urllib.request.urlretrieve(DATASET_URL, zip_path)

    print("Extracting spa.txt to", output_dir)
    with zipfile.ZipFile(zip_path, "r") as zip_handle:
        member_name = None
        for name in zip_handle.namelist():
            if name.endswith("/spa.txt") or name == "spa.txt":
                member_name = name
                break

        if member_name is None:
            raise FileNotFoundError("Could not find spa.txt in downloaded archive")

        with zip_handle.open(member_name) as source, open(spa_txt_path, "wb") as destination:
            destination.write(source.read())

    print("Prepared dataset at", spa_txt_path)


if __name__ == "__main__":
    main()
