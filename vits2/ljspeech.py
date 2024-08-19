import csv
from pathlib import Path


def parse_ljspeech(dataset_path: Path):
    metadata_path = dataset_path / "metadata.csv"
    with metadata_path.open() as f:
        reader = csv.reader(f, delimiter="|", quotechar=None)
        for row in reader:
            audio_path = dataset_path / "wavs" / f"{row[0]}.wav"
            yield {
                "id": row[0],
                "text_surface": row[1],
                "text_normalised": row[2],
                "audio_path": audio_path,
            }
