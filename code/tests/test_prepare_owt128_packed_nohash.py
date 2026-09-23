import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.prepare_owt128_packed_nohash import (
    DEFAULT_TRAIN_SEQUENCES,
    DEFAULT_VALIDATION_SEQUENCES,
    ExactTextSet,
    PackedWriter,
    REVISION,
    SOURCE,
    build_manifest,
    normalize_text,
    source_shard_url,
)


class NoHashPackedPreparationTests(unittest.TestCase):
    def test_source_shard_url_accepts_mirror_and_trailing_slash(self):
        remote_path = "plain_text/train/0007.parquet"
        suffix = (
            "/datasets/Skylion007/openwebtext/resolve/"
            f"{REVISION}/{remote_path}")
        self.assertEqual(
            source_shard_url("https://huggingface.co/", remote_path),
            "https://huggingface.co" + suffix)
        self.assertEqual(
            source_shard_url("https://hf-mirror.com///", remote_path),
            "https://hf-mirror.com" + suffix)

    def test_normalization_and_sqlite_exact_deduplication(self):
        with tempfile.TemporaryDirectory() as directory:
            database = ExactTextSet(Path(directory) / "normalized_text.sqlite3")
            first = normalize_text("  alpha\n\n\n\nbeta  ")
            same = normalize_text("alpha\n\n\nbeta")
            distinct = normalize_text("Alpha\n\nbeta")
            self.assertEqual(first, "alpha\n\nbeta")
            self.assertTrue(database.add(first))
            self.assertFalse(database.add(same))
            self.assertTrue(database.add(distinct))
            record = database.close()
            self.assertEqual(record["unique_normalized_texts"], 2)
            self.assertGreater(record["bytes"], 0)

    def test_packing_preserves_126_payload_and_split_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "validation.bin"
            writer = PackedWriter(path, 2, length=6, special=9)
            writer.add([1, 2])
            writer.add([3, 4, 5, 6, 7, 8, 0])
            record = writer.finish()
            rows = np.fromfile(path, dtype="<u4").reshape(-1, 6)
            self.assertEqual(
                rows.tolist(),
                [[9, 1, 2, 9, 3, 9], [9, 4, 5, 6, 7, 9]])
            self.assertEqual(record["tokens"], 12)
            self.assertEqual(record["bytes"], 48)
            self.assertEqual(record["discarded_tail_tokens"], 3)
            self.assertNotIn("sha256", record)

    def test_manifest_records_only_paths_bytes_and_counts(self):
        manifest = build_manifest(
            source_files=[{
                "path": "plain_text/train/0000.parquet",
                "local_path": "source_shards/0000.parquet",
                "bytes": 123,
                "rows_in_file": 7,
                "rows_read": 6,
            }],
            split_records={
                "validation": {
                    "file": "validation.bin", "sequences": 2,
                    "tokens": 256, "bytes": 1024,
                    "discarded_tail_tokens": 1,
                },
                "train": {
                    "file": "train.bin", "sequences": 3,
                    "tokens": 384, "bytes": 1536,
                    "discarded_tail_tokens": 2,
                },
            },
            documents={"validation": 1, "train": 2},
            skipped_empty=1,
            skipped_duplicate=1,
            document_index={
                "path": "documents.jsonl", "bytes": 91, "records": 3},
            deduplication_database={
                "path": "normalized_text.sqlite3", "bytes": 4096,
                "unique_normalized_texts": 3,
            },
            tokenizer_files=[{"path": "tokenizer/tokenizer.json", "bytes": 77}],
            elapsed_seconds=1.25,
            validation_target=2,
            train_target=3,
        )
        self.assertIs(manifest["content_hashes"], False)
        self.assertEqual(manifest["source"]["dataset"], SOURCE)
        self.assertEqual(manifest["source"]["revision"], REVISION)
        self.assertEqual(manifest["payload_tokens_per_sequence"], 126)
        self.assertEqual(manifest["split_rule"],
                         "source shard/row order; validation documents first until target; "
                         "subsequent documents train; documents never cross splits")

        def check_keys(value):
            if isinstance(value, dict):
                for key, child in value.items():
                    if key != "content_hashes":
                        self.assertNotIn("digest", key.lower())
                        self.assertNotIn("sha256", key.lower())
                    check_keys(child)
            elif isinstance(value, list):
                for child in value:
                    check_keys(child)

        check_keys(manifest)

    def test_default_sequence_targets_are_frozen(self):
        self.assertEqual(DEFAULT_VALIDATION_SEQUENCES, 8192)
        self.assertEqual(DEFAULT_TRAIN_SEQUENCES, 2_560_000)


if __name__ == "__main__":
    unittest.main()
