"""Prepare the fixed packed OWT-128 dataset without computing content hashes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import time


SOURCE = "Skylion007/openwebtext"
REVISION = "433fe0f44ed7894fea29c08b3202aa348ccc6369"
SOURCE_CONFIG = "plain_text"
SOURCE_SPLIT = "train"
SOURCE_SHARD_COUNT = 80
SEQUENCE_LENGTH = 128
PAYLOAD_LENGTH = 126
VOCAB_SIZE = 50257
SPECIAL_TOKEN_ID = 50256
DEFAULT_VALIDATION_SEQUENCES = 8192
DEFAULT_TRAIN_SEQUENCES = 2_560_000
DEFAULT_HF_ENDPOINT = "https://huggingface.co"


def normalize_text(text: str) -> str:
    """Keep the established OWT normalization exactly."""
    return re.sub(r"\n\n\n+", "\n\n", text).strip()


def source_shard_url(endpoint: str, remote_path: str) -> str:
    base = endpoint.rstrip("/")
    if not base:
        raise ValueError("Hugging Face endpoint must not be empty")
    return f"{base}/datasets/{SOURCE}/resolve/{REVISION}/{remote_path}"


class ExactTextSet:
    """Disk-backed exact normalized-text set using SQLite B-tree equality."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("PRAGMA journal_mode=DELETE")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute("PRAGMA temp_store=FILE")
        self.connection.execute(
            "CREATE TABLE seen ("
            "normalized_text TEXT PRIMARY KEY COLLATE BINARY"
            ") WITHOUT ROWID")
        self.count = 0

    def add(self, normalized_text: str) -> bool:
        cursor = self.connection.execute(
            "INSERT OR IGNORE INTO seen(normalized_text) VALUES (?)",
            (normalized_text,))
        inserted = cursor.rowcount == 1
        if inserted:
            self.count += 1
        return inserted

    def commit(self) -> None:
        self.connection.commit()

    def close(self) -> dict[str, object]:
        self.connection.commit()
        stored = self.connection.execute("SELECT COUNT(*) FROM seen").fetchone()[0]
        if stored != self.count:
            raise RuntimeError(
                f"exact-text database count mismatch: {stored}/{self.count}")
        self.connection.close()
        return {
            "path": self.path.name,
            "bytes": self.path.stat().st_size,
            "unique_normalized_texts": stored,
        }


class PackedWriter:
    """Write exact-count uint32 sequences with 126-token payloads."""

    def __init__(self, path: Path, sequences: int,
                 length: int = SEQUENCE_LENGTH,
                 special: int = SPECIAL_TOKEN_ID):
        if sequences <= 0:
            raise ValueError("sequence target must be positive")
        self.path = Path(path)
        self.target = sequences
        self.length = length
        self.special = special
        self.count = 0
        self.pending: list[int] = []
        self.part_path = self.path.with_suffix(self.path.suffix + ".part")
        self.handle = self.part_path.open("wb")

    def add(self, ids: list[int]) -> None:
        import numpy as np

        self.pending.extend(ids)
        self.pending.append(self.special)
        inner = self.length - 2
        count = min(len(self.pending) // inner, self.target - self.count)
        if count:
            rows = np.full(
                (count, self.length), self.special, dtype="<u4")
            rows[:, 1:-1] = np.asarray(
                self.pending[:count * inner], dtype="<u4").reshape(
                    count, inner)
            rows.tofile(self.handle)
            self.pending = self.pending[count * inner:]
            self.count += count

    def finish(self) -> dict[str, object]:
        self.handle.close()
        if self.count != self.target:
            raise RuntimeError(f"incomplete split: {self.count}/{self.target}")
        os.replace(self.part_path, self.path)
        return {
            "file": self.path.name,
            "sequences": self.count,
            "tokens": self.count * self.length,
            "bytes": self.path.stat().st_size,
            "discarded_tail_tokens": len(self.pending),
        }


def tokenizer_inventory(directory: Path) -> list[dict[str, object]]:
    return [
        {"path": str(path.relative_to(directory.parent)),
         "bytes": path.stat().st_size}
        for path in sorted(directory.iterdir()) if path.is_file()
    ]


def build_manifest(
        source_files: list[dict[str, object]],
        split_records: dict[str, dict[str, object]],
        documents: dict[str, int],
        skipped_empty: int,
        skipped_duplicate: int,
        document_index: dict[str, object],
        deduplication_database: dict[str, object],
        tokenizer_files: list[dict[str, object]],
        elapsed_seconds: float,
        validation_target: int = DEFAULT_VALIDATION_SEQUENCES,
        train_target: int = DEFAULT_TRAIN_SEQUENCES,
) -> dict[str, object]:
    return {
        "schema": "owt128-packed-no-content-hash-v1",
        "format": "owt128-uint32-v1",
        "content_hashes": False,
        "source": {
            "dataset": SOURCE,
            "revision": REVISION,
            "config": SOURCE_CONFIG,
            "split": SOURCE_SPLIT,
            "shard_order": "0000.parquet through 0079.parquet, ascending",
            "row_order": "physical row order within each parquet shard",
            "files": source_files,
        },
        "sequence_length": SEQUENCE_LENGTH,
        "payload_tokens_per_sequence": PAYLOAD_LENGTH,
        "dtype": "<u4",
        "vocab_size": VOCAB_SIZE,
        "bos_token_id": SPECIAL_TOKEN_ID,
        "eos_token_id": SPECIAL_TOKEN_ID,
        "packing": (
            "document EOS; continuous 126-token payload wrapped with "
            "BOS/EOS; split tails discarded"),
        "normalization": "strip and collapse 3+ newlines to two",
        "split_rule": (
            "source shard/row order; validation documents first until target; "
            "subsequent documents train; documents never cross splits"),
        "targets": {
            "validation_sequences": validation_target,
            "train_sequences": train_target,
        },
        "deduplication": {
            "method": (
                "SQLite UNIQUE normalized_text TEXT PRIMARY KEY COLLATE BINARY; "
                "exact normalized-text storage and comparison across both splits"),
            "database": deduplication_database,
        },
        "documents": documents,
        "skipped_empty_documents": skipped_empty,
        "skipped_duplicate_documents": skipped_duplicate,
        "document_index": document_index,
        "tokenizer": {
            "identity": "GPT-2; vocab_size=50257; BOS/EOS=50256",
            "path": "tokenizer",
            "files": tokenizer_files,
        },
        "splits": split_records,
        "elapsed_seconds": elapsed_seconds,
        "verification": "fixed identities plus paths, byte sizes, and counts only",
    }


def refuse_stale_outputs(output: Path) -> None:
    names = (
        "manifest.json", "documents.jsonl", "normalized_text.sqlite3",
        "validation.bin", "validation.bin.part", "train.bin", "train.bin.part",
        "tokenizer",
    )
    existing = [str(output / name) for name in names if (output / name).exists()]
    if existing:
        raise RuntimeError(
            "no-hash preparation requires a new output directory; existing: "
            + ", ".join(existing))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--hf-endpoint", default=DEFAULT_HF_ENDPOINT)
    parser.add_argument(
        "--train-sequences", type=int, default=DEFAULT_TRAIN_SEQUENCES)
    parser.add_argument(
        "--validation-sequences", type=int,
        default=DEFAULT_VALIDATION_SEQUENCES)
    args = parser.parse_args()
    if args.train_sequences <= 0 or args.validation_sequences <= 0:
        raise ValueError("split sequence targets must be positive")

    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    refuse_stale_outputs(output)
    source_dir = output / "source_shards"
    source_dir.mkdir(exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path, local_files_only=True)
    if (tokenizer.vocab_size != VOCAB_SIZE or
            tokenizer.eos_token_id != SPECIAL_TOKEN_ID or
            tokenizer.bos_token_id != SPECIAL_TOKEN_ID):
        raise RuntimeError("tokenizer must be the fixed GPT-2 tokenizer")
    tokenizer_dir = output / "tokenizer"
    tokenizer.save_pretrained(tokenizer_dir)

    writers = {
        "validation": PackedWriter(
            output / "validation.bin", args.validation_sequences),
        "train": PackedWriter(output / "train.bin", args.train_sequences),
    }
    split = "validation"
    exact_texts = ExactTextSet(output / "normalized_text.sqlite3")
    documents = {"validation": 0, "train": 0}
    skipped_empty = 0
    skipped_duplicate = 0
    source_files: list[dict[str, object]] = []
    started = time.monotonic()
    done = False
    document_index_path = output / "documents.jsonl"

    with document_index_path.open("w", encoding="utf-8") as document_index:
        for shard in range(SOURCE_SHARD_COUNT):
            filename = f"{shard:04d}.parquet"
            local_path = source_dir / filename
            remote_path = f"{SOURCE_CONFIG}/{SOURCE_SPLIT}/{filename}"
            url = source_shard_url(args.hf_endpoint, remote_path)
            if not local_path.exists():
                print(f"download {filename}", flush=True)
                subprocess.run([
                    "curl", "-fL", "--connect-timeout", "15",
                    "--max-time", "600", "--speed-limit", "1024",
                    "--speed-time", "60", "--retry", "1", "-o",
                    str(local_path) + ".part", url,
                ], check=True)
                os.replace(str(local_path) + ".part", local_path)

            rows_read = 0
            parquet = pq.ParquetFile(local_path)
            for batch in parquet.iter_batches(batch_size=256, columns=["text"]):
                texts = [normalize_text(text)
                         for text in batch.column(0).to_pylist()]
                encoded = tokenizer(
                    texts, add_special_tokens=False,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                    verbose=False)["input_ids"]
                for text, token_ids in zip(texts, encoded):
                    current_row = rows_read
                    rows_read += 1
                    if not token_ids:
                        skipped_empty += 1
                        continue
                    if not exact_texts.add(text):
                        skipped_duplicate += 1
                        continue
                    writers[split].add(token_ids)
                    documents[split] += 1
                    document_index.write(json.dumps({
                        "split": split,
                        "source_path": remote_path,
                        "row": current_row,
                    }, ensure_ascii=False) + "\n")
                    if writers[split].count == writers[split].target:
                        if split == "validation":
                            split = "train"
                        else:
                            done = True
                            break
                exact_texts.commit()
                print(json.dumps({
                    "train_sequences": writers["train"].count,
                    "validation_sequences": writers["validation"].count,
                    "documents": documents,
                    "seconds": round(time.monotonic() - started, 1),
                }), flush=True)
                if done:
                    break
            source_files.append({
                "path": remote_path,
                "local_path": str(local_path.relative_to(output)),
                "bytes": local_path.stat().st_size,
                "rows_in_file": parquet.metadata.num_rows,
                "rows_read": rows_read,
            })
            if done:
                break

    split_records = {
        name: writer.finish() for name, writer in writers.items()}
    deduplication_database = exact_texts.close()
    if deduplication_database["unique_normalized_texts"] != sum(
            documents.values()):
        raise RuntimeError("deduplication/document count mismatch")
    document_index = {
        "path": document_index_path.name,
        "bytes": document_index_path.stat().st_size,
        "records": sum(documents.values()),
    }
    manifest = build_manifest(
        source_files=source_files,
        split_records=split_records,
        documents=documents,
        skipped_empty=skipped_empty,
        skipped_duplicate=skipped_duplicate,
        document_index=document_index,
        deduplication_database=deduplication_database,
        tokenizer_files=tokenizer_inventory(tokenizer_dir),
        elapsed_seconds=time.monotonic() - started,
        validation_target=args.validation_sequences,
        train_target=args.train_sequences,
    )
    manifest_part = output / "manifest.json.part"
    manifest_part.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    os.replace(manifest_part, output / "manifest.json")
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
