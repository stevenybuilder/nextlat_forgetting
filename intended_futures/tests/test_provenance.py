from pathlib import Path

from intended_futures.provenance import sha256_source_tree


def test_source_tree_hash_ignores_mutable_output_directories(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "code.py").write_text("x = 1\n", encoding="utf-8")
    first = sha256_source_tree(tmp_path)
    for name in ("raw", "results", "runs", ".runtime"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "output.txt").write_text("mutable\n", encoding="utf-8")
    assert sha256_source_tree(tmp_path) == first
    (tmp_path / "src" / "code.py").write_text("x = 2\n", encoding="utf-8")
    assert sha256_source_tree(tmp_path) != first
