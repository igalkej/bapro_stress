"""One-time seeder: loads data/docs/*.txt + data/stress_index.csv into PostgreSQL."""
import os
import sys
import csv
from pathlib import Path

# Allow imports from repo root when running as a script
sys.path.insert(0, "/workspace")

from sqlalchemy import text
from db.connection import get_engine

DOCS_DIR = Path("/workspace/data/docs")
STRESS_CSV = Path("/workspace/data/stress_index.csv")


def seed_documents(conn):
    doc_files = sorted(DOCS_DIR.glob("*.txt"))
    if not doc_files:
        print("No .txt files found in data/docs/. Aborting.")
        return 0

    inserted = 0
    for doc_path in doc_files:
        filename = doc_path.name
        # Parse date from filename: YYYY-MM-DD_<style>.txt
        date_str = filename.split("_")[0]
        # Determine doc_type from filename
        if "bloomberg" in filename:
            doc_type = "bloomberg_wrap"
        elif "bluefin" in filename:
            doc_type = "bluefin_report"
        elif "rap" in filename:
            doc_type = "rap_brief"
        else:
            doc_type = "other"

        content = doc_path.read_text(encoding="utf-8")

        result = conn.execute(
            text(
                """
                INSERT INTO documents (doc_date, filename, doc_type, content)
                VALUES (:doc_date, :filename, :doc_type, :content)
                ON CONFLICT (doc_date) DO NOTHING
                RETURNING id
                """
            ),
            {
                "doc_date": date_str,
                "filename": filename,
                "doc_type": doc_type,
                "content": content,
            },
        )
        if result.rowcount:
            inserted += 1

    return inserted


def seed_stress_index(conn):
    if not STRESS_CSV.exists():
        print(f"stress_index.csv not found at {STRESS_CSV}. Aborting.")
        return 0

    inserted = 0
    with STRESS_CSV.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            result = conn.execute(
                text(
                    """
                    INSERT INTO stress_index (index_date, stress_value)
                    VALUES (:index_date, :stress_value)
                    ON CONFLICT (index_date) DO NOTHING
                    RETURNING id
                    """
                ),
                {"index_date": row["date"], "stress_value": float(row["stress_value"])},
            )
            if result.rowcount:
                inserted += 1

    return inserted


def main():
    engine = get_engine()
    with engine.begin() as conn:
        docs_inserted = seed_documents(conn)
        stress_inserted = seed_stress_index(conn)

    print(f"Seeded {docs_inserted} documents and {stress_inserted} stress index rows.")

    # Quick verification
    with engine.connect() as conn:
        doc_count = conn.execute(text("SELECT COUNT(*) FROM documents")).scalar()
        stress_count = conn.execute(text("SELECT COUNT(*) FROM stress_index")).scalar()
    print(f"Total in DB → documents: {doc_count}, stress_index: {stress_count}")


if __name__ == "__main__":
    main()
