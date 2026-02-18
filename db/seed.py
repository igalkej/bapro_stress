"""One-time seeder: loads data/docs/*.txt + data/stress_index.csv into the DB."""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text
from db.connection import get_engine
from config import DOCS_DIR, STRESS_CSV


def seed_documents(conn):
    doc_files = sorted(DOCS_DIR.glob("*.txt"))
    if not doc_files:
        print(f"No .txt files found in {DOCS_DIR}. Aborting.")
        return 0

    inserted = 0
    for doc_path in doc_files:
        filename = doc_path.name
        date_str = filename.split("_")[0]
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
                INSERT OR IGNORE INTO documents (doc_date, filename, doc_type, content)
                VALUES (:doc_date, :filename, :doc_type, :content)
                """
            ),
            {"doc_date": date_str, "filename": filename, "doc_type": doc_type, "content": content},
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
                    INSERT OR IGNORE INTO stress_index (index_date, stress_value)
                    VALUES (:index_date, :stress_value)
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

    with engine.connect() as conn:
        doc_count = conn.execute(text("SELECT COUNT(*) FROM documents")).scalar()
        stress_count = conn.execute(text("SELECT COUNT(*) FROM stress_index")).scalar()
    print(f"Total in DB: documents={doc_count}, stress_index={stress_count}")


if __name__ == "__main__":
    main()
