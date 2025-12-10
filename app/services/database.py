# app/services/database.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, List
import csv
import sqlite3

from app.models.schemas import ProductLocation  # Pydantic schema


class ProductDatabase:
    """
    Simple SQLite-backed product database.

    - On first run, creates data/products.db (if not present)
    - Seeds from the CSV file (data/products.csv) if the table is empty
    - Provides `lookup()` and `all_products()` for the rest of the app
    """

    def __init__(self, csv_path: str | Path, db_path: str | Path | None = None):
        self.csv_path = Path(csv_path)
        self.db_path = Path(db_path) if db_path else self.csv_path.with_suffix(".db")

        # Ensure parent dir exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Connect to SQLite
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row

        self._ensure_schema_and_seed()

    def _ensure_schema_and_seed(self) -> None:
        cur = self._conn.cursor()

        # Create table if needed
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS products (
                product_name TEXT PRIMARY KEY,
                category     TEXT NOT NULL,
                aisle        TEXT NOT NULL,
                section      TEXT NOT NULL,
                price        REAL NOT NULL,
                in_stock     INTEGER NOT NULL
            )
            """
        )

        # If table is empty and CSV exists, seed from CSV
        cur.execute("SELECT COUNT(*) AS c FROM products")
        count = cur.fetchone()["c"]

        if count == 0 and self.csv_path.exists():
            with self.csv_path.open("r", newline="") as f:
                reader = csv.DictReader(f)
                rows = []
                for row in reader:
                    name = row["product_name"].strip().lower()
                    category = row["category"]
                    aisle = row["aisle"]
                    section = row["section"]
                    price = float(row["price"])
                    in_stock = 1 if row["stock"].strip().lower() in ("true", "1", "yes") else 0

                    rows.append((name, category, aisle, section, price, in_stock))

            if rows:
                cur.executemany(
                    """
                    INSERT OR REPLACE INTO products
                    (product_name, category, aisle, section, price, in_stock)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
                self._conn.commit()

        cur.close()

    def lookup(self, product_name: str) -> Optional[ProductLocation]:
        """Return a single product by name, or None if not found."""
        name = product_name.strip().lower()
        cur = self._conn.cursor()
        cur.execute(
            "SELECT * FROM products WHERE product_name = ?",
            (name,),
        )
        row = cur.fetchone()
        cur.close()

        if row is None:
            return None

        return ProductLocation(
            product_name=row["product_name"],
            category=row["category"],
            aisle=row["aisle"],
            section=row["section"],
            price=row["price"],
            in_stock=bool(row["in_stock"]),
        )

    def all_products(self) -> List[ProductLocation]:
        """Return all products in a stable order (by aisle then name)."""
        cur = self._conn.cursor()
        cur.execute("SELECT * FROM products ORDER BY aisle, product_name")
        rows = cur.fetchall()
        cur.close()

        return [
            ProductLocation(
                product_name=row["product_name"],
                category=row["category"],
                aisle=row["aisle"],
                section=row["section"],
                price=row["price"],
                in_stock=bool(row["in_stock"]),
            )
            for row in rows
        ]

    def __del__(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass


# --- Singleton-style accessor used by the API layer --- #

_db_instance: Optional[ProductDatabase] = None


def get_db() -> ProductDatabase:
    """
    Lazy-initialize and return a shared ProductDatabase instance.
    """
    global _db_instance
    if _db_instance is None:
        _db_instance = ProductDatabase(csv_path=Path("data/products.csv"))
    return _db_instance

