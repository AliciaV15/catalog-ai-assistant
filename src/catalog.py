"""Load and clean the catalog data (Google Sheet or local CSV)"""

from __future__ import annotations

import re
import unicodedata
import pandas as pd

REQUIRED = ["nombre", "categoria", "precio", "stock"]
OPTIONAL = ["id", "marca", "specs"]

# alternative names for columns in the catalog, to be mapped to the standard names
COLUMN_ALIASES = {
    "producto": "nombre", "name": "nombre",
    "category": "categoria",
    "price": "precio",
    "cantidad": "stock", "existencias": "stock", "inventario": "stock",
    "caracteristicas": "specs", "especificaciones": "specs", "descripcion": "specs",
    "brand": "marca",
    "codigo": "id", "sku": "id",
}

def _norm(s: str) -> str:
    """'Categoría ' -> 'categoria'"""
    s = unicodedata.normalize("NFD", str(s).strip().lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def sheet_to_csv_url(source) -> str:
    """ Converts a Google Sheet URL to a CSV export URL. If the source is already a CSV or local path, returns it unchanged. """
    source = str(source)
    m = re.search(r"docs\.google\.com/spreadsheets/d/([\w-]+)", source)
    if not m:
        return source  # already a CSV URL or local path
    url = f"https://docs.google.com/spreadsheets/d/{m.group(1)}/export?format=csv"
    gid = re.search(r"[#&?]gid=(\d+)", source)
    return url + (f"&gid={gid.group(1)}" if gid else "")


def parse_number(value) -> float | None:
    """'Bs 4.500' -> 4500.0 | '4,500.50' -> 4500.5 | '45.5' -> 45.5"""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = re.sub(r"[^\d.,]", "", str(value))
    if not s:
        return None
    if "," in s and "." in s:
        dec = "," if s.rfind(",") > s.rfind(".") else "."
        thousands = "." if dec == "," else ","
        s = s.replace(thousands, "").replace(dec, ".")
    elif "," in s or "." in s:
        sep = "," if "," in s else "."
        parts = s.split(sep)
        if len(parts) > 2 or len(parts[-1]) == 3:  # es separador de miles
            s = s.replace(sep, "")
        else:
            s = s.replace(sep, ".")
    return float(s)


def format_precio(p: float) -> str:
    """4500 -> '4.500'"""
    p = float(p)
    if p == int(p):
        return f"{int(p):,}".replace(",", ".")
    return f"{p:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def load_catalog(source) -> pd.DataFrame:
    """Loads the catalog and returns it cleaned.

    source: Google Sheets link, CSV URL or local file path.
    Output columns: id, nombre, categoria, marca, precio (float), stock (int), specs.
    """
    df = pd.read_csv(sheet_to_csv_url(source), dtype=str, keep_default_na=False)

    df.columns = [COLUMN_ALIASES.get(_norm(c), _norm(c)) for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")
    for col in OPTIONAL:
        if col not in df.columns:
            df[col] = ""

    for col in ["nombre", "categoria", "marca", "specs", "id"]:
        df[col] = df[col].astype(str).str.strip()

    df["precio"] = df["precio"].map(parse_number)
    df["stock"] = df["stock"].map(parse_number).fillna(0).astype(int).clip(lower=0)

    invalid = df["nombre"].eq("") | df["precio"].isna()
    if invalid.any():
        print(f"⚠️  Se descartaron {int(invalid.sum())} filas sin nombre o sin precio válido.")
    df = df[~invalid].copy()

    # ids únicos (si el Sheet no trae)
    empty_id = df["id"].eq("")
    df.loc[empty_id, "id"] = [f"P-{i:04d}" for i in range(1, int(empty_id.sum()) + 1)]
    df = df.drop_duplicates(subset="id", keep="first")

    return df[["id", "nombre", "categoria", "marca", "precio", "stock", "specs"]].reset_index(drop=True)


def to_documents(df: pd.DataFrame) -> list[dict]:
    """Converts the catalog into documents for the RAG index.

    - text: what is converted into an embedding (only stable data).
    - metadata: for identifying the product. Price and stock are NOT included in the text:
      they are read from the fresh catalog, using the id, at the time of responding.
    """
    docs = []
    for r in df.to_dict("records"):
        text = f"{r['nombre']}. Categoría: {r['categoria']}. Marca: {r['marca']}. {r['specs']}".strip()
        docs.append({
            "id": r["id"],
            "text": text,
            "metadata": {"nombre": r["nombre"], "categoria": r["categoria"], "marca": r["marca"]},
        })
    return docs