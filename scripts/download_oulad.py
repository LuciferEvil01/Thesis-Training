"""Descarga el Open University Learning Analytics Dataset (OULAD).

El zip se guarda en data/raw/oulad.zip y los CSV se extraen en data/raw/oulad/.
Solo usa la biblioteca estándar de Python.

Uso (desde la raíz de Proposal):
    python scripts/download_oulad.py           # descarga y extrae si falta
    python scripts/download_oulad.py --force   # vuelve a descargar
"""

import argparse
import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

# La copia de UCI es la fuente principal: el enlace oficial de la Open University ya no sirve el zip.
SOURCES = [
    "https://archive.ics.uci.edu/static/public/349/open+university+learning+analytics+dataset.zip",
    "http://schools.stem.open.ac.uk/cdn/files/anonymisedData.zip",
]
UCI_SHA256 = "f2ed1902616c1fe8d2824d872c0b7d2d72be435bf0124d077044fe4be2c6d3e4"
EXPECTED = ["assessments.csv", "courses.csv", "studentAssessment.csv", "studentInfo.csv",
            "studentRegistration.csv", "studentVle.csv", "vle.csv"]

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
ZIP = RAW / "oulad.zip"
OUT = RAW / "oulad"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str) -> None:
    tmp = ZIP.with_suffix(".part")
    with urllib.request.urlopen(url, timeout=60) as resp, tmp.open("wb") as f:
        total, done = int(resp.headers.get("Content-Length", 0)), 0
        while chunk := resp.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e6:6.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print()
    if not zipfile.is_zipfile(tmp):
        tmp.unlink()
        raise ValueError("el archivo descargado no es un zip válido")
    tmp.replace(ZIP)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="volver a descargar el zip")
    args = parser.parse_args()

    if not args.force and all((OUT / f).exists() for f in EXPECTED):
        print(f"OULAD ya está en {OUT.relative_to(ROOT)}/")
        return
    RAW.mkdir(parents=True, exist_ok=True)
    if args.force or not ZIP.exists():
        for url in SOURCES:
            print(f"Descargando {url}")
            try:
                download(url)
                break
            except Exception as e:  # se prueba la siguiente fuente
                print(f"  Falló: {e}")
        else:
            sys.exit("No se pudo descargar OULAD desde ninguna fuente.")

    print("SHA-256 verificado (copia de UCI)." if sha256(ZIP) == UCI_SHA256
          else "Aviso: el SHA-256 no coincide con la copia de UCI conocida.")
    with zipfile.ZipFile(ZIP) as z:
        z.extractall(OUT)
    for csv in OUT.rglob("*.csv"):  # algunas copias traen los CSV en una subcarpeta
        if csv.parent != OUT:
            csv.replace(OUT / csv.name)
    missing = [f for f in EXPECTED if not (OUT / f).exists()]
    if missing:
        sys.exit(f"Faltan archivos tras extraer: {missing}")
    print(f"CSV extraídos en {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
