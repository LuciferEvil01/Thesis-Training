#!/usr/bin/env bash
# Prepara el entorno y los datos con un solo comando:   ./preparar.sh
#
# 1. Crea el entorno de Python (.venv) con las versiones fijas de requirements.txt.
# 2. Descarga OULAD (una vez, ~47 MB) y construye el conjunto de datos de la sección 2.1.
#
# Se puede ejecutar todas las veces que haga falta: lo que ya está hecho no se repite.
set -euo pipefail
cd "$(dirname "$0")"

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

# --- Python -----------------------------------------------------------------------------------
PY="${PYTHON:-python3}"
if ! "$PY" -c 'import sys; sys.exit(sys.version_info < (3, 11))' 2>/dev/null; then
    echo "Hace falta Python 3.11 o posterior (prueba: PYTHON=python3.13 ./preparar.sh)." >&2
    exit 1
fi
if [ ! -x .venv/bin/python ]; then
    say "Creando el entorno de Python (.venv)"
    "$PY" -m venv .venv
fi
REQ_HASH="$(sha256sum requirements.txt | cut -d' ' -f1)"
if [ "$(cat .venv/.requirements 2>/dev/null)" != "$REQ_HASH" ]; then
    say "Instalando dependencias (solo la primera vez)"
    .venv/bin/pip install -q --upgrade pip
    .venv/bin/pip install -q -r requirements.txt
    .venv/bin/python -m ipykernel install --user --name proposal --display-name "Python (Proposal)" >/dev/null
    echo "$REQ_HASH" > .venv/.requirements
fi

# --- Datos ------------------------------------------------------------------------------------
say "Datos de OULAD"
.venv/bin/python scripts/download_oulad.py
if [ ! -f data/processed/dataset.parquet ]; then
    say "Construyendo el conjunto de datos (sección 2.1)"
    .venv/bin/python scripts/build_dataset.py
fi


say "Listo. Los notebooks están en notebooks/ (núcleo «Python (Proposal)»)."
