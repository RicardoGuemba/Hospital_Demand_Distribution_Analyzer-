#!/bin/bash
# Launcher macOS — duplo clique no Finder para abrir o app
cd "$(dirname "$0")"

PYTHON=""
for cmd in python3 python; do
    if command -v "$cmd" >/dev/null 2>&1; then
        PYTHON="$cmd"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    echo "Erro: Python 3.11+ não encontrado."
    echo "Instale em https://www.python.org/downloads/ ou via: brew install python@3.11"
    read -r -p "Pressione Enter para fechar..."
    exit 1
fi

VENV=".venv/bin"
if [ ! -x "$VENV/python" ]; then
    echo "Criando ambiente virtual..."
    "$PYTHON" -m venv .venv
    "$VENV/pip" install -r requirements.txt
    "$VENV/pip" install -e .
fi

echo "Iniciando Hospital Demand Analyzer..."
"$VENV/python" -m hospital_demand_analyzer
EXIT_CODE=$?

if [ "$EXIT_CODE" -ne 0 ]; then
    echo ""
    echo "Erro ao iniciar (código $EXIT_CODE). Veja logs/app.log"
    read -r -p "Pressione Enter para fechar..."
fi

exit "$EXIT_CODE"
