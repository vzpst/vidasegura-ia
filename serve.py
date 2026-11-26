"""Servidor local do VidaSegura IA sem cache.

O `python -m http.server` deixa o navegador guardar versões antigas dos módulos
JavaScript; depois de um `git pull` a página pode continuar rodando o código
velho (ex.: o cartão de quedas preso em "NÃO AVALIADO"). Este servidor manda o
navegador sempre conferir os arquivos.

Uso (na pasta do projeto):
    python serve.py            -> http://localhost:8765
    python serve.py 9000       -> outra porta
"""

from __future__ import annotations

import functools
import http.server
import sys
from pathlib import Path


class NoCacheHandler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map,
                      ".js": "text/javascript", ".mjs": "text/javascript", ".json": "application/json"}

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        super().end_headers()


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    handler = functools.partial(NoCacheHandler, directory=str(Path(__file__).resolve().parent))
    with http.server.ThreadingHTTPServer(("127.0.0.1", port), handler) as server:
        print(f"VidaSegura IA em http://localhost:{port}  (Ctrl+C para parar)")
        server.serve_forever()


if __name__ == "__main__":
    main()
