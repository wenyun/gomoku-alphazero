"""Web UI for playing against the trained engine.

    .venv/bin/python serve.py --checkpoint runs/az/latest.pt --port 8000

Then open http://localhost:8000 in a browser.

The server is deliberately boring: Python's standard-library HTTP server, one
JSON endpoint per action, and all the game state held in memory.  The board logic
is the same `Board` class the training run used, so there is no chance of the UI
and the engine disagreeing about the rules.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

from gomoku_zero.board import BLACK, WHITE, Board, action_to_coord
from gomoku_zero.player import AlphaZeroPlayer

WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")


class Game:
    """One human-vs-engine game, plus enough history to support undo."""

    def __init__(self, human_colour: int) -> None:
        self.human_colour = human_colour
        self.board = Board()
        self.history: list[Board] = []      # snapshots taken before each move
        self.moves: list[int] = []
        self.last_ai = None

    def play(self, action: int) -> None:
        self.history.append(self.board.copy())
        self.board.play(action)
        self.moves.append(action)

    def undo(self, count: int) -> None:
        for _ in range(count):
            if not self.history:
                return
            self.board = self.history.pop()
            self.moves.pop()
        self.last_ai = None

    def to_json(self) -> dict:
        board = self.board
        return {
            "stones": board.stones.reshape(-1).astype(int).tolist(),
            "to_play": int(board.to_play),
            "human_colour": self.human_colour,
            "last_move": int(board.last_move),
            "move_count": int(board.move_count),
            "is_over": bool(board.is_over),
            "winner": int(board.winner),
            "moves": [int(m) for m in self.moves],
            "legal": board.candidate_mask().astype(int).tolist(),
            "ai": self.last_ai,
        }


class Engine:
    """Thread-safe wrapper around the search, and owner of the current game."""

    def __init__(self, checkpoint: str, device: str, simulations: int) -> None:
        self.lock = threading.Lock()
        self.player = AlphaZeroPlayer(checkpoint, device=device,
                                      simulations=simulations, batch=1,
                                      temperature=0.0)
        self.simulations = simulations
        self.game = Game(BLACK)

    def new_game(self, human_colour: int, simulations: int) -> dict:
        with self.lock:
            self.player.simulations = simulations
            self.game = Game(human_colour)
            if human_colour == WHITE:
                self._engine_move()
            return self.game.to_json()

    def human_move(self, action: int) -> dict:
        with self.lock:
            game = self.game
            if game.board.is_over:
                return game.to_json()
            if game.board.stones.reshape(-1)[action] != 0:
                return game.to_json()          # occupied: ignore
            game.play(action)
            game.last_ai = None
            if not game.board.is_over:
                self._engine_move()
            return game.to_json()

    def undo(self) -> dict:
        with self.lock:
            # Step back to the human's turn: two plies normally, one if the
            # engine has just moved and it is already the human's turn.
            self.game.undo(2 if len(self.game.moves) >= 2 else len(self.game.moves))
            return self.game.to_json()

    def hint(self) -> dict:
        with self.lock:
            if self.game.board.is_over:
                return self.game.to_json()
            info = self._analyse()
            info["is_hint"] = True
            self.game.last_ai = info
            return self.game.to_json()

    # ------------------------------------------------------------------ private
    def _analyse(self) -> dict:
        start = time.perf_counter()
        result = self.player.analyse(self.game.board)
        return {
            "move": result["move"],
            "coord": action_to_coord(result["move"]),
            # The search always evaluates from the point of view of the side to
            # move.  That is the engine when it is choosing its own move, but the
            # *human* when this is a hint, so say which and let the UI label it.
            "for_player": int(self.game.board.to_play),
            "win_probability": result["win_probability"],
            "value": result["value"],
            "top": result["top"],
            "simulations": self.player.simulations,
            "seconds": time.perf_counter() - start,
        }

    def _engine_move(self) -> None:
        info = self._analyse()
        self.game.play(info["move"])
        self.game.last_ai = info


class Handler(SimpleHTTPRequestHandler):
    engine: Engine = None                       # set in main()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WEB_DIR, **kwargs)

    def log_message(self, fmt, *args):
        pass                                    # keep the console quiet

    def do_POST(self):
        try:
            state = self._dispatch()
        except KeyError:
            self.send_error(404)
            return
        except Exception as exc:                # noqa: BLE001 - report, don't die
            # Without this the handler thread would die silently and the browser
            # would just see a dropped connection with no explanation.
            import traceback
            traceback.print_exc()
            self.send_error(500, f"{type(exc).__name__}: {exc}")
            return

        body = json.dumps(state).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _dispatch(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        route = self.path.rstrip("/")

        if route == "/api/new":
            colour = BLACK if payload.get("human", "black") == "black" else WHITE
            return self.engine.new_game(colour, int(payload.get("sims", 800)))
        if route == "/api/move":
            return self.engine.human_move(int(payload["move"]))
        if route == "/api/undo":
            return self.engine.undo()
        if route == "/api/hint":
            return self.engine.hint()
        raise KeyError(route)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="runs/az/latest.pt")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--sims", type=int, default=800)
    args = ap.parse_args()

    Handler.engine = Engine(args.checkpoint, args.device, args.sims)
    print(f"loaded {Handler.engine.player.name}")
    print(f"open http://localhost:{args.port}  (Ctrl-C to stop)", flush=True)
    ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
