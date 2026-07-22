# About ChessEngine 

Prerequisites:
- Place `chess.hpp` in the same folder as `chess_engine_simple.cpp` (already present).

Compile with g++ (MinGW / MSYS2 / WSL):

```bash
cd "C:\Users\mundr\OneDrive\Desktop\Neural\Neural Gambit"
g++ -std=c++17 -O2 -o engine chess_engine_simple.cpp
```

Run:

```bash
# on Unix/WSL or msys
./engine
# on Windows cmd / PowerShell
engine.exe
```

Quick runtime tips:
- The program uses UCI. Type `uci` then `isready` to test.
- To run a search: `position startpos` then `go depth 5`.
- To load NNUE weights (if you have them): place `nnue.bin` next to the executable or use `loadnnue <path>` or `setoption name EvalFile value <path>`.

I have used stockfish for testing my engine against it.
I would continue to develop this engine and keep on training it to increase its strength.



