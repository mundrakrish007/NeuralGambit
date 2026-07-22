# BeginnerChessEngine — Compile & Run

Prerequisites:
- Place `chess.hpp` in the same folder as `chess_engine_simple.cpp` (already present).
- Install a C++17 compiler: either g++ (MinGW/MSYS2/WSL) or MSVC (Visual Studio Developer Tools).

Compile with g++ (MinGW / MSYS2 / WSL):

```bash
cd "C:\Users\mundr\OneDrive\Desktop\Neural\Neural Gambit"
g++ -std=c++17 -O2 -o engine chess_engine_simple.cpp
```

Compile with MSVC (Developer Command Prompt):

```bat
cd "C:\Users\mundr\OneDrive\Desktop\Neural\Neural Gambit"
cl /std:c++17 /O2 /EHsc /Fe:engine.exe chess_engine_simple.cpp
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
- If `g++` or `cl` is not found in your terminal, use:
  - Install MSYS2 and install `mingw-w64-x86_64-gcc`.
  - Or open "Developer Command Prompt for VS" before running `cl`.

If you want, I can:
- Try compiling inside WSL (if available).
- Create a Visual Studio project.
- Install/configure MSYS2 g++ and retry compilation here.
