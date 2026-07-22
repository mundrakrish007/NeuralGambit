/*
╔══════════════════════════════════════════════════════════════════════╗
║         CHESS ENGINE  —  Initial Version                             ║
║         Uses: Disservin's chess-library  (chess.hpp)                 ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  WHAT THIS FILE CONTAINS:                                            ║
║                                                                      ║
║   Section 1  — Includes (bring in chess.hpp and standard C++ tools)  ║
║   Section 2  — Constants (scores, depth limits, piece values)        ║
║   Section 3  — Piece-Square Tables (positional bonuses per piece)    ║
║   Section 4  — Classical evaluation (the original hand-written one)  ║
║   Section 4B — NNUE evaluation (small neural net, weights optional)  ║
║   Section 4C — Evaluation dispatch (evaluate() picks NNUE or not)    ║
║   Section 5  — Move ordering (try promising moves first)             ║
║   Section 6  — Time management (stop when the clock runs out)        ║
║   Section 7  — Quiescence search (don't stop mid-capture)            ║
║   Section 8  — Alpha-Beta search (the engine's thinking algorithm)   ║
║   Section 9  — Iterative deepening (search deeper and deeper)        ║
║   Section 9B — Training data export (for training the NNUE later)    ║
║   Section 10 — Perft test (verify move generation is correct)        ║
║   Section 11 — Board display (pretty-print the board)                ║
║   Section 12 — UCI loop (talk to chess GUIs)                         ║
║   Section 13 — main()                                                ║
║                                                                      ║
║  WHAT THE LIBRARY (chess.hpp) DOES SO WE DON'T HAVE TO:              ║
║    • All chess rules (piece movement, castling, en passant, etc.)    ║
║    • Generating every legal move in any position                     ║
║    • Making and undoing moves on the board                           ║
║    • Detecting check, checkmate, stalemate, draws                    ║
║    • Parsing FEN strings (text descriptions of chess positions)      ║
║                                                                      ║
║  WHAT WE WRITE OURSELVES (the engine's brain):                       ║
║    • evaluateClassic() — hand-written score (material + PSTs)        ║
║    • evaluateNNUE()    — neural-network score (needs a weights file) ║
║    • evaluate()        — picks whichever of the two above to use     ║
║    • alphaBeta()       — look ahead through moves to find the best   ║
║    • quiescence()      — don't stop searching mid-capture            ║
║    • UCI loop          — communicate with chess GUIs                 ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
*/


// ════════════════════════════════════════════════════════════════
// SECTION 1 — INCLUDES
//
// #include brings in pre-written code so we can use it here.
// ════════════════════════════════════════════════════════════════

#include "chess.hpp"   // The chess library — handles all board and move logic

#include <iostream>    // For printing text (std::cout) and reading input (std::cin)
#include <string>      // For std::string (text storage)
#include <sstream>     // For std::istringstream (splitting a text line into words)
#include <algorithm>   // For std::sort (sorting a list of moves)
#include <chrono>      // For measuring time in milliseconds
#include <fstream>     // NEW: for reading/writing files (NNUE weights, training data)
#include <vector>      // NEW: for std::vector (resizable arrays for NNUE weights)
#include <random>      // NEW: for std::mt19937 (random self-play games for training data)

// "using namespace chess" lets us write "Board" instead of "chess::Board",
// "Move" instead of "chess::Move", etc. Just saves typing.
using namespace chess;


// ════════════════════════════════════════════════════════════════
// SECTION 2 — CONSTANTS
//
// These are fixed values used throughout the engine.
// Defining them here in one place makes them easy to understand
// and easy to change if you want to experiment.
// ════════════════════════════════════════════════════════════════

// All scores in this engine are measured in "centipawns" (cp).
//   100 cp  = 1 pawn's worth of advantage
//   Positive score = good for whoever is currently to move
//   Negative score = bad for whoever is currently to move

const int INF        = 1000000;  // "Infinity" — bigger than any real chess score.
                                  // Used as a starting "worst case" value.

const int MATE_SCORE = 900000;   // Score we assign to checkmate.
                                  // (Slightly less than INF so math doesn't overflow)

const int DRAW_SCORE = 0;        // A draw is worth nothing for either side.

const int MAX_DEPTH  = 64;       // We never search more than 64 moves ahead.
const int MAX_PLY    = 128;      // Maximum half-moves in any single search path.

// How much each chess piece is worth, in centipawns.
// The chess library numbers piece types like this:
//   PAWN=0, KNIGHT=1, BISHOP=2, ROOK=3, QUEEN=4, KING=5
// So MATERIAL[0] = 100 means a pawn is worth 100 cp, and so on.
const int MATERIAL[6] = {
    100,    // Pawn   (worth 1 pawn — obviously!)
    320,    // Knight (worth about 3.2 pawns)
    330,    // Bishop (slightly better than a knight in open positions)
    500,    // Rook   (worth about 5 pawns)
    900,    // Queen  (the strongest piece, worth about 9 pawns)
    20000   // King   (huge value — losing it ends the game!)
};

// A simple helper function: give it a PieceType, get back its material value.
// static_cast<int>(pt) just converts the PieceType to a plain number (0-5).
int materialOf(PieceType pt) {
    return MATERIAL[static_cast<int>(pt)];
}


// ════════════════════════════════════════════════════════════════
// SECTION 3 — PIECE-SQUARE TABLES (PSTs)
//
// A chess piece is not equally good on every square. For example:
//   • A knight in the centre (e4) controls 8 squares.
//   • A knight in the corner (a1) controls only 2 squares.
//
// These tables give a BONUS (positive number) for good squares
// and a PENALTY (negative number) for bad squares.
//
// The engine adds this bonus on top of the piece's material value
// to get a total score for each piece on the board.
//
// TABLE LAYOUT:
//   Index 0 = a1 (White's bottom-left corner)
//   Index 7 = h1, Index 56 = a8, Index 63 = h8
//   Index formula: rank * 8 + file  (rank 0 = rank-1, rank 7 = rank-8)
//
// All tables are written from WHITE's point of view.
// For BLACK pieces, we flip the table vertically:
//   mirroredIndex = originalIndex XOR 56
//   (XOR 56 flips the rank while keeping the file the same)
//
// Example: White knight on e4 → index 28 → PST_KNIGHT[28] = +20 (bonus!)
//          Black knight on e5 → mirror → index 28 → same value for fairness
//
// NOTE: These tables are only used by evaluateClassic() (Section 4).
// The NNUE evaluator (Section 4B) learns its own positional knowledge
// directly from training data, so it doesn't need these tables at all.
// ════════════════════════════════════════════════════════════════

// PAWN TABLE
// Pawns want to advance toward the opponent's back rank (promotion!).
// Central pawns (d and e files) on ranks 4-6 get big bonuses.
const int PST_PAWN[64] = {
     0,  0,  0,  0,  0,  0,  0,  0,   // rank 1 — pawns can't be here (they start on rank 2)
     5, 10, 10,-20,-20, 10, 10,  5,   // rank 2 — starting rank (small penalty on d2/e2)
     5, -5,-10,  0,  0,-10, -5,  5,   // rank 3
     0,  0,  0, 20, 20,  0,  0,  0,   // rank 4 — big bonus for e4 and d4!
     5,  5, 10, 25, 25, 10,  5,  5,   // rank 5 — even better
    10, 10, 20, 30, 30, 20, 10, 10,   // rank 6 — getting close to promotion
    50, 50, 50, 50, 50, 50, 50, 50,   // rank 7 — one step away from becoming a queen!
     0,  0,  0,  0,  0,  0,  0,  0    // rank 8 — promotion rank (pawn becomes another piece)
};

// KNIGHT TABLE
// "A knight on the rim is dim" — corners and edges are terrible for knights.
// The centre (d4, e4, d5, e5) gives the knight maximum reach.
const int PST_KNIGHT[64] = {
    -50,-40,-30,-30,-30,-30,-40,-50,   // rank 1 — awful in corners
    -40,-20,  0,  5,  5,  0,-20,-40,   // rank 2
    -30,  5, 10, 15, 15, 10,  5,-30,   // rank 3
    -30,  0, 15, 20, 20, 15,  0,-30,   // rank 4 — centre is best (+20)
    -30,  5, 15, 20, 20, 15,  5,-30,   // rank 5
    -30,  0, 10, 15, 15, 10,  0,-30,   // rank 6
    -40,-20,  0,  0,  0,  0,-20,-40,   // rank 7
    -50,-40,-30,-30,-30,-30,-40,-50    // rank 8 — awful in corners
};

// BISHOP TABLE
// Bishops work best when they have open diagonals.
// They hate being blocked by their own pawns.
const int PST_BISHOP[64] = {
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -20,-10,-10,-10,-10,-10,-10,-20
};

// ROOK TABLE
// Rooks are powerful on open files (no pawns blocking them).
// The 7th rank is special — a rook there attacks the opponent's pawns.
const int PST_ROOK[64] = {
     0,  0,  0,  5,  5,  0,  0,  0,   // rank 1
    -5,  0,  0,  0,  0,  0,  0, -5,   // rank 2
    -5,  0,  0,  0,  0,  0,  0, -5,   // rank 3
    -5,  0,  0,  0,  0,  0,  0, -5,   // rank 4
    -5,  0,  0,  0,  0,  0,  0, -5,   // rank 5
    -5,  0,  0,  0,  0,  0,  0, -5,   // rank 6
     5, 10, 10, 10, 10, 10, 10,  5,   // rank 7 — rook on the 7th is powerful! (+5 to +10)
     0,  0,  0,  0,  0,  0,  0,  0    // rank 8
};

// QUEEN TABLE
// The queen should avoid coming out too early (it can be chased away).
// It likes active central positions once development is complete.
const int PST_QUEEN[64] = {
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -10,  5,  5,  5,  5,  5,  0,-10,
      0,  0,  5,  5,  5,  5,  0, -5,
     -5,  0,  5,  5,  5,  5,  0, -5,
    -10,  0,  5,  5,  5,  5,  0,-10,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20
};

// KING — MIDDLEGAME TABLE
// In the middlegame, the king is vulnerable. It should castle and
// hide behind its pawns (g1 or c1 for White are safe spots).
// Moving the king to the centre in the middlegame is very dangerous!
const int PST_KING_MID[64] = {
     20, 30, 10,  0,  0, 10, 30, 20,   // rank 1 — g1 (castled kingside) = +30, c1 = +10
     20, 20,  0,  0,  0,  0, 20, 20,   // rank 2 — one step up is okay
    -10,-20,-20,-20,-20,-20,-20,-10,   // rank 3 — venturing forward is dangerous
    -20,-30,-30,-40,-40,-30,-30,-20,   // rank 4 — very bad
    -30,-40,-40,-50,-50,-40,-40,-30,   // rank 5 — terrible
    -30,-40,-40,-50,-50,-40,-40,-30,   // rank 6
    -30,-40,-40,-50,-50,-40,-40,-30,   // rank 7
    -30,-40,-40,-50,-50,-40,-40,-30    // rank 8 — worst (open to all attacks)
};

// KING — ENDGAME TABLE
// In the endgame, most pieces are gone so the king can't be mated easily.
// Now the king should MARCH TO THE CENTRE to help push pawns to promotion!
const int PST_KING_END[64] = {
    -50,-30,-30,-30,-30,-30,-30,-50,
    -30,-30,  0,  0,  0,  0,-30,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,   // Centre is best in endgame (+40!)
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-20,-10,  0,  0,-10,-20,-30,
    -50,-40,-30,-20,-20,-30,-40,-50
};

// PST LOOKUP FUNCTION
// Given a piece type, its color, and its square, return the PST bonus.
// For Black pieces: XOR 56 mirrors the index vertically so Black's rank 1
// maps to White's rank 8 (since they start from opposite ends of the board).
int pstScore(PieceType pt, Color color, Square sq) {
    // For White: use the square index directly.
    // For Black: XOR with 56 flips the rank (e.g. rank 0 ↔ rank 7).
    int index = (color == Color::WHITE) ? sq.index() : (sq.index() ^ 56);

    switch (static_cast<int>(pt)) {
        case 0:  return PST_PAWN[index];
        case 1:  return PST_KNIGHT[index];
        case 2:  return PST_BISHOP[index];
        case 3:  return PST_ROOK[index];
        case 4:  return PST_QUEEN[index];
        case 5:  return PST_KING_MID[index];  // endgame table selected in evaluateClassic()
        default: return 0;
    }
}


// ════════════════════════════════════════════════════════════════
// SECTION 4 — CLASSICAL EVALUATION
//
// evaluateClassic(board) answers the question: "Who is winning, and
// by how much?" using hand-written chess knowledge (material values
// and piece-square tables). This is the ORIGINAL evaluator.
//
// It returns a score in centipawns from the perspective of whoever
// is currently to move:
//   Positive → the side to move is winning
//   Negative → the side to move is losing
//   Near 0   → roughly equal position
//
// This function now serves TWO purposes:
//   1. It's the fallback evaluator used whenever no NNUE weights
//      file has been loaded (see Section 4B/4C).
//   2. It's the "teacher" used to LABEL training positions when you
//      export training data for the NNUE (see Section 9B) — i.e. we
//      bootstrap the neural network by first teaching it to imitate
//      this hand-written evaluator, which is a common, simple way
//      to get an NNUE project started.
//
// HOW IT WORKS:
//   We go through every square. For each piece we find, we add:
//     1. Its material value  (how much the piece is worth)
//     2. Its PST bonus       (how good its current square is)
//   White pieces ADD to the score. Black pieces SUBTRACT.
//
//   At the end, if it's Black's turn, we flip the sign.
//   This is because our search always tries to MAXIMISE the score
//   for "whoever is to move," so the score must be from their perspective.
// ════════════════════════════════════════════════════════════════

// Count non-pawn, non-king pieces to decide if we're in the endgame.
// Returns a "phase" number from 0 (pure endgame) to 24 (full middlegame).
// In the endgame, the king should centralise (use a different PST).
int gamePhase(const Board& board) {
    int phase = 0;
    for (Color c : {Color::WHITE, Color::BLACK}) {
        // board.pieces(pieceType, color) returns a bitboard (set of squares).
        // .count() tells us how many of that piece remain on the board.
        phase += 1 * board.pieces(PieceType::KNIGHT, c).count();
        phase += 1 * board.pieces(PieceType::BISHOP, c).count();
        phase += 2 * board.pieces(PieceType::ROOK,   c).count();
        phase += 4 * board.pieces(PieceType::QUEEN,  c).count();
    }
    if (phase > 24) phase = 24;
    return phase;
}

// Build a bitboard with all squares on a given file.
// fileIndex: 0=a-file, 1=b-file, ..., 7=h-file
Bitboard fileBitboard(int fileIndex) {
    Bitboard bb;
    for (int rank = 0; rank < 8; rank++) {
        bb.set(rank * 8 + fileIndex);
    }
    return bb;
}

// THE CLASSICAL (hand-written) EVALUATION FUNCTION
int evaluateClassic(const Board& board) {
    int  score   = 0;              // We add and subtract pieces to build this
    int  phase   = gamePhase(board);
    bool endgame = (phase <= 8);   // Use endgame king table when few pieces remain

    // ── Step 1: Count material + positional bonuses ──────────────
    //
    // board.occ() gives us a Bitboard of ALL occupied squares.
    // A Bitboard is just a 64-bit number where each bit = one square.
    // We "pop" squares out one at a time using .pop(), which removes
    // and returns the index of the lowest set bit.
    Bitboard occupied = board.occ();
    while (occupied) {
        Square sq(occupied.pop());          // Get next occupied square
        Piece  p  = board.at<Piece>(sq);    // What piece is on that square?
        Color  c  = p.color();              // Is it White or Black?
        PieceType t = p.type();             // What type? (pawn, knight, etc.)
        int sign  = (c == Color::WHITE) ? +1 : -1;  // +1 for White, -1 for Black

        // Add the piece's material value
        score += sign * materialOf(t);

        // Add the piece's positional bonus from its PST
        // (Kings use a different table in the endgame)
        if (t == PieceType::KING && endgame) {
            int idx = (c == Color::WHITE) ? sq.index() : (sq.index() ^ 56);
            score += sign * PST_KING_END[idx];
        } else {
            score += sign * pstScore(t, c, sq);
        }
    }

    // ── Step 2: Bishop pair bonus ────────────────────────────────
    // Two bishops working together cover both diagonal colours
    // and are worth about 30 cp more than their individual values.
    if (board.pieces(PieceType::BISHOP, Color::WHITE).count() >= 2) score += 30;
    if (board.pieces(PieceType::BISHOP, Color::BLACK).count() >= 2) score -= 30;

    // ── Step 3: Rook on open file bonus ─────────────────────────
    // A rook on a file with NO pawns can move freely up and down.
    //   Open file    (no pawns at all)    → +20 cp bonus
    //   Semi-open    (no own pawns, but   → +10 cp bonus
    //                opponent has pawns)
    for (Color c : {Color::WHITE, Color::BLACK}) {
        int sign = (c == Color::WHITE) ? +1 : -1;
        Bitboard rooks = board.pieces(PieceType::ROOK, c);
        while (rooks) {
            Square rSq(rooks.pop());
            int    fileIdx  = static_cast<int>(rSq.file());
            Bitboard theFile = fileBitboard(fileIdx);

            bool openFile = (board.pieces(PieceType::PAWN) & theFile).empty();
            bool semiOpen = (board.pieces(PieceType::PAWN, c) & theFile).empty()
                         && !(board.pieces(PieceType::PAWN, ~c) & theFile).empty();

            if (openFile)      score += sign * 20;
            else if (semiOpen) score += sign * 10;
        }
    }

    // ── Step 4: Return from the current player's perspective ────
    // All math above was "positive = White winning."
    // The search always maximises for whoever is to move, so:
    //   If White to move → return as-is (positive = good for White)
    //   If Black to move → negate     (positive = good for Black)
    return (board.sideToMove() == Color::WHITE) ? score : -score;
}


// ════════════════════════════════════════════════════════════════
// SECTION 4B — NNUE EVALUATION
//
// This is a small neural network that can REPLACE evaluateClassic()
// once you've trained it. We deliberately keep the design as simple
// as possible — this is NOT the same fancy incremental-accumulator
// architecture that Stockfish uses, just a plain little network you
// can understand end-to-end and train yourself.
//
// ════════════════════════════════════════════════════════════════

const int NNUE_INPUT_SIZE   = 768;   // 12 planes x 64 squares
const int NNUE_HIDDEN1_SIZE = 256;
const int NNUE_HIDDEN2_SIZE = 32;
const int NNUE_OUTPUT_SIZE  = 1;

// This struct just holds the network's weights and biases once loaded.
struct NNUENetwork {
    std::vector<float> W1;   // size 768 * 256
    std::vector<float> B1;   // size 256
    std::vector<float> W2;   // size 256 * 32
    std::vector<float> B2;   // size 32
    std::vector<float> W3;   // size 32 * 1
    std::vector<float> B3;   // size 1
    bool loaded = false;     // becomes true once a valid file has been read
};

NNUENetwork nnue;          // the one network the engine keeps in memory
bool useNNUE = true;       // if true AND nnue.loaded, evaluate() uses NNUE

// Read one block of floats from a binary file into a vector.
// Returns false if the file ran out of data early (a corrupt/short file).
bool readFloatBlock(std::ifstream& file, std::vector<float>& dest, size_t count) {
    dest.resize(count);
    file.read(reinterpret_cast<char*>(dest.data()),
               static_cast<std::streamsize>(count * sizeof(float)));
    return static_cast<bool>(file);
}

// Load NNUE weights from a raw binary file (see the format documented
// above). Returns true on success. On failure, the network is left
// exactly as it was before (still usable if it was already loaded).
bool loadNNUEFile(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cout << "info string Could not open NNUE file: " << path << "\n";
        return false;
    }

    NNUENetwork candidate;   // load into a temporary network first,
                             // so a bad file never corrupts a working one
    bool ok = true;
    ok = ok && readFloatBlock(file, candidate.W1, NNUE_INPUT_SIZE   * NNUE_HIDDEN1_SIZE);
    ok = ok && readFloatBlock(file, candidate.B1, NNUE_HIDDEN1_SIZE);
    ok = ok && readFloatBlock(file, candidate.W2, NNUE_HIDDEN1_SIZE * NNUE_HIDDEN2_SIZE);
    ok = ok && readFloatBlock(file, candidate.B2, NNUE_HIDDEN2_SIZE);
    ok = ok && readFloatBlock(file, candidate.W3, NNUE_HIDDEN2_SIZE * NNUE_OUTPUT_SIZE);
    ok = ok && readFloatBlock(file, candidate.B3, NNUE_OUTPUT_SIZE);

    if (!ok) {
        std::cout << "info string NNUE file is too short or malformed: " << path << "\n";
        return false;
    }

    candidate.loaded = true;
    nnue = candidate;   // swap the working network for the newly loaded one
    std::cout << "info string Loaded NNUE weights from " << path << "\n";
    return true;
}

// Work out the list of "active" input features (the ones equal to 1)
// for the current position, always described from the point of view
// of whoever is about to move (see the big comment above).
void extractActiveFeatures(const Board& board, std::vector<int>& active) {
    active.clear();

    Color us = board.sideToMove();

    Bitboard occupied = board.occ();
    while (occupied) {
        Square sq(occupied.pop());
        Piece  p = board.at<Piece>(sq);
        Color  c = p.color();
        PieceType t = p.type();

        // Flip the square vertically if we're viewing the board as Black,
        // exactly like the PST mirroring trick in Section 3.
        int sqIndex = (us == Color::WHITE) ? sq.index() : (sq.index() ^ 56);

        // Plane 0-5 = "my" pieces, plane 6-11 = "their" pieces.
        int plane = static_cast<int>(t) + ((c == us) ? 0 : 6);

        active.push_back(plane * 64 + sqIndex);
    }
}

// Run the position through the trained network and return a centipawn
// score from the perspective of the side to move — same convention as
// evaluateClassic(), so the rest of the engine can't tell the difference.
int evaluateNNUE(const Board& board) {
    std::vector<int> active;
    extractActiveFeatures(board, active);

    // ── Layer 1: input -> hidden1, then ReLU ──────────────────────
    // Since almost all 768 inputs are 0, we only add up the weights
    // for the handful of features that are actually "on" (active).
    float hidden1[NNUE_HIDDEN1_SIZE];
    for (int j = 0; j < NNUE_HIDDEN1_SIZE; j++) {
        hidden1[j] = nnue.B1[j];
    }
    for (int featureIndex : active) {
        const float* row = &nnue.W1[featureIndex * NNUE_HIDDEN1_SIZE];
        for (int j = 0; j < NNUE_HIDDEN1_SIZE; j++) {
            hidden1[j] += row[j];
        }
    }
    for (int j = 0; j < NNUE_HIDDEN1_SIZE; j++) {
        if (hidden1[j] < 0.0f) hidden1[j] = 0.0f;   // ReLU
    }

    // ── Layer 2: hidden1 -> hidden2, then ReLU ────────────────────
    float hidden2[NNUE_HIDDEN2_SIZE];
    for (int k = 0; k < NNUE_HIDDEN2_SIZE; k++) {
        float sum = nnue.B2[k];
        for (int j = 0; j < NNUE_HIDDEN1_SIZE; j++) {
            sum += hidden1[j] * nnue.W2[j * NNUE_HIDDEN2_SIZE + k];
        }
        hidden2[k] = (sum < 0.0f) ? 0.0f : sum;     // ReLU
    }

    // ── Layer 3: hidden2 -> output (no activation — raw score) ────
    float output = nnue.B3[0];
    for (int k = 0; k < NNUE_HIDDEN2_SIZE; k++) {
        output += hidden2[k] * nnue.W3[k];
    }

    return static_cast<int>(output);
}


// ════════════════════════════════════════════════════════════════
// SECTION 4C — EVALUATION DISPATCH
//
// Every other part of the engine (quiescence, alpha-beta, the "eval"
// debug command, training data export) calls evaluate(). This one
// function decides which evaluator actually does the work:
//
//   • If NNUE weights are loaded AND useNNUE is turned on → use the network.
//   • Otherwise → fall back to the classic hand-written evaluator.
//
// This means you can flip between the two at any time.
// ════════════════════════════════════════════════════════════════

int evaluate(const Board& board) {
    if (useNNUE && nnue.loaded) {
        return evaluateNNUE(board);
    }
    return evaluateClassic(board);
}


// ════════════════════════════════════════════════════════════════
// SECTION 5 — MOVE ORDERING
//
// Alpha-beta pruning works best when we try the BEST moves first.
// If we find a great move early, we establish a high "alpha" value,
// which lets us skip (prune) many bad moves without searching them.
//
// Our simple ordering strategy:
//   Priority 1: Queen promotions  — turning a pawn into a queen is almost always great
//   Priority 2: Captures          — ordered by MVV-LVA (see below)
//   Priority 3: Everything else   — quiet moves, ordered by 0 (no preference)
//
// MVV-LVA (Most Valuable Victim / Least Valuable Attacker):
//   Prefer captures that win the most material.
//   Score = (victim's value × 10) - attacker's value
//   Example: Pawn captures Queen → 900×10 − 100 = 8900   (excellent trade!)
//            Queen captures Pawn → 100×10 − 900 = 100    (risky — queen is exposed)
//
// ════════════════════════════════════════════════════════════════

// Score a move for ordering. Higher score = try this move earlier.
int scoreMoveOrder(const Board& board, const Move& m) {

    // Priority 1: Queen promotions (pawn reaching the far end of the board)
    if (m.typeOf() == Move::PROMOTION && m.promotionType() == PieceType::QUEEN) {
        return 9000000;
    }

    // Priority 2: Captures, scored by MVV-LVA
    if (board.isCapture(m)) {
        // What piece are we capturing? (the victim)
        // En passant always captures a pawn (it's a special pawn capture).
        PieceType victim;
        if (m.typeOf() == Move::ENPASSANT) {
            victim = PieceType::PAWN;
        } else {
            victim = board.at<PieceType>(m.to());   // piece on the destination square
        }

        // What piece are we capturing with? (the attacker)
        PieceType attacker = board.at<PieceType>(m.from());

        // Score = 1,000,000 base (to separate from quiet moves)
        //       + victim value × 10  (prefer more valuable victims)
        //       - attacker value     (prefer cheaper attackers)
        return 1000000 + materialOf(victim) * 10 - materialOf(attacker);
    }

    // Priority 3: All other moves (quiet moves) get score 0
    return 0;
}


// ════════════════════════════════════════════════════════════════
// SECTION 6 — TIME MANAGEMENT
//
// When playing with a clock, the engine must stop thinking before
// time runs out. We check the clock regularly and set a flag
// when the limit is exceeded.
//
// We only check every 2048 nodes to keep overhead low —
// reading the system clock on every single node would slow us down.
// The trick: (n & 2047) == 0 is true only when n is a multiple of
// 2048, because 2047 in binary is eleven 1s.
// ════════════════════════════════════════════════════════════════

long long nodesSearched    = 0;      // Total positions evaluated so far
bool      timeLimitReached = false;  // Becomes true when time is up
int       searchStartMs    = 0;      // Time when current search started
int       timeLimitMs      = 0;      // How long we can think (0 = no limit)

// NEW: when generating training games (Section 9B) we call findBestMove()
// many, many times in a row. Without this flag, the "info depth ..." lines
// it prints would flood the terminal with thousands of lines. Setting this
// to true silences that output during data generation.
bool suppressInfo = false;

// Get the current time in milliseconds using the system clock.
int nowMs() {
    using namespace std::chrono;
    return static_cast<int>(
        duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count()
    );
}

// Check whether we've exceeded our time limit.
void checkTime() {
    if (timeLimitMs <= 0) return;            // No time limit — nothing to check
    if ((nodesSearched & 2047) != 0) return; // Only check every 2048 nodes
    if ((nowMs() - searchStartMs) >= timeLimitMs) {
        timeLimitReached = true;
    }
}


// ════════════════════════════════════════════════════════════════
// SECTION 7 — QUIESCENCE SEARCH
//
// PROBLEM: The engine searches to a fixed depth, then evaluates.
// But what if the last move was a piece capture that can be
// immediately recaptured? The evaluation would be wrong!
//
// EXAMPLE WITHOUT QUIESCENCE:
//   Depth 5: White captures Black's queen. Search stops. Score looks great!
//   Reality: Black immediately recaptures. White is NOT actually winning.
//   This mistake is called the "horizon effect."
//
// SOLUTION: When we reach depth 0, instead of evaluating straight away,
// we keep searching CAPTURES ONLY until no captures are left.
// Only then do we evaluate the "quiet" (stable) position.
//
// STAND-PAT: At any point in quiescence, we're allowed to "do nothing"
// and just return the current static evaluation. If the current position
// is already so good that it beats beta even without making any capture,
// we stop (prune). This is called stand-pat.
// ════════════════════════════════════════════════════════════════

int quiescence(Board& board, int alpha, int beta, int ply) {
    nodesSearched++;
    checkTime();
    if (timeLimitReached) return 0;

    // STAND-PAT: evaluate the position as-is (without making any capture)
    // NOTE: evaluate() (Section 4C) automatically uses NNUE if it's loaded.
    int standPat = evaluate(board);

    // If the stand-pat score already beats beta, the opponent won't allow
    // this position — return beta immediately (prune this branch).
    if (standPat >= beta) return beta;

    // If stand-pat is better than our current best (alpha), update alpha.
    if (standPat > alpha) alpha = standPat;

    // Generate ONLY capture moves from this position.
    // movegen::MoveGenType::CAPTURE tells the library we only want captures.
    Movelist captures;
    movegen::legalmoves<movegen::MoveGenType::CAPTURE>(captures, board);

    // Sort captures by MVV-LVA so we try the best captures first.
    for (int i = 0; i < static_cast<int>(captures.size()); i++) {
        captures[i].setScore(
            static_cast<int16_t>(scoreMoveOrder(board, captures[i]))
        );
    }
    std::sort(captures.begin(), captures.end(),
              [](const Move& a, const Move& b) { return a.score() > b.score(); });

    // Try each capture and keep track of the best score.
    for (int i = 0; i < static_cast<int>(captures.size()); i++) {
        board.makeMove(captures[i]);
        // Recursively search. We NEGATE because we're now in the opponent's position.
        int score = -quiescence(board, -beta, -alpha, ply + 1);
        board.unmakeMove(captures[i]);

        if (timeLimitReached) return 0;
        if (score >= beta) return beta;        // Beta cutoff — stop searching
        if (score > alpha) alpha = score;      // New best score found
    }

    return alpha;  // Return the best score we found
}


// ════════════════════════════════════════════════════════════════
// SECTION 8 — ALPHA-BETA SEARCH
//
// This is the core algorithm that makes the engine "think."
// It searches 'depth' moves ahead and returns the best achievable
// score for whoever is currently to move.
//
// ── HOW NEGAMAX WORKS ────────────────────────────────────────
//
// We use a version called "Negamax" where we always maximise the
// score for the side to move. When we make a move and search deeper,
// we are now in the OPPONENT's position — they also want to maximise
// THEIR score. But maximising the opponent's score = minimising ours.
// So we simply negate the score when we come back up:
//
//   my_score = -(opponent's best score from the next position)
//
// This lets us write ONE search function instead of two (one for
// White maximising, one for Black minimising). Both sides use the
// same code and just negate at each level.
//
// ── HOW ALPHA-BETA PRUNING WORKS ─────────────────────────────
//
// alpha = the best score WE are already guaranteed to get
//         (starts at -INF = "we haven't found anything good yet")
// beta  = the best score the OPPONENT is already guaranteed to get
//         (starts at +INF = "opponent hasn't limited us yet")
//
// As we search, if a move gives us a score >= beta, it means the
// opponent already has a way (elsewhere in the tree) to avoid this
// line entirely — they'd never allow us to reach this position.
// We can SKIP all remaining moves at this node. This is a "beta cutoff."
//
// This can make the search MASSIVELY faster without missing anything.
// In the best case, it reduces the work from branching_factor^depth
// to roughly branching_factor^(depth/2).
//
// ── WHAT WE HAVE REMOVED (compared to advanced engines) ─────
//
// This version does NOT include:
//   • Null move pruning      — too tricky (risk of missing checkmates)
//   • Late Move Reductions   — complex two-pass search
//   • Transposition table    — complex hash table management

// ════════════════════════════════════════════════════════════════

int alphaBeta(Board& board, int depth, int alpha, int beta, int ply) {
    nodesSearched++;
    checkTime();
    if (timeLimitReached) return 0;

    // ── Draw detection ───────────────────────────────────────────
    // The library checks all draw conditions for us automatically.

    // 50-move rule: if 50 full moves pass without a capture or pawn push, it's a draw.
    if (board.isHalfMoveDraw()) {
        auto [reason, result] = board.getHalfMoveDrawType();
        return (reason == GameResultReason::CHECKMATE) ? -MATE_SCORE + ply : DRAW_SCORE;
    }

    // Threefold repetition: the same position appears three times → draw.
    // We detect at 1 repeat (conservative) to avoid repeating loops.
    if (board.isRepetition(1)) return DRAW_SCORE;

    // Insufficient material: e.g. King vs King → neither side can force checkmate.
    if (board.isInsufficientMaterial()) return DRAW_SCORE;

    // ── Leaf node: use quiescence search ────────────────────────
    // We've searched as deep as we planned to. Instead of evaluating
    // immediately, run quiescence to finish off any active captures.
    if (depth <= 0) {
        return quiescence(board, alpha, beta, ply);
    }

    // ── Generate all legal moves ─────────────────────────────────
    // The chess library gives us ONLY legal moves — we never have to
    // worry about generating illegal ones.
    Movelist moves;
    movegen::legalmoves(moves, board);

    // ── Terminal position: no moves available ────────────────────
    bool inCheck = board.inCheck();
    if (moves.empty()) {
        if (inCheck) {
            // No moves + in check = checkmate! We lose.
            // We add ply so that "mate in 1" scores higher than "mate in 3" —
            // the engine prefers to delay being mated as long as possible.
            return -MATE_SCORE + ply;
        } else {
            // No moves + not in check = stalemate! That's a draw.
            return DRAW_SCORE;
        }
    }

    // ── Order moves: try the best-looking ones first ─────────────
    // Captures and promotions get high scores and are tried first.
    // This makes alpha-beta pruning much more effective.
    for (int i = 0; i < static_cast<int>(moves.size()); i++) {
        moves[i].setScore(
            static_cast<int16_t>(scoreMoveOrder(board, moves[i]))
        );
    }
    std::sort(moves.begin(), moves.end(),
              [](const Move& a, const Move& b) { return a.score() > b.score(); });

    // ── Search each move ─────────────────────────────────────────
    int  bestScore = -INF;     // Best score found so far at this node
    Move bestMove  = moves[0]; // Best move found so far

    for (int i = 0; i < static_cast<int>(moves.size()); i++) {
        Move m = moves[i];

        board.makeMove(m);     // Try the move on the board

        // Search the resulting position at one depth less.
        // NEGATE the result: opponent's best = our worst.
        int score = -alphaBeta(board, depth - 1, -beta, -alpha, ply + 1);

        board.unmakeMove(m);   // Undo the move

        if (timeLimitReached) return 0;

        // Did this move give us a better score?
        if (score > bestScore) {
            bestScore = score;
            bestMove  = m;
        }

        // Did it improve our guaranteed minimum (alpha)?
        if (score > alpha) {
            alpha = score;
        }

        // ── Beta cutoff ───────────────────────────────────────────
        // Our score is now >= beta.
        // The opponent already has a way to avoid this situation
        // (they found a better alternative earlier in the tree).
        // There's no point searching more moves — PRUNE!
        if (alpha >= beta) {
            return beta;   // "Fail high" — return the upper bound
        }
    }

    return bestScore;  // Return the best score we found at this node
}


// ════════════════════════════════════════════════════════════════
// SECTION 9 — ITERATIVE DEEPENING
//
// Instead of jumping straight to depth 5, we search:
//   depth 1 → depth 2 → depth 3 → depth 4 → depth 5
//
// WHY? Two reasons:
//
// REASON 1 — Time safety:
//   If the clock runs out during depth 5, we still have a complete,
//   correct best move from depth 4 to return. Without this, we might
//   have nothing to return if time runs out mid-search.
//
// REASON 2 — Better move ordering:
//   The best move found at depth 4 is tried FIRST when searching
//   depth 5. Since the best move from depth 4 is often still best at
//   depth 5, this dramatically improves alpha-beta pruning efficiency.
//   It's like already knowing a good answer before you start searching.
//
// The extra work of re-searching shallower depths is small because
// the tree grows exponentially: depth 5 has about 35× more nodes
// than depth 4, so repeating depth 4 adds only ~3% extra work.
// ════════════════════════════════════════════════════════════════

Move findBestMove(Board& board, int maxDepth, int timeLimitMillis) {
    // Set up search state
    timeLimitMs      = timeLimitMillis;
    timeLimitReached = false;
    searchStartMs    = nowMs();
    nodesSearched    = 0;

    // Safety fallback: pick the first legal move in case we run out of time
    // before completing even depth 1 (extremely unlikely, but safe to handle).
    Move bestMove(Move::NO_MOVE);
    {
        Movelist fallback;
        movegen::legalmoves(fallback, board);
        if (!fallback.empty()) bestMove = fallback[0];
    }

    // Search from depth 1 up to maxDepth
    for (int depth = 1; depth <= maxDepth; depth++) {

        // Generate and order all moves at the root (current position)
        Movelist rootMoves;
        movegen::legalmoves(rootMoves, board);
        if (rootMoves.empty()) break;  // No legal moves — game is over

        // Order moves: put the PREVIOUS best move first.
        // We do this by giving it a very high score.
        // (This is a simple version of the "hash move" idea.)
        for (int i = 0; i < static_cast<int>(rootMoves.size()); i++) {
            int s = scoreMoveOrder(board, rootMoves[i]);
            // Extra boost: try last iteration's best move first
            if (rootMoves[i] == bestMove) s += 50000000;
            rootMoves[i].setScore(static_cast<int16_t>(std::min(s, 30000)));
        }
        std::sort(rootMoves.begin(), rootMoves.end(),
                  [](const Move& a, const Move& b) { return a.score() > b.score(); });

        // Search all root moves at this depth
        Move thisDepthBest  = rootMoves[0];
        int  thisDepthScore = -INF;
        int  alpha = -INF;
        int  beta  =  INF;

        for (int i = 0; i < static_cast<int>(rootMoves.size()); i++) {
            board.makeMove(rootMoves[i]);
            int score = -alphaBeta(board, depth - 1, -beta, -alpha, 1);
            board.unmakeMove(rootMoves[i]);

            // If time ran out mid-search, stop immediately.
            // IMPORTANT: don't update bestMove — the search was incomplete
            //            and the result might be wrong.
            if (timeLimitReached) goto searchDone;

            if (score > thisDepthScore) {
                thisDepthScore = score;
                thisDepthBest  = rootMoves[i];
            }
            if (score > alpha) alpha = score;
        }

        // This whole depth completed cleanly — update the best move
        bestMove = thisDepthBest;

        // Print UCI info so the GUI shows thinking progress.
        // NEW: skipped when suppressInfo is true (used during training
        // data generation in Section 9B, where this would otherwise
        // print thousands of lines per second).
        if (!suppressInfo) {
            int elapsed = nowMs() - searchStartMs;
            int nps = (elapsed > 0)
                    ? static_cast<int>(nodesSearched * 1000LL / elapsed)
                    : 0;
            std::cout << "info depth " << depth
                      << " score cp "  << thisDepthScore
                      << " nodes "     << nodesSearched
                      << " time "      << elapsed
                      << " nps "       << nps
                      << " pv "        << uci::moveToUci(bestMove)
                      << std::endl;
        }

        // If we found a checkmate, no need to search deeper
        if (thisDepthScore >= MATE_SCORE - MAX_PLY) break;
    }

searchDone:
    return bestMove;
}


// ════════════════════════════════════════════════════════════════
// SECTION 9B — TRAINING DATA EXPORT (for training the NNUE later)
//
// To train the NNUE (Section 4B) you need a big file of chess
// positions paired with a "correct" score to learn from. This
// section gives you two simple ways to build that file. Both write
// plain text lines in this format:
//
//     <FEN string>;<score in centipawns, from side-to-move's view>
//
// Example line:
//     rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1;-25
//
// We label every position using evaluateClassic() (NOT NNUE) — this
// way, even before you've trained anything, you can generate a
// dataset that teaches the network to imitate the hand-written
// evaluator. That's a simple, common way to bootstrap an NNUE
// project: train a first network to match the classical eval, then
// later improve on that with real game results.
//
//   "gensfen <numGames> <depth> <file>"
//       Has the engine play numGames quick games AGAINST ITSELF
//       (using its own shallow search to choose most moves, with a
//       little randomness thrown in so games aren't all identical),
//       and writes out one labeled position every few plies. This
//       is a simple way to generate a lot of realistic training
//       positions without needing an external game database.
// ════════════════════════════════════════════════════════════════

// Append one labeled training line for the given position to a file.
void exportPosition(const Board& board, std::ofstream& out) {
    int score = evaluateClassic(board);   // always label with the classic eval
    out << board.getFen() << ";" << score << "\n";
}

// Play numGames short self-play games and record labeled positions.
// depth controls how "smart" the self-play moves are (higher = slower
// but more realistic games). A small depth like 3-4 is plenty to start.
void genSelfPlayData(int numGames, int depth, const std::string& outfile) {
    std::ofstream out(outfile, std::ios::app);
    if (!out) {
        std::cout << "info string Could not open " << outfile << " for writing\n";
        return;
    }

    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> chance(0.0f, 1.0f);

    suppressInfo = true;   // don't flood the console with search info

    int totalPositions = 0;
    for (int g = 0; g < numGames; g++) {
        Board game(constants::STARTPOS);

        for (int ply = 0; ply < 200; ply++) {   // hard cap so games can't run forever
            Movelist moves;
            movegen::legalmoves(moves, game);
            if (moves.empty()) break;   // checkmate or stalemate — game over

            Move chosen;
            // 15% of the time, play a random legal move instead of the
            // engine's own choice. This keeps games from all following
            // the exact same lines, which makes for a more varied,
            // more useful training set.
            if (chance(rng) < 0.15f) {
                std::uniform_int_distribution<int> dist(0, static_cast<int>(moves.size()) - 1);
                chosen = moves[dist(rng)];
            } else {
                chosen = findBestMove(game, depth, 0);
            }

            game.makeMove(chosen);

            // Skip the very first few plies (openings are repetitive and
            // less informative) and skip positions where the side to move
            // is in check (about to make a forced move, less typical).
            if (ply >= 6 && !game.inCheck()) {
                exportPosition(game, out);
                totalPositions++;
            }

            if (game.isHalfMoveDraw() || game.isInsufficientMaterial() || game.isRepetition(1)) {
                break;
            }
        }
    }

    suppressInfo = false;   // restore normal output for future searches

    std::cout << "info string Wrote " << totalPositions << " labeled positions from "
              << numGames << " self-play games to " << outfile << "\n";
}


// ════════════════════════════════════════════════════════════════
// SECTION 10 — PERFT (Move Generation Test)
//
// "Perft" = Performance Test.
// It counts EXACTLY how many legal positions exist at depth d
// from the current position. We compare the result against
// known-correct values to verify nothing is broken.
//
// Since we use the chess library's move generator, these numbers
// should be correct. Running perft is still useful to make sure
// you've set up the library correctly.
//
// EXPECTED VALUES FROM THE STARTING POSITION:
//   perft 1 =          20
//   perft 2 =         400
//   perft 3 =       8,902
//   perft 4 =     197,281
//   perft 5 =   4,865,609
//
// Type "perft 4" in the terminal after loading the engine.
// If it prints 197,281 — everything is working correctly!
// ════════════════════════════════════════════════════════════════

uint64_t perft(Board& board, int depth) {
    if (depth == 0) return 1;  // Base case: this position counts as 1 leaf node

    Movelist moves;
    movegen::legalmoves(moves, board);

    // Optimisation: at depth 1 we can just count moves directly,
    // without making and unmaking each one.
    if (depth == 1) return static_cast<uint64_t>(moves.size());

    uint64_t total = 0;
    for (int i = 0; i < static_cast<int>(moves.size()); i++) {
        board.makeMove(moves[i]);
        total += perft(board, depth - 1);  // Recursively count from the new position
        board.unmakeMove(moves[i]);
    }
    return total;
}

// Perft divide: shows the count for EACH root move separately.
// If your perft total is wrong, compare each move's count against
// a reference engine (like Stockfish's "go perft N" command) to
// find which move is generating wrong positions.
void perftDivide(Board& board, int depth) {
    Movelist moves;
    movegen::legalmoves(moves, board);
    uint64_t total = 0;

    for (int i = 0; i < static_cast<int>(moves.size()); i++) {
        board.makeMove(moves[i]);
        uint64_t count = perft(board, depth - 1);
        board.unmakeMove(moves[i]);
        std::cout << uci::moveToUci(moves[i]) << ": " << count << "\n";
        total += count;
    }
    std::cout << "Total: " << total << "\n";
}


// ════════════════════════════════════════════════════════════════
// SECTION 11 — BOARD DISPLAY
//
// The chess library has no built-in board printer, so we write
// our own. Called by the "d" debug command.
//
// Uppercase letters = White pieces (P N B R Q K)
// Lowercase letters = Black pieces (p n b r q k)
// Dots              = Empty squares
// ════════════════════════════════════════════════════════════════

void printBoard(const Board& board) {
    std::cout << "\n  +---+---+---+---+---+---+---+---+\n";

    // Print rank 8 first (top of board), then rank 7, down to rank 1
    for (int rank = 7; rank >= 0; rank--) {
        std::cout << (rank + 1) << " |";  // Rank number label on the left

        for (int file = 0; file < 8; file++) {
            Square sq(rank * 8 + file);
            Piece  p = board.at<Piece>(sq);
            char   ch = '.';  // Empty square

            if (p != Piece::NONE) {
                // " pnbrqk" — index 0 is a space (padding), then pawn=p, knight=n, etc.
                const char* letters = " pnbrqk";
                ch = letters[static_cast<int>(p.type()) + 1];
                if (p.color() == Color::WHITE) ch = static_cast<char>(toupper(ch));
            }
            std::cout << " " << ch << " |";
        }
        std::cout << "\n  +---+---+---+---+---+---+---+---+\n";
    }

    std::cout << "    a   b   c   d   e   f   g   h\n\n";
    std::cout << "FEN:   " << board.getFen() << "\n";
    std::cout << "Turn:  " << (board.sideToMove() == Color::WHITE ? "White" : "Black") << "\n";
    std::cout << "Check: " << (board.inCheck() ? "YES" : "No") << "\n";
    std::cout << "Eval:  using " << (useNNUE && nnue.loaded ? "NNUE" : "classical") << " evaluator\n\n";
}


// ════════════════════════════════════════════════════════════════
// SECTION 12 — UCI PROTOCOL LOOP
//
// UCI (Universal Chess Interface) is how chess GUIs talk to engines.
// The GUI sends commands as text lines over stdin (standard input).
// The engine reads them and sends replies over stdout (standard output).
//
// Every output line must be printed immediately (flushed), or some
// GUIs will hang waiting. We handle this with cout.setf(unitbuf).
//

// EXTRA COMMANDS (for testing in the terminal, not part of UCI):
//   d              → Draw the board
//   eval           → Print the evaluation score
//   perft N        → Count positions at depth N
//   divide N       → Perft with per-move breakdown
//   moves          → List all legal moves
//   loadnnue <path>            → Load NNUE weights (NEW, same as setoption EvalFile)
//   exportdata <file>          → Append current position as a labeled training line (NEW)
//   gensfen <games> <depth> <file> → Generate self-play training data (NEW)
// ════════════════════════════════════════════════════════════════

// The current chess position the engine is working with
Board mainBoard;

void uciLoop() {
    // unitbuf = flush output after every << operation.
    // Without this, some GUIs receive nothing and time out.
    std::cout.setf(std::ios::unitbuf);

    std::string line;
    while (std::getline(std::cin, line)) {

        if (line.empty()) continue;  // Skip blank lines

        // Split the line into words.
        // ss >> cmd gets the first word (the command name).
        std::istringstream ss(line);
        std::string cmd;
        ss >> cmd;

        // ────────────────────────────────────────────────────
        if (cmd == "uci") {
            std::cout << "id name BeginnerChessEngine\n";
            std::cout << "id author Student\n";
            std::cout << "option name Depth type spin default 5 min 1 max 8\n";
            std::cout << "option name EvalFile type string default <empty>\n";
            std::cout << "option name UseNNUE type check default true\n";
            std::cout << "uciok\n";
        }

        // ────────────────────────────────────────────────────
        else if (cmd == "isready") {
            std::cout << "readyok\n";
        }

        // ────────────────────────────────────────────────────
        else if (cmd == "setoption") {
            
            std::string word, name, value;
            ss >> word;   // should be "name"
            ss >> name;   // the option name, e.g. "EvalFile"
            ss >> word;   // should be "value"
            std::getline(ss, value);
            size_t firstNonSpace = value.find_first_not_of(' ');
            if (firstNonSpace != std::string::npos) {
                value = value.substr(firstNonSpace);
            } else {
                value.clear();
            }

            if (name == "EvalFile") {
                loadNNUEFile(value);
            } else if (name == "UseNNUE") {
                useNNUE = (value == "true" || value == "1");
                std::cout << "info string UseNNUE set to " << (useNNUE ? "true" : "false") << "\n";
            }
            
        }

        // ────────────────────────────────────────────────────
        else if (cmd == "ucinewgame") {
            // Reset everything for a fresh game
            mainBoard = Board(constants::STARTPOS);  // Starting position
        }

        // ────────────────────────────────────────────────────
        else if (cmd == "position") {
            std::string token;
            ss >> token;

            if (token == "startpos") {
                // Load the standard starting position
                mainBoard = Board(constants::STARTPOS);
                ss >> token;  // Read the next word — might be "moves"
            }
            else if (token == "fen") {
                // Read FEN tokens until we hit "moves" or end of line
                std::string fen;
                while (ss >> token && token != "moves") {
                    if (!fen.empty()) fen += " ";
                    fen += token;
                }
                mainBoard = Board(fen);  // Library parses the FEN for us
            }

            // Apply each move in the move list
            if (token == "moves") {
                std::string moveStr;
                while (ss >> moveStr) {
                    // Library converts "e2e4" string → Move object
                    Move m = uci::uciToMove(mainBoard, moveStr);
                    if (m != Move(Move::NO_MOVE)) {
                        mainBoard.makeMove(m);
                    }
                }
            }
        }

        // ────────────────────────────────────────────────────
        else if (cmd == "go") {
            int depth  = 5;   // Default: search 5 moves deep
            int timeMs = 0;   // 0 = no time limit

            std::string token;
            while (ss >> token) {
                if (token == "depth") {
                    ss >> depth;
                }
                else if (token == "movetime") {
                    ss >> timeMs;
                }
                else if (token == "wtime" && mainBoard.sideToMove() == Color::WHITE) {
                    int remaining;
                    ss >> remaining;
                    // Use about 4% of remaining time per move.
                    timeMs = remaining / 25;
                    if (timeMs < 100)   timeMs = 100;    // Never less than 100ms
                    if (timeMs > 10000) timeMs = 10000;  // Never more than 10s
                }
                else if (token == "btime" && mainBoard.sideToMove() == Color::BLACK) {
                    int remaining;
                    ss >> remaining;
                    timeMs = remaining / 25;
                    if (timeMs < 100)   timeMs = 100;
                    if (timeMs > 10000) timeMs = 10000;
                }
                else if (token == "infinite") {
                    timeMs = 0;
                    depth  = MAX_DEPTH;
                }
                // We ignore: winc, binc, movestogo, nodes, mate, searchmoves
            }

            Move best = findBestMove(mainBoard, depth, timeMs);
            std::cout << "bestmove " << uci::moveToUci(best) << "\n";
        }

        // ────────────────────────────────────────────────────
        else if (cmd == "stop") {
            timeLimitReached = true;
        }

        // ────────────────────────────────────────────────────
        else if (cmd == "quit") {
            return;  // Exit the loop (and then main() returns 0)
        }

        // ──── DEBUG COMMANDS (not part of UCI) ──────────────

        // "d" — print the board in ASCII art
        else if (cmd == "d") {
            printBoard(mainBoard);
        }

        // "eval" — show the static score for the current position
        else if (cmd == "eval") {
            int score = evaluate(mainBoard);
            std::cout << "Evaluation: " << score << " cp"
                      << "  (positive = good for the side to move, using "
                      << (useNNUE && nnue.loaded ? "NNUE" : "classical") << ")\n";
        }

        // "perft N" — count all positions at depth N (correctness check)
        else if (cmd == "perft") {
            int d = 5;
            ss >> d;
            std::cout << "Running perft(" << d << ")...\n";
            auto t0 = std::chrono::steady_clock::now();
            uint64_t nodes = perft(mainBoard, d);
            auto t1 = std::chrono::steady_clock::now();
            int ms = static_cast<int>(
                std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()
            );
            std::cout << "Nodes: " << nodes << "  Time: " << ms << " ms";
            if (ms > 0) std::cout << "  NPS: " << (nodes * 1000 / ms);
            std::cout << "\n";
        }

        // "divide N" — perft with per-move counts (for debugging)
        else if (cmd == "divide") {
            int d = 1;
            ss >> d;
            perftDivide(mainBoard, d);
        }

        // "moves" — list all legal moves from the current position
        else if (cmd == "moves") {
            Movelist ml;
            movegen::legalmoves(ml, mainBoard);
            std::cout << "Legal moves (" << ml.size() << "): ";
            for (int i = 0; i < static_cast<int>(ml.size()); i++) {
                std::cout << uci::moveToUci(ml[i]);
                if (i + 1 < static_cast<int>(ml.size())) std::cout << " ";
            }
            std::cout << "\n";
        }

        // "loadnnue <path>" — load NNUE weights from a file (NEW)
        else if (cmd == "loadnnue") {
            std::string path;
            ss >> path;
            if (path.empty()) {
                std::cout << "Usage: loadnnue <path-to-weights-file>\n";
            } else {
                loadNNUEFile(path);
            }
        }

        // "exportdata <file>" — append the CURRENT position as one
        // labeled training line (FEN;score) to <file> (NEW)
        else if (cmd == "exportdata") {
            std::string path;
            ss >> path;
            if (path.empty()) path = "training_data.csv";

            std::ofstream out(path, std::ios::app);
            if (!out) {
                std::cout << "Could not open " << path << " for writing\n";
            } else {
                exportPosition(mainBoard, out);
                std::cout << "Exported 1 position to " << path << "\n";
            }
        }

        // "gensfen <games> <depth> <file>" — generate self-play training
        // data by having the engine play quick games against itself (NEW)
        else if (cmd == "gensfen") {
            int numGames = 100;
            int depth    = 4;
            std::string outfile = "training_data.csv";

            ss >> numGames;
            ss >> depth;
            std::string maybeFile;
            if (ss >> maybeFile) outfile = maybeFile;

            std::cout << "Generating training data: " << numGames
                      << " games at depth " << depth << " -> " << outfile
                      << " (this may take a while)...\n";
            genSelfPlayData(numGames, depth, outfile);
        }
    }
}


// ════════════════════════════════════════════════════════════════
// SECTION 13 — MAIN
//
// main() is where the program starts.
// We set up the board and enter the UCI loop.
// The loop runs until the GUI sends "quit".
// ════════════════════════════════════════════════════════════════

int main() {
    // Set up the standard chess starting position.
    // constants::STARTPOS is a FEN string defined in chess.hpp.
    mainBoard = Board(constants::STARTPOS);

    loadNNUEFile("nnue.bin");

    // Start the UCI communication loop.
    // This reads commands from the GUI (or terminal) and replies.
    uciLoop();

    return 0;
}


/*
════════════════════════════════════════════════════════════════════
  QUICK REFERENCE — TESTING COMMANDS
════════════════════════════════════════════════════════════════════

  COMPILE:
    g++ -std=c++17 -O2 -o engine chess_engine_simple.cpp

  RUN:
    ./engine

  TYPE THESE ONE BY ONE (press Enter after each line):

    uci                            → prints "uciok"
    isready                        → prints "readyok"
    position startpos
    d                              → displays the board
    eval                           → score should be near 0 (balanced)
    moves                          → lists all 20 legal opening moves
    perft 4                        → should print exactly: Nodes: 197281
    go depth 5                     → engine thinks and prints bestmove

    position startpos moves e2e4 e7e5
    d                              → board after 1. e4 e5
    go depth 5                     → engine plays as White

    quit                           → exits the program

  PERFT CORRECT VALUES (from starting position):
    perft 1 =          20
    perft 2 =         400
    perft 3 =       8,902
    perft 4 =     197,281   ← most important check
    perft 5 =   4,865,609


  ALGORITHMS USED :
    ✓ Alpha-Beta Search (the main search algorithm)
    ✓ Negamax formulation (one function for both sides)
    ✓ Quiescence Search (avoids evaluating mid-capture)
    ✓ Simple Move Ordering (captures first with MVV-LVA)
    ✓ Iterative Deepening (search 1,2,3...N for time safety)
    ✓ Material evaluation (piece values)
    ✓ Piece-Square Tables (positional bonuses)
    ✓ Bishop pair bonus
    ✓ Rook open file bonus
    ✓ NNUE evaluation (small feedforward net, weights trained externally)
    ✓ Self-play training data generation (for bootstrapping the NNUE)

  Next steps (will add later):
    ✗ Transposition Table         (complex hash table and bound types)
    ✗ Null Move Pruning           (risk of zugzwang errors)
    ✗ Late Move Reductions        (two-pass search with narrow windows)
    ✗ Killer Moves                (state shared across branches)
    ✗ History Heuristic           (needs careful table management)
    ✗ Incremental NNUE accumulator (real NNUE engines update the hidden
                                     layer move-by-move instead of
                                     recomputing it from scratch every
                                     time — much faster, but a lot more
                                     bookkeeping than a beginner project
                                     needs to start with)

════════════════════════════════════════════════════════════════════
*/
