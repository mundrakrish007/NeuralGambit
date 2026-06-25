/*
=============================================================
  CHESS ENGINE — Single-File, Built on Disservin's chess-library
=============================================================

  SETUP:
    1. Put chess.hpp in the SAME folder as this file.
       Download: https://github.com/Disservin/chess-library
       (Releases → chess.hpp → Save As)

  COMPILE:
    g++ -std=c++17 -O2 -o chess_engine chess_engine_lib.cpp

  RUN:
    ./chess_engine
    > uci          → engine says uciok
    > isready      → readyok
    > position startpos
    > go depth 5   → bestmove ...

  CONNECT TO GUI:
    Download Cute Chess (free) from cutechess.com
    Tools → Engines → Add → point to ./chess_engine, Protocol = UCI

  HOW THIS FILE IS ORGANISED :
    1.  Includes and setup
    2.  Constants
    3.  Piece-Square Tables (our own — library doesn't provide these)
    4.  [Board struct REMOVED — we use chess::Board from the library]
    5.  [FEN parser REMOVED — library does it: Board(fen) and getFen()]
    6.  [Attack helpers REMOVED — library does it: isAttacked(), inCheck()]
    7.  Move scoring struct (for ordering)
    8.  [Move generation REMOVED — library: movegen::legalmoves()]
    9.  [Make/Undo REMOVED — library: board.makeMove(), board.unmakeMove()]
    10. [Check detection REMOVED — library: board.inCheck()]
    11. Position evaluation (we write this — the library has no eval)
    12. Transposition table
    13. Search (Alpha-Beta + Iterative Deepening — we write this)
    14. UCI protocol loop
    15. main()
=============================================================
*/

// ─────────────────────────────────────────────────────────
// SECTION 1: INCLUDES
// ─────────────────────────────────────────────────────────

// The Disservin chess library — one header, handles everything board-related.
// It gives us: Board, Move, Movelist, movegen::legalmoves,
//              board.makeMove/unmakeMove, board.inCheck(), board.hash(), etc.
#include "chess.hpp"

#include <iostream>   // cin / cout
#include <string>     // std::string
#include <sstream>    // splitting UCI tokens
#include <algorithm>  // std::sort
#include <cstring>    // memset
#include <chrono>     // timing

// Pull the chess namespace in so we can write "Board" instead of "chess::Board"
using namespace chess;

// ─────────────────────────────────────────────────────────
// SECTION 2: CONSTANTS
// ─────────────────────────────────────────────────────────

// All scores are in centipawns (cp).  1 pawn = 100 cp.
// The search always returns from the SIDE-TO-MOVE's perspective:
//   Positive → side to move is winning
//   Negative → side to move is losing
static const int INF        = 1'000'000;
static const int MATE_SCORE =   900'000;  // base mate score (never reached in normal play)
static const int DRAW_SCORE =         0;

// Search limits
static const int MAX_DEPTH = 64;   // max search depth (iterative deepening stops here)
static const int MAX_PLY   = 128;  // max ply in any search path (depth × extensions)

// Material values in centipawns.
// Indexed by PieceType: PAWN=0, KNIGHT=1, BISHOP=2, ROOK=3, QUEEN=4, KING=5
// (These match chess::PieceType::underlying enum values)
static const int MATERIAL[6] = {
    100,    // PAWN
    320,    // KNIGHT
    330,    // BISHOP  — slightly more than knight (better in open positions)
    500,    // ROOK
    900,    // QUEEN
    20000   // KING    — enormous so the engine never "trades" its king
};

// Helper: get material value from a chess::PieceType
inline int materialOf(PieceType pt) {
    return MATERIAL[static_cast<int>(pt)];
}

// ─────────────────────────────────────────────────────────
// SECTION 3: PIECE-SQUARE TABLES (PSTs)
//
// For each piece type, a 64-value table that adds a bonus or
// penalty depending on WHERE the piece sits.
// Examples:
//   Knight in the centre → bonus    (PST_KNIGHT[e4] = +20)
//   Knight in the corner → penalty  (PST_KNIGHT[a1] = -50)
//   King behind pawns    → bonus    (PST_KING_MID[g1] = +30)
//
// The library doesn't provide these — this is our engine's "knowledge".
//
// Layout: index 0 = a1, index 7 = h1, index 56 = a8, index 63 = h8
// (This matches chess-library's Square::index() convention.)
//
// Tables are written from White's perspective.
// For Black we mirror vertically: mirroredIndex = index ^ 56
// (XOR 56 flips the rank bits while keeping the file bits.)
// ─────────────────────────────────────────────────────────

// Each table is laid out rank-by-rank from rank 1 (bottom) to rank 8 (top)

static const int PST_PAWN[64] = {
     0,  0,  0,  0,  0,  0,  0,  0,   // rank 1 (impossible for pawns)
     5, 10, 10,-20,-20, 10, 10,  5,   // rank 2 (starting rank — small penalty to block centre)
     5, -5,-10,  0,  0,-10, -5,  5,   // rank 3
     0,  0,  0, 20, 20,  0,  0,  0,   // rank 4 (e4/d4 advance = bonus)
     5,  5, 10, 25, 25, 10,  5,  5,   // rank 5
    10, 10, 20, 30, 30, 20, 10, 10,   // rank 6
    50, 50, 50, 50, 50, 50, 50, 50,   // rank 7 (one push from promotion!)
     0,  0,  0,  0,  0,  0,  0,  0    // rank 8 (promotion rank — handled by promo logic)
};

static const int PST_KNIGHT[64] = {
    -50,-40,-30,-30,-30,-30,-40,-50,  // rank 1 — corners/edges are bad
    -40,-20,  0,  5,  5,  0,-20,-40,  // rank 2
    -30,  5, 10, 15, 15, 10,  5,-30,  // rank 3
    -30,  0, 15, 20, 20, 15,  0,-30,  // rank 4 — centre is best
    -30,  5, 15, 20, 20, 15,  5,-30,  // rank 5
    -30,  0, 10, 15, 15, 10,  0,-30,  // rank 6
    -40,-20,  0,  0,  0,  0,-20,-40,  // rank 7
    -50,-40,-30,-30,-30,-30,-40,-50   // rank 8
};

static const int PST_BISHOP[64] = {
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -20,-10,-10,-10,-10,-10,-10,-20
};

static const int PST_ROOK[64] = {
     0,  0,  0,  5,  5,  0,  0,  0,   // rank 1
    -5,  0,  0,  0,  0,  0,  0, -5,   // rank 2
    -5,  0,  0,  0,  0,  0,  0, -5,   // rank 3
    -5,  0,  0,  0,  0,  0,  0, -5,   // rank 4
    -5,  0,  0,  0,  0,  0,  0, -5,   // rank 5
    -5,  0,  0,  0,  0,  0,  0, -5,   // rank 6
     5, 10, 10, 10, 10, 10, 10,  5,   // rank 7 — rook on 7th rank is powerful!
     0,  0,  0,  0,  0,  0,  0,  0    // rank 8
};

static const int PST_QUEEN[64] = {
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -10,  5,  5,  5,  5,  5,  0,-10,
      0,  0,  5,  5,  5,  5,  0, -5,
     -5,  0,  5,  5,  5,  5,  0, -5,
    -10,  0,  5,  5,  5,  5,  0,-10,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20
};

// King in the middlegame: hide! Castle early, stay behind pawns.
static const int PST_KING_MID[64] = {
     20, 30, 10,  0,  0, 10, 30, 20,   // rank 1 — g1/c1 (castled) are good
     20, 20,  0,  0,  0,  0, 20, 20,   // rank 2
    -10,-20,-20,-20,-20,-20,-20,-10,   // rank 3 — venturing forward is dangerous
    -20,-30,-30,-40,-40,-30,-30,-20,   // rank 4
    -30,-40,-40,-50,-50,-40,-40,-30,   // rank 5
    -30,-40,-40,-50,-50,-40,-40,-30,   // rank 6
    -30,-40,-40,-50,-50,-40,-40,-30,   // rank 7
    -30,-40,-40,-50,-50,-40,-40,-30    // rank 8
};

// King in the endgame: centralise! No pieces to attack the exposed king.
static const int PST_KING_END[64] = {
    -50,-30,-30,-30,-30,-30,-30,-50,
    -30,-30,  0,  0,  0,  0,-30,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-20,-10,  0,  0,-10,-20,-30,
    -50,-40,-30,-20,-20,-30,-40,-50
};

// Look up the PST bonus for a piece of given type and color on a given square.
// sq.index() follows chess-library convention: a1=0, h8=63.
inline int pstScore(PieceType pt, Color color, Square sq) {
    // Mirror for Black: sq ^ 56 flips rank bits, keeping file bits.
    int idx = (color == Color::WHITE) ? sq.index() : (sq.index() ^ 56);
    switch (static_cast<int>(pt)) {
        case 0: return PST_PAWN[idx];
        case 1: return PST_KNIGHT[idx];
        case 2: return PST_BISHOP[idx];
        case 3: return PST_ROOK[idx];
        case 4: return PST_QUEEN[idx];
        case 5: return PST_KING_MID[idx]; // endgame table used separately below
        default: return 0;
    }
}

// ─────────────────────────────────────────────────────────
// SECTION 4: TRANSPOSITION TABLE (TT)
//
// A hash table that caches search results so we don't re-search
// the same position when it is reached via different move orders
// (a "transposition" in the game tree).
//
// The chess library maintains a Zobrist hash for us (board.hash()),
// so we don't need to write our own hash computation.
//
// Each TT entry stores:
//   hash  — full Zobrist key (to confirm the entry is for this position)
//   score — the cached score
//   depth — how deep the subtree was searched
//   flag  — whether the score is exact, a lower bound, or an upper bound
//   move  — the best move found (used for move ordering)
// ─────────────────────────────────────────────────────────

enum class TTFlag : uint8_t {
    NONE  = 0,
    EXACT = 1,  // score is exact: we found the true minimax value
    LOWER = 2,  // lower bound: a beta cutoff occurred (score ≥ beta)
    UPPER = 3   // upper bound: score never exceeded alpha (fail-low)
};

struct TTEntry {
    uint64_t hash  = 0;
    int      score = 0;
    int      depth = 0;
    TTFlag   flag  = TTFlag::NONE;
    Move     move  = Move(Move::NO_MOVE); // best move (for move ordering)
};

// Power-of-2 size → fast index via bitwise AND instead of expensive modulo.
// 1<<22 = ~4 million entries × ~24 bytes each ≈ 96 MB.
// Reduce to 1<<20 (~24 MB) if your machine is memory-constrained.
static const int TT_BITS = 22;
static const int TT_SIZE = 1 << TT_BITS;
static const int TT_MASK = TT_SIZE - 1;

TTEntry tt[TT_SIZE];

void clearTT() {
    for (auto& e : tt) e = TTEntry{};
}

// Store a result in the TT ("always replace" — simplest replacement policy)
void ttStore(uint64_t hash, int score, int depth, TTFlag flag, Move move) {
    TTEntry& e = tt[hash & TT_MASK];
    e.hash  = hash;
    e.score = score;
    e.depth = depth;
    e.flag  = flag;
    e.move  = move;
}

// Probe the TT.
// Returns true if we found a usable cached score.
// Always fills hashMove if any entry exists at this slot (even if score unusable).
bool ttProbe(uint64_t hash, int depth, int alpha, int beta,
             int& score, Move& hashMove) {
    TTEntry& e = tt[hash & TT_MASK];
    if (e.hash != hash) return false;   // wrong position (hash collision)

    hashMove = e.move;                  // always grab stored move for ordering

    if (e.depth < depth) return false;  // searched too shallowly — can't trust score

    if (e.flag == TTFlag::EXACT)                       { score = e.score; return true; }
    if (e.flag == TTFlag::LOWER && e.score >= beta)    { score = e.score; return true; }
    if (e.flag == TTFlag::UPPER && e.score <= alpha)   { score = e.score; return true; }
    return false;
}

// ─────────────────────────────────────────────────────────
// SECTION 5: MOVE ORDERING TABLES
//
// Alpha-beta pruning cuts the most branches when the BEST moves
// are tried first. We maintain two heuristic tables:
//
//  killers[2][MAX_PLY]
//    Quiet moves (non-captures) that caused a beta cutoff at the
//    same depth in a sibling node. Two "killer slots" per ply.
//    If they worked in a nearby branch, they may work here too.
//
//  history[2][64][64]
//    A score for every (color, from, to) combination of quiet moves.
//    Incremented by depth² whenever a quiet move causes a beta cutoff.
//    High history score → the move was often good in the past.
// ─────────────────────────────────────────────────────────

Move killers[2][MAX_PLY];           // killers[slot][ply]
int  history[2][64][64];            // history[colorIdx][from][to]

void clearMoveOrdering() {
    for (int p = 0; p < MAX_PLY; p++) {
        killers[0][p] = Move(Move::NO_MOVE);
        killers[1][p] = Move(Move::NO_MOVE);
    }
    memset(history, 0, sizeof(history));
}

// ─────────────────────────────────────────────────────────
// SECTION 6: POSITION EVALUATION
//
// Returns a score in centipawns from the SIDE-TO-MOVE's perspective.
// The search (negamax) always maximises, so a positive score means
// "good for whoever is to move right now".
//
// What we evaluate:
//   • Material: sum of piece values for each side
//   • Piece-Square Tables: positional bonuses per piece
//   • Bishop pair: two bishops together ≈ +30 cp
//   • Rook on open/semi-open file: small bonus
//   • Game phase: use endgame king table when few pieces remain
//
// The chess library provides piece bitboards (board.pieces()),
// square queries (board.at<PieceType>(sq)), and side bitboards
// (board.us() / board.them()), which makes this fast and clean.
// ─────────────────────────────────────────────────────────

// Count non-pawn, non-king material to estimate the game phase.
// Knights/Bishops = 1 pt, Rooks = 2 pt, Queens = 4 pt.
// Phase 0 = pure endgame. Phase 24 = full middlegame.
static int gamePhase(const Board& board) {
    int phase = 0;
    for (Color c : {Color::WHITE, Color::BLACK}) {
        phase += 1 * board.pieces(PieceType::KNIGHT, c).count();
        phase += 1 * board.pieces(PieceType::BISHOP, c).count();
        phase += 2 * board.pieces(PieceType::ROOK,   c).count();
        phase += 4 * board.pieces(PieceType::QUEEN,  c).count();
    }
    return (phase < 24) ? phase : 24;
}

// Build a bitboard of all squares on a given file (0=a, 7=h)
static Bitboard fileBB(int fileIdx) {
    Bitboard bb;
    for (int rank = 0; rank < 8; rank++) bb.set(rank * 8 + fileIdx);
    return bb;
}

int evaluate(const Board& board) {
    int score = 0;
    int phase = gamePhase(board);
    bool isEndgame = (phase <= 8); // treat as endgame when little material left

    // ── Material + PST ──────────────────────────────────────
    // Iterate over each square using the Bitboard of ALL occupied squares.
    // board.occ() returns a Bitboard where each set bit = an occupied square.
    // We pop() bits one at a time (pop() removes and returns the lowest set bit).
    Bitboard occupied = board.occ();
    while (occupied) {
        // Get next occupied square (and remove it from the bitboard)
        Square sq(occupied.pop());

        // Query what piece is on this square (returns chess::Piece)
        Piece  p = board.at<Piece>(sq);
        Color  c = p.color();
        PieceType pt = p.type();
        int    sign = (c == Color::WHITE) ? +1 : -1;

        // Material value
        score += sign * materialOf(pt);

        // Piece-Square Table bonus
        // King uses a different table in endgame
        if (pt == PieceType::KING && isEndgame) {
            int idx = (c == Color::WHITE) ? sq.index() : (sq.index() ^ 56);
            score += sign * PST_KING_END[idx];
        } else {
            score += sign * pstScore(pt, c, sq);
        }
    }

    // ── Bishop pair bonus (+30 cp) ───────────────────────────
    // Two same-colored bishops together coordinate powerfully.
    if (board.pieces(PieceType::BISHOP, Color::WHITE).count() >= 2) score += 30;
    if (board.pieces(PieceType::BISHOP, Color::BLACK).count() >= 2) score -= 30;

    // ── Rook on open / semi-open file bonus ─────────────────
    // Open file (no pawns at all) = +20 cp per rook.
    // Semi-open (no OWN pawns, but enemy pawns present) = +10 cp.
    for (Color c : {Color::WHITE, Color::BLACK}) {
        int    sign  = (c == Color::WHITE) ? +1 : -1;
        Bitboard rooks = board.pieces(PieceType::ROOK, c);
        while (rooks) {
            Square rSq(rooks.pop());
            int    f    = static_cast<int>(rSq.file());
            Bitboard file = fileBB(f);
            Bitboard allPawns  = board.pieces(PieceType::PAWN);
            Bitboard ownPawns  = board.pieces(PieceType::PAWN, c);
            Bitboard enemyPawns= board.pieces(PieceType::PAWN, ~c);
            bool openFile  = (allPawns   & file).empty();
            bool semiOpen  = (ownPawns   & file).empty() && !(enemyPawns & file).empty();
            if (openFile) score += sign * 20;
            else if (semiOpen) score += sign * 10;
        }
    }

    // ── Return from side-to-move's perspective ───────────────
    // All arithmetic above was "positive = white winning".
    // Negamax always maximises, so we flip the sign for Black.
    return (board.sideToMove() == Color::WHITE) ? score : -score;
}

// ─────────────────────────────────────────────────────────
// SECTION 7: SEARCH HELPERS
//
// Score a move for ordering (higher = try earlier).
// Good ordering is critical: alpha-beta prunes the most when
// the best moves are tried first.
//
// Priority order:
//   1. TT/hash move        — proved best in a previous search
//   2. Queen promotions    — almost always good
//   3. Winning captures    — MVV-LVA (Most Valuable Victim / Least Valuable Attacker)
//   4. Killer moves        — quiet moves that caused beta cutoffs at this ply
//   5. History heuristic   — quiet moves that were historically good
// ─────────────────────────────────────────────────────────

int scoreMoveOrder(const Board& board, const Move& m,
                   const Move& hashMove, int ply) {

    // 1. Hash move: use the best move stored from a previous search.
    //    Almost always the best or near-best move again.
    if (m == hashMove) return 10'000'000;

    // 2. Queen promotion — almost always better than any quiet move.
    if (m.typeOf() == Move::PROMOTION &&
        m.promotionType() == PieceType::QUEEN)
        return 9'000'000;

    // 3. Captures — ordered by MVV-LVA.
    //    Capture value = (victim material × 10) − attacker material
    //    e.g. pawn×queen = 900×10 − 100 = 8900  (huge!)
    //         queen×pawn = 100×10 − 900 = 100   (risky)
    if (board.isCapture(m)) {
        // For en passant, victim is always a pawn
        PieceType victim = (m.typeOf() == Move::ENPASSANT)
                         ? PieceType::PAWN
                         : board.at<PieceType>(m.to());
        PieceType attacker = board.at<PieceType>(m.from());
        return 1'000'000 + materialOf(victim) * 10 - materialOf(attacker);
    }

    // 4. Killer moves (quiet moves that recently caused beta cutoffs nearby)
    if (m == killers[0][ply]) return 900'000;
    if (m == killers[1][ply]) return 800'000;

    // 5. History heuristic (quiet moves that were historically strong)
    int colorIdx = (board.sideToMove() == Color::WHITE) ? 0 : 1;
    return history[colorIdx][m.from().index()][m.to().index()];
}

// Update killers when a quiet move causes a beta cutoff
void updateKillers(const Move& m, int ply) {
    if (m != killers[0][ply]) {
        killers[1][ply] = killers[0][ply];
        killers[0][ply] = m;
    }
}

// Update history when a quiet move causes a beta cutoff
void updateHistory(const Board& board, const Move& m, int depth) {
    int colorIdx = (board.sideToMove() == Color::WHITE) ? 0 : 1;
    int& h = history[colorIdx][m.from().index()][m.to().index()];
    h += depth * depth;
    if (h > 20'000) h = 20'000; // cap to prevent int overflow
}

// ─────────────────────────────────────────────────────────
// SECTION 8: SEARCH — TIME MANAGEMENT
// ─────────────────────────────────────────────────────────

long long nodesSearched     = 0;
bool      timeLimitReached  = false;
int       searchStartMs     = 0;
int       timeLimitMs       = 0;   // 0 = no time limit

int nowMs() {
    using namespace std::chrono;
    return static_cast<int>(
        duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count()
    );
}

// Called every 2048 nodes to avoid expensive clock reads on every single node
void checkTime() {
    if (timeLimitMs <= 0) return;
    if ((nodesSearched & 2047) != 0) return;
    if ((nowMs() - searchStartMs) >= timeLimitMs)
        timeLimitReached = true;
}

// ─────────────────────────────────────────────────────────
// SECTION 9: QUIESCENCE SEARCH
//
// At depth 0 we don't just call evaluate() — we keep searching
// CAPTURES until the position is quiet (no captures left).
// This avoids the "horizon effect":
//
//   Without quiescence: engine captures a queen at depth 5,
//   search ends, score looks great. But the queen is immediately
//   recaptured at depth 6 — and the engine never sees that!
//
//   With quiescence: we keep searching captures past the horizon
//   until no more are available. Only then do we evaluate.
//
// "Stand-pat": at any point in quiescence, we can "do nothing"
// (return the static evaluation). If stand-pat ≥ beta, we prune.
// ─────────────────────────────────────────────────────────

int quiescence(Board& board, int alpha, int beta, int ply) {
    nodesSearched++;
    checkTime();
    if (timeLimitReached) return 0;

    // Stand-pat: assume we can choose not to make any capture
    int standPat = evaluate(board);
    if (standPat >= beta) return beta;   // already too good — opponent won't allow it
    if (standPat > alpha) alpha = standPat;

    // Generate ONLY captures (the library supports this directly)
    Movelist captures;
    movegen::legalmoves<movegen::MoveGenType::CAPTURE>(captures, board);

    // Order captures by MVV-LVA
    Move dummy(Move::NO_MOVE);
    for (int i = 0; i < static_cast<int>(captures.size()); i++)
        captures[i].setScore(
            static_cast<int16_t>(scoreMoveOrder(board, captures[i], dummy, ply))
        );
    std::sort(captures.begin(), captures.end(),
              [](const Move& a, const Move& b) { return a.score() > b.score(); });

    for (int i = 0; i < static_cast<int>(captures.size()); i++) {
        board.makeMove(captures[i]);
        int score = -quiescence(board, -beta, -alpha, ply + 1);
        board.unmakeMove(captures[i]);

        if (timeLimitReached) return 0;
        if (score >= beta) return beta;
        if (score > alpha) alpha = score;
    }
    return alpha;
}

// ─────────────────────────────────────────────────────────
// SECTION 10: ALPHA-BETA SEARCH (Negamax formulation)
//
// The engine's brain. Searches 'depth' plies ahead and returns
// the best achievable score from the current side-to-move's perspective.
//
// NEGAMAX: always "maximise my score" = "minimise opponent's score"
//   score = -search(child)   (negate because child returns opponent's perspective)
//
// ALPHA-BETA PRUNING:
//   alpha = best score WE are guaranteed so far
//   beta  = best score OPPONENT is guaranteed so far
//   If score ≥ beta → "beta cutoff": opponent already has something better,
//   they'll never let us reach this position. Skip remaining moves.
//
// ENHANCEMENTS beyond basic alpha-beta:
//   • Transposition table   — cache searched positions
//   • Null move pruning     — if passing our turn still beats beta, prune
//   • Late Move Reductions  — search late moves at reduced depth
//   • Check extension       — extend by 1 ply when in check
// ─────────────────────────────────────────────────────────

int alphaBeta(Board& board, int depth, int alpha, int beta, int ply,
              bool nullMoveAllowed = true) {
    nodesSearched++;
    checkTime();
    if (timeLimitReached) return 0;

    // ── Draw detection ───────────────────────────────────────
    // The library checks all draw conditions for us.
    // isHalfMoveDraw: 50-move rule.  isRepetition: threefold repetition.
    // isInsufficientMaterial: K vs K, K+B vs K, etc.
    if (board.isHalfMoveDraw()) {
        auto [reason, result] = board.getHalfMoveDrawType();
        return (reason == GameResultReason::CHECKMATE) ? -MATE_SCORE + ply : DRAW_SCORE;
    }
    if (board.isRepetition(1))           return DRAW_SCORE;
    if (board.isInsufficientMaterial())  return DRAW_SCORE;

    // ── Transposition table probe ────────────────────────────
    Move hashMove(Move::NO_MOVE);
    int  ttScore = 0;
    if (ttProbe(board.hash(), depth, alpha, beta, ttScore, hashMove))
        return ttScore;

    // ── Leaf node: enter quiescence ──────────────────────────
    if (depth <= 0)
        return quiescence(board, alpha, beta, ply);

    // ── Null move pruning ────────────────────────────────────
    // If we "pass" our turn and the opponent STILL can't beat beta,
    // our position is so strong we can prune.
    // Skip when:
    //   - In check (can't legally make a null move)
    //   - At root (ply 0) — we must return a real move there
    //   - Side has only king + pawns (risk of zugzwang)
    //   - Previous move was also a null move (no double null)
    bool inCheck = board.inCheck();
    if (nullMoveAllowed && !inCheck && ply > 0 && depth >= 3 &&
        board.hasNonPawnMaterial(board.sideToMove())) {
        board.makeNullMove();
        // Reduced-depth search (R=3 means we skip 3 plies)
        int nullScore = -alphaBeta(board, depth - 3, -beta, -beta + 1, ply + 1,
                                   /*nullMoveAllowed=*/false);
        board.unmakeNullMove();
        if (timeLimitReached) return 0;
        if (nullScore >= beta) return beta; // null move cutoff!
    }

    // ── Generate all legal moves ─────────────────────────────
    // The library produces ONLY legal moves — no pseudo-legal filtering needed.
    Movelist moves;
    movegen::legalmoves(moves, board);

    // ── Terminal node: checkmate or stalemate ────────────────
    if (moves.empty()) {
        if (inCheck) return -MATE_SCORE + ply; // checkmate — +ply prefers faster mates
        else         return DRAW_SCORE;         // stalemate
    }

    // ── Score and sort moves ──────────────────────────────────
    for (int i = 0; i < static_cast<int>(moves.size()); i++)
        moves[i].setScore(
            static_cast<int16_t>(scoreMoveOrder(board, moves[i], hashMove, ply))
        );
    std::sort(moves.begin(), moves.end(),
              [](const Move& a, const Move& b) { return a.score() > b.score(); });

    // ── Search each move ──────────────────────────────────────
    int  bestScore     = -INF;
    Move bestMove      = moves[0];
    int  originalAlpha = alpha;

    for (int i = 0; i < static_cast<int>(moves.size()); i++) {
        Move m = moves[i];

        // ── Check extension ─────────────────────────────────
        // Extend search by 1 ply when in check — tactics in check need full depth.
        int extension = inCheck ? 1 : 0;

        board.makeMove(m);
        int score;

        // ── Late Move Reductions (LMR) ───────────────────────
        // Moves late in the sorted list are less likely to be best.
        // Search them at reduced depth first. If the score looks promising
        // (beats alpha), re-search at full depth to confirm.
        if (i >= 4 && depth >= 3 && !board.isCapture(m) &&
            m.typeOf() != Move::PROMOTION && !inCheck) {
            // Reduced search (depth - 2)
            score = -alphaBeta(board, depth - 2, -alpha - 1, -alpha, ply + 1);
            // If it looks good, do a proper full-depth re-search
            if (score > alpha)
                score = -alphaBeta(board, depth - 1 + extension, -beta, -alpha, ply + 1);
        } else {
            score = -alphaBeta(board, depth - 1 + extension, -beta, -alpha, ply + 1);
        }

        board.unmakeMove(m);
        if (timeLimitReached) return 0;

        if (score > bestScore) {
            bestScore = score;
            bestMove  = m;
        }

        // ── Alpha update ──────────────────────────────────────
        if (score > alpha) alpha = score;

        // ── Beta cutoff ───────────────────────────────────────
        if (alpha >= beta) {
            // Update move ordering tables for quiet moves
            if (!board.isCapture(m)) {
                updateKillers(m, ply);
                updateHistory(board, m, depth);
            }
            ttStore(board.hash(), beta, depth, TTFlag::LOWER, m);
            return beta;
        }
    }

    // ── Store result in TT ────────────────────────────────────
    TTFlag flag = (bestScore > originalAlpha) ? TTFlag::EXACT : TTFlag::UPPER;
    ttStore(board.hash(), bestScore, depth, flag, bestMove);

    return bestScore;
}

// ─────────────────────────────────────────────────────────
// SECTION 11: ITERATIVE DEEPENING
//
// Instead of searching to depth N directly, we search:
//   depth 1 → depth 2 → depth 3 → ... → depth N
//
// Why? Two reasons:
//   1. Time safety: if time expires during depth d+1, we still
//      have a complete best move from depth d to return.
//   2. Better ordering: the best move from depth d is used as
//      the TT hash move when searching depth d+1, greatly
//      improving alpha-beta efficiency.
// ─────────────────────────────────────────────────────────

Move findBestMove(Board& board, int maxDepth, int timeLimitMillis) {
    timeLimitMs     = timeLimitMillis;
    timeLimitReached= false;
    searchStartMs   = nowMs();
    nodesSearched   = 0;
    clearMoveOrdering();

    // Safe fallback: first legal move (in case even depth 1 is interrupted)
    Move bestMove(Move::NO_MOVE);
    {
        Movelist fallback;
        movegen::legalmoves(fallback, board);
        if (!fallback.empty()) bestMove = fallback[0];
    }

    for (int depth = 1; depth <= maxDepth; depth++) {
        // Generate and score root moves
        Movelist rootMoves;
        movegen::legalmoves(rootMoves, board);
        if (rootMoves.empty()) break;

        // Use the overall best move from the previous depth as the TT/hash move
        for (int i = 0; i < static_cast<int>(rootMoves.size()); i++)
            rootMoves[i].setScore(static_cast<int16_t>(
                scoreMoveOrder(board, rootMoves[i], bestMove, 0)
            ));
        std::sort(rootMoves.begin(), rootMoves.end(),
                  [](const Move& a, const Move& b) { return a.score() > b.score(); });

        Move depthBestMove = rootMoves[0];
        int  depthBestScore= -INF;
        int  alpha = -INF, beta = INF;

        for (int i = 0; i < static_cast<int>(rootMoves.size()); i++) {
            board.makeMove(rootMoves[i]);
            int score = -alphaBeta(board, depth - 1, -beta, -alpha, 1);
            board.unmakeMove(rootMoves[i]);

            if (timeLimitReached) goto done;

            if (score > depthBestScore) {
                depthBestScore = score;
                depthBestMove  = rootMoves[i];
            }
            if (score > alpha) alpha = score;
        }

        // Only update bestMove when this depth completed cleanly
        bestMove = depthBestMove;

        // ── UCI info line ──────────────────────────────────
        {
            int elapsed = nowMs() - searchStartMs;
            int nps     = (elapsed > 0)
                        ? static_cast<int>(nodesSearched * 1000LL / elapsed) : 0;
            std::cout << "info depth "  << depth
                      << " score cp "   << depthBestScore
                      << " nodes "      << nodesSearched
                      << " time "       << elapsed
                      << " nps "        << nps
                      << " pv "         << uci::moveToUci(bestMove)
                      << std::endl;
        }

        // No point searching deeper if we already found checkmate
        if (depthBestScore >= MATE_SCORE - MAX_PLY) break;
    }

done:
    return bestMove;
}

// ─────────────────────────────────────────────────────────
// SECTION 12: PERFT — MOVE GENERATION TESTING
//
// Count exactly the number of leaf nodes at depth d.
// Because the library generates legal moves, these counts should
// match the known-correct values exactly:
//
//   Starting position:
//   depth 1 →         20
//   depth 2 →        400
//   depth 3 →      8,902
//   depth 4 →    197,281
//   depth 5 →  4,865,609
//
// Run "perft 4" after loading the engine and verify 197,281.
// If it matches, move generation is correct.
// ─────────────────────────────────────────────────────────

uint64_t perft(Board& board, int depth) {
    if (depth == 0) return 1;

    Movelist moves;
    movegen::legalmoves(moves, board);

    // Optimisation: at depth 1, just count moves without recursing
    if (depth == 1) return static_cast<uint64_t>(moves.size());

    uint64_t nodes = 0;
    for (int i = 0; i < static_cast<int>(moves.size()); i++) {
        board.makeMove(moves[i]);
        nodes += perft(board, depth - 1);
        board.unmakeMove(moves[i]);
    }
    return nodes;
}

// Perft divide: print the count for each root move.
// Useful when perft gives a wrong total — compare per-move counts
// against a reference engine to find which move has the bug.
void perftDivide(Board& board, int depth) {
    Movelist moves;
    movegen::legalmoves(moves, board);
    uint64_t total = 0;
    for (int i = 0; i < static_cast<int>(moves.size()); i++) {
        board.makeMove(moves[i]);
        uint64_t cnt = perft(board, depth - 1);
        board.unmakeMove(moves[i]);
        std::cout << uci::moveToUci(moves[i]) << ": " << cnt << "\n";
        total += cnt;
    }
    std::cout << "Total: " << total << "\n";
}

// ─────────────────────────────────────────────────────────
// SECTION 13: BOARD DISPLAY HELPER
//
// The library does not have a built-in pretty-printer,
// so we write a simple one using board.at<Piece>(sq).
// ─────────────────────────────────────────────────────────

void printBoard(const Board& board) {
    std::cout << "\n  +---+---+---+---+---+---+---+---+\n";
    for (int rank = 7; rank >= 0; rank--) {
        std::cout << (rank + 1) << " |";
        for (int file = 0; file < 8; file++) {
            Square sq(rank * 8 + file);
            Piece  p = board.at<Piece>(sq);
            char   ch = '.';
            if (p != Piece::NONE) {
                // " pnbrqk" indexed by PieceType (0=pawn,1=knight,...,5=king)
                const char* sym = " pnbrqk";
                ch = sym[static_cast<int>(p.type()) + 1];
                if (p.color() == Color::WHITE) ch = static_cast<char>(toupper(ch));
            }
            std::cout << " " << ch << " |";
        }
        std::cout << "\n  +---+---+---+---+---+---+---+---+\n";
    }
    std::cout << "    a   b   c   d   e   f   g   h\n\n";
    std::cout << "FEN:  " << board.getFen() << "\n";
    std::cout << "Side: " << (board.sideToMove() == Color::WHITE ? "White" : "Black") << "\n";
    std::cout << "Check: " << (board.inCheck() ? "Yes" : "No") << "\n\n";
}

// ─────────────────────────────────────────────────────────
// SECTION 14: UCI PROTOCOL LOOP
//
// UCI (Universal Chess Interface) is the standard way chess GUIs
// talk to engines. Everything goes through stdin → stdout, line by line.
//
// Standard UCI commands handled:
//   uci            → identify engine, print options, say "uciok"
//   isready        → print "readyok"
//   ucinewgame     → reset board and tables for a new game
//   position ...   → set up a position (startpos or fen) + optional moves
//   go ...         → search and print "bestmove XYZ"
//   stop           → (synchronous engine: search already done)
//   quit           → exit
//
// Extra debug commands (type in terminal):
//   d              → print board
//   eval           → print static evaluation
//   perft N        → run perft at depth N
//   divide N       → perft with per-move breakdown
//   moves          → list all legal moves
// ─────────────────────────────────────────────────────────

// The engine's current board
Board mainBoard;

void uciLoop() {
    // UCI requires every output line to be flushed immediately.
    // Without this, some GUIs hang waiting for output stuck in a buffer.
    std::cout.setf(std::ios::unitbuf);

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;

        std::istringstream ss(line);
        std::string cmd;
        ss >> cmd;

        // ── uci ────────────────────────────────────────────
        if (cmd == "uci") {
            std::cout << "id name BeginnerChessEngine\n";
            std::cout << "id author LearningC++\n";
            std::cout << "option name Depth    type spin default 5 min 1 max 8\n";
            std::cout << "option name Hash     type spin default 64 min 1 max 512\n";
            std::cout << "uciok\n";
        }

        // ── isready ────────────────────────────────────────
        else if (cmd == "isready") {
            std::cout << "readyok\n";
        }

        // ── setoption ──────────────────────────────────────
        // We accept options but keep things simple for now.
        else if (cmd == "setoption") {
            // Format: "setoption name X value Y"
            // (Ignored — depth and movetime come from the "go" command)
        }

        // ── ucinewgame ─────────────────────────────────────
        else if (cmd == "ucinewgame") {
            mainBoard = Board(constants::STARTPOS); // library constant for start FEN
            clearTT();
            clearMoveOrdering();
        }

        // ── position ───────────────────────────────────────
        // Two forms:
        //   "position startpos [moves e2e4 e7e5 ...]"
        //   "position fen <fen string> [moves ...]"
        else if (cmd == "position") {
            std::string token;
            ss >> token;

            if (token == "startpos") {
                mainBoard = Board(constants::STARTPOS);
                ss >> token; // may read "moves" next
            } else if (token == "fen") {
                // FEN has up to 6 space-separated parts; read until "moves" or end
                std::string fen;
                while (ss >> token && token != "moves")
                    fen += (fen.empty() ? "" : " ") + token;
                mainBoard = Board(fen);
                // token is now either "moves" or we hit end-of-stream
            }

            // Apply the move list (if any)
            if (token == "moves") {
                std::string moveStr;
                while (ss >> moveStr) {
                    // The library parses "e2e4" → Move object for us
                    Move m = uci::uciToMove(mainBoard, moveStr);
                    if (m != Move(Move::NO_MOVE))
                        mainBoard.makeMove(m);
                }
            }
        }

        // ── go ─────────────────────────────────────────────
        // Start the search. Returns "bestmove XYZ" when done.
        //
        // Supported sub-commands:
        //   depth N       → search exactly N plies
        //   movetime N    → think for exactly N milliseconds
        //   wtime/btime N → use ~4% of remaining clock time
        //   infinite      → search until "stop" (use depth limit as fallback)
        else if (cmd == "go") {
            int  depth   = 5;   // default search depth
            int  timeMs  = 0;   // 0 = no time limit
            std::string tok;

            while (ss >> tok) {
                if (tok == "depth") {
                    ss >> depth;
                } else if (tok == "movetime") {
                    ss >> timeMs;
                } else if ((tok == "wtime" && mainBoard.sideToMove() == Color::WHITE) ||
                           (tok == "btime" && mainBoard.sideToMove() == Color::BLACK)) {
                    int remaining;
                    ss >> remaining;
                    // Use ~4% of remaining time (very simple time management).
                    // A production engine considers: increment, moves to go, etc.
                    timeMs = remaining / 25;
                    timeMs = std::max(timeMs, 100);    // at least 100 ms
                    timeMs = std::min(timeMs, 10'000); // at most 10 s
                } else if (tok == "infinite") {
                    timeMs = 0;          // no time limit
                    depth  = MAX_DEPTH;  // search as deep as possible
                }
                // Silently ignore: winc, binc, movestogo, nodes, mate, searchmoves
            }

            Move best = findBestMove(mainBoard, depth, timeMs);
            std::cout << "bestmove " << uci::moveToUci(best) << "\n";
        }

        // ── stop ───────────────────────────────────────────
        else if (cmd == "stop") {
            timeLimitReached = true;
        }

        // ── quit ───────────────────────────────────────────
        else if (cmd == "quit") {
            return;
        }

        // ─────────────────────────────────────────────────
        // DEBUG COMMANDS (not part of UCI, for terminal use)
        // ─────────────────────────────────────────────────

        // "d" — pretty-print the current board
        else if (cmd == "d") {
            printBoard(mainBoard);
        }

        // "eval" — show the static evaluation score
        else if (cmd == "eval") {
            int score = evaluate(mainBoard);
            std::cout << "Eval: " << score << " cp"
                      << " (positive = good for side to move)\n";
        }

        // "perft N" — count nodes (verifies move generation correctness)
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
            if (ms > 0) std::cout << "  NPS: " << nodes * 1000 / ms;
            std::cout << "\n";
        }

        // "divide N" — perft per root move (for debugging wrong counts)
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
            for (int i = 0; i < static_cast<int>(ml.size()); i++)
                std::cout << uci::moveToUci(ml[i]) << " ";
            std::cout << "\n";
        }
    }
}

// ─────────────────────────────────────────────────────────
// SECTION 15: MAIN
// ─────────────────────────────────────────────────────────

int main() {
    // Set up the starting position using the library's constant
    mainBoard = Board(constants::STARTPOS);

    // Clear the transposition table
    clearTT();

    // Enter the UCI loop — runs until "quit"
    uciLoop();

    return 0;
}

/*
=============================================================
  QUICK REFERENCE — TEST COMMANDS
=============================================================

  Compile:
    g++ -std=c++17 -O2 -o chess_engine chess_engine_lib.cpp

  Run and type:
    uci                              → uciok
    isready                          → readyok
    position startpos
    d                                → print board
    eval                             → static evaluation (~0)
    moves                            → list 20 legal moves
    perft 4                          → should print 197281
    go depth 5                       → bestmove ...
    position startpos moves e2e4 e7e5
    d                                → board after 1.e4 e5
    go depth 5                       → engine plays as White
    quit                             → exit

  PERFT EXPECTED VALUES (starting position):
    depth 1 →         20  ✓
    depth 2 →        400  ✓
    depth 3 →      8,902  ✓
    depth 4 →    197,281  ✓
    depth 5 →  4,865,609  ✓

  WHAT THE LIBRARY REPLACES (vs hand-written engine):
    ✓ Board struct          → chess::Board
    ✓ FEN parsing           → Board(fen_string) and board.getFen()
    ✓ Move struct           → chess::Move
    ✓ Move generation       → movegen::legalmoves(list, board)
    ✓ makeMove / undoMove   → board.makeMove(m) / board.unmakeMove(m)
    ✓ isAttacked / inCheck  → board.isAttacked(sq, color) / board.inCheck()
    ✓ Capture detection     → board.isCapture(m)
    ✓ Zobrist hash          → board.hash()
    ✓ Draw detection        → board.isRepetition(), isHalfMoveDraw(), isInsufficientMaterial()
    ✓ Null moves            → board.makeNullMove() / board.unmakeNullMove()
    ✓ UCI move strings      → uci::moveToUci(m), uci::uciToMove(board, str)

  WHAT WE STILL WRITE OURSELVES:
    ✓ Piece-Square Tables   (library has none)
    ✓ evaluate()            (library has no eval)
    ✓ Transposition table   (library has no TT)
    ✓ alphaBeta() search    (library has no search)
    ✓ findBestMove()        (library has no iterative deepening)
    ✓ Move ordering         (library has no ordering heuristics)
    ✓ UCI loop              (library has no UCI protocol)
=============================================================
*/
