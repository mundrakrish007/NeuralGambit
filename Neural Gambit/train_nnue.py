"""
train_nnue.py
=============

Trains the small neural network described in Section 4B of
chess_engine_simple.cpp, using the training data produced by the
engine's "gensfen" / "exportdata" commands, and saves the weights in
the exact binary format the engine's loadnnue() function expects.

BEFORE RUNNING, INSTALL THE TWO LIBRARIES THIS SCRIPT NEEDS:

    pip install torch chess

(python-chess is only used to read FEN strings and figure out where
the pieces are — it does not play or search chess, that's still all
done by your C++ engine.)

HOW TO RUN:

    python train_nnue.py --data training_data.csv --out nnue.bin

Then in the engine:

    loadnnue nnue.bin
    eval

------------------------------------------------------------------
WHY THE FEATURE EXTRACTION BELOW MUST MATCH THE C++ CODE EXACTLY
------------------------------------------------------------------
The engine describes every position as 768 numbers (12 "planes" of
64 squares each), always from the point of view of whoever is about
to move:

    plane 0-5  = MY   pawn, knight, bishop, rook, queen, king
    plane 6-11 = THEIR pawn, knight, bishop, rook, queen, king
    feature index = plane * 64 + square

    If it's White to move, squares are numbered a1=0 ... h8=63 as usual.
    If it's Black to move, every square is flipped vertically
    (square XOR 56) so the network always "sees" the board the same
    way regardless of which side is moving.

If this script computed features ANY differently than the C++ code
does, the trained weights would be meaningless once loaded into the
engine - so double check this part first if something looks wrong.
"""

import argparse
import random
import struct

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import chess   # python-chess: only used to parse FEN strings


# ════════════════════════════════════════════════════════════════
# STEP 1 — FEATURE EXTRACTION (must match Section 4B of the C++ file)
# ════════════════════════════════════════════════════════════════

NUM_FEATURES = 768  # 12 planes x 64 squares — same constant as NNUE_INPUT_SIZE in the engine


def fen_to_features(fen: str) -> np.ndarray:
    """Turn one FEN string into the same 768-length 0/1 vector the
    engine's extractActiveFeatures() would build for that position."""

    board = chess.Board(fen)
    features = np.zeros(NUM_FEATURES, dtype=np.float32)

    us = board.turn  # True = White to move, False = Black to move

    for square in chess.SQUARES:               # 0..63, a1=0 ... h8=63 (same numbering as chess.hpp)
        piece = board.piece_at(square)
        if piece is None:
            continue

        # Mirror the square vertically if Black is to move — identical
        # trick to the PST mirroring and NNUE mirroring in the C++ code.
        sq_index = square if us else (square ^ 56)

        # python-chess piece_type: PAWN=1 ... KING=6. The engine's
        # PieceType numbering is PAWN=0 ... KING=5, so subtract 1.
        piece_plane = piece.piece_type - 1

        # Plane 0-5 = "my" pieces, plane 6-11 = "their" pieces —
        # exactly like extractActiveFeatures() in the engine.
        plane = piece_plane if (piece.color == us) else (piece_plane + 6)

        features[plane * 64 + sq_index] = 1.0

    return features


# ════════════════════════════════════════════════════════════════
# STEP 2 — DATASET: reads the "FEN;score" lines gensfen/exportdata wrote
# ════════════════════════════════════════════════════════════════

class ChessPositionDataset(Dataset):
    def __init__(self, csv_path: str, score_clip: float = 2000.0):
        self.samples = []   # list of (fen, score) pairs

        with open(csv_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or ";" not in line:
                    continue
                fen, score_str = line.rsplit(";", 1)
                try:
                    score = float(score_str)
                except ValueError:
                    continue

                # Clip extreme values. evaluateClassic() only reports
                # material + positional bonuses (no mate scores), so
                # this is mostly just a safety net against outliers.
                score = max(-score_clip, min(score_clip, score))
                self.samples.append((fen, score))

        print(f"Loaded {len(self.samples)} labeled positions from {csv_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fen, score = self.samples[idx]
        features = fen_to_features(fen)
        return (
            torch.from_numpy(features),
            torch.tensor([score], dtype=torch.float32),
        )


# ════════════════════════════════════════════════════════════════
# STEP 3 — THE NETWORK: 768 -> 256 (ReLU) -> 32 (ReLU) -> 1
# Same shape as NNUE_HIDDEN1_SIZE / NNUE_HIDDEN2_SIZE in the engine.
# ════════════════════════════════════════════════════════════════

class NNUE(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(768, 256)
        self.fc2 = nn.Linear(256, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)   # no activation on the output — raw centipawn score
        return x


# ════════════════════════════════════════════════════════════════
# STEP 4 — SAVE WEIGHTS IN THE EXACT BINARY LAYOUT loadnnue() EXPECTS
#
#   W1 (768x256), B1 (256), W2 (256x32), B2 (32), W3 (32x1), B3 (1)
#   All float32, row-major, no header, in that order.
#
# PyTorch's nn.Linear stores its weight matrix as (out_features,
# in_features) — the OPPOSITE shape/order the engine wants
# (in_features rows, out_features columns). So we transpose each
# weight matrix before saving.
# ════════════════════════════════════════════════════════════════

def save_nnue_weights(model: NNUE, path: str):
    def w(layer):
        # layer.weight shape is (out, in) -> transpose to (in, out)
        # to match the engine's [input][hidden] row-major layout.
        return np.ascontiguousarray(
            layer.weight.detach().cpu().numpy().T.astype(np.float32)
        )

    def b(layer):
        return np.ascontiguousarray(
            layer.bias.detach().cpu().numpy().astype(np.float32)
        )

    with open(path, "wb") as f:
        f.write(w(model.fc1).tobytes())   # W1: 768 x 256
        f.write(b(model.fc1).tobytes())   # B1: 256
        f.write(w(model.fc2).tobytes())   # W2: 256 x 32
        f.write(b(model.fc2).tobytes())   # B2: 32
        f.write(w(model.fc3).tobytes())   # W3: 32 x 1
        f.write(b(model.fc3).tobytes())   # B3: 1

    total_floats = (768 * 256) + 256 + (256 * 32) + 32 + (32 * 1) + 1
    print(f"Saved {total_floats} floats ({total_floats * 4} bytes) to {path}")


# ════════════════════════════════════════════════════════════════
# STEP 5 — THE TRAINING LOOP
# ════════════════════════════════════════════════════════════════

def train(args):
    full_dataset = ChessPositionDataset(args.data, score_clip=args.score_clip)
    if len(full_dataset) == 0:
        print("No data found — check that your CSV path is correct and "
              "gensfen/exportdata actually wrote some lines.")
        return

    # Split into train / validation sets so we can tell if the network
    # is actually learning or just memorising.
    val_size = max(1, int(len(full_dataset) * args.val_fraction))
    train_size = len(full_dataset) - val_size
    train_set, val_set = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    model = NNUE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)

            optimizer.zero_grad()
            predictions = model(features)
            loss = loss_fn(predictions, targets)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * features.size(0)

        train_loss = train_loss_sum / len(train_set)

        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(device), targets.to(device)
                predictions = model(features)
                loss = loss_fn(predictions, targets)
                val_loss_sum += loss.item() * features.size(0)
        val_loss = val_loss_sum / len(val_set)

        # RMSE (root mean squared error) is easier to read than raw MSE —
        # it's in centipawns, same units as the scores themselves.
        train_rmse = train_loss ** 0.5
        val_rmse = val_loss ** 0.5
        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train RMSE: {train_rmse:7.1f} cp   val RMSE: {val_rmse:7.1f} cp")

        # Save the best-so-far network so you keep the version that
        # generalises best, not necessarily the very last epoch.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_nnue_weights(model, args.out)
            print(f"  -> new best model saved to {args.out}")

    print("Training complete.")


# ════════════════════════════════════════════════════════════════
# COMMAND-LINE ENTRY POINT
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the engine's NNUE evaluator.")
    parser.add_argument("--data", type=str, default="training_data.csv",
                         help="Path to the CSV file written by gensfen/exportdata")
    parser.add_argument("--out", type=str, default="nnue.bin",
                         help="Where to save the trained weights (load with 'loadnnue')")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.1,
                         help="Fraction of data held out for validation")
    parser.add_argument("--score-clip", type=float, default=2000.0,
                         help="Clip labels to +/- this many centipawns before training")
    args = parser.parse_args()

    random.seed(42)
    torch.manual_seed(42)

    train(args)
