
#!/usr/bin/env python3
import numpy as np, csv
from run_estimation import generate_dataset
from tictactoe_play_utils import X, O

def board_to_string(board): return ''.join('.XO'[v] for v in board)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=1000)
    ap.add_argument("--theta", type=float, nargs=6, default=[3.0,2.0,0.5,0.3,0.0,0.6])
    ap.add_argument("--out", type=str, default="tictactoe_dataset.csv")
    args = ap.parse_args()

    data, theta_star = generate_dataset(n_games=args.games, theta_star=np.array(args.theta))
    with open(args.out, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(["board","player","action"])
        for (b,p,a) in data: w.writerow([board_to_string(b), p, a])
    print(f"Wrote {len(data)} rows to {args.out} with theta*={theta_star}")

if __name__ == "__main__":
    main()
