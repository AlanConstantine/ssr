from __future__ import annotations

import argparse
from pathlib import Path


STRUCTURES = [
    ('Frame0_Li_1DMC_id0.xyz', 'Li_1DMC', [('Li', 0, 0, 0), ('O', 1.9, 0, 0), ('C', 2.6, 0.4, 0), ('H', 3.0, 1.0, 0)]),
    ('Frame1_Li_1DMC_id1.xyz', 'Li_1DMC', [('Li', 0, 0, 0), ('O', 2.0, 0.1, 0), ('C', 2.7, 0.3, 0), ('H', 3.2, 0.9, 0)]),
    ('Frame2_Li_2EC_id2.xyz', 'Li_2EC', [('Li', 0, 0, 0), ('O', 1.8, 0, 0), ('O', 0, 1.8, 0), ('C', 2.4, 0.5, 0)]),
    ('Frame3_Li_2EC_id3.xyz', 'Li_2EC', [('Li', 0, 0, 0), ('O', 1.9, 0.1, 0), ('O', 0.2, 1.7, 0), ('C', 2.5, 0.2, 0)]),
    ('Frame4_Li_1EC_1DMC_id4.xyz', 'Li_1EC_1DMC', [('Li', 0, 0, 0), ('O', 1.9, 0, 0), ('O', 0, 2.6, 0), ('C', 2.6, 0.2, 0)]),
    ('Frame5_Li_1EC_1DMC_id5.xyz', 'Li_1EC_1DMC', [('Li', 0, 0, 0), ('O', 2.0, 0.2, 0), ('O', 0.1, 2.5, 0), ('C', 2.7, 0.1, 0)]),
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for filename, signature, atoms in STRUCTURES:
        lines = [str(len(atoms)), f'signature: {signature}']
        lines += [f'{sym} {x:.3f} {y:.3f} {z:.3f}' for sym, x, y, z in atoms]
        (out_dir / filename).write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
