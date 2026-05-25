from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder


PALETTE = [
    '#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#ff7f0e',
    '#17becf', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings', required=True)
    parser.add_argument('--metadata_csv', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--label_column', default='signature')
    parser.add_argument('--max_heatmap_samples', type=int, default=80)
    return parser.parse_args()


def read_metadata(path: str, label_column: str) -> tuple[list[str], list[str]]:
    labels = []
    names = []
    with open(path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        if label_column not in (reader.fieldnames or []):
            raise ValueError(f'label column {label_column!r} not found in {path}')
        for row in reader:
            labels.append(row[label_column])
            names.append(Path(row.get('path', '')).name)
    return names, labels


def write_scatter_svg(points: np.ndarray, labels: list[str], out_path: Path) -> None:
    encoded = LabelEncoder().fit_transform(labels)
    x = points[:, 0]
    y = points[:, 1]
    x = (x - x.min()) / max(float(x.max() - x.min()), 1e-8)
    y = (y - y.min()) / max(float(y.max() - y.min()), 1e-8)
    width, height, pad = 720, 520, 36
    circles = []
    for xi, yi, label_id in zip(x, y, encoded):
        cx = pad + xi * (width - 2 * pad)
        cy = height - pad - yi * (height - 2 * pad)
        color = PALETTE[int(label_id) % len(PALETTE)]
        circles.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="5" fill="{color}" opacity="0.82" />')
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        '<rect width="100%" height="100%" fill="white"/>\n'
        '<text x="24" y="28" font-family="Arial" font-size="16">Embedding PCA projection</text>\n'
        + '\n'.join(circles)
        + '\n</svg>\n'
    )
    out_path.write_text(svg)


def write_heatmap_svg(matrix: np.ndarray, out_path: Path) -> None:
    n = matrix.shape[0]
    cell = max(4, min(12, 560 // max(n, 1)))
    pad = 32
    size = pad * 2 + n * cell
    rects = []
    for i in range(n):
        for j in range(n):
            value = (matrix[i, j] + 1.0) / 2.0
            red = int(255 * value)
            blue = int(255 * (1.0 - value))
            rects.append(
                f'<rect x="{pad + j * cell}" y="{pad + i * cell}" width="{cell}" height="{cell}" '
                f'fill="rgb({red},40,{blue})" />'
            )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}">\n'
        '<rect width="100%" height="100%" fill="white"/>\n'
        '<text x="12" y="22" font-family="Arial" font-size="14">Cosine similarity heatmap</text>\n'
        + '\n'.join(rects)
        + '\n</svg>\n'
    )
    out_path.write_text(svg)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    embeddings = np.load(args.embeddings)
    _, labels = read_metadata(args.metadata_csv, args.label_column)
    if len(labels) != embeddings.shape[0]:
        raise ValueError('metadata row count does not match embeddings row count')

    if embeddings.shape[0] >= 2:
        points = PCA(n_components=2, random_state=0).fit_transform(embeddings)
    else:
        points = np.zeros((embeddings.shape[0], 2), dtype=np.float32)
    write_scatter_svg(points, labels, out_dir / 'embedding_pca.svg')

    sample_count = min(args.max_heatmap_samples, embeddings.shape[0])
    sim = cosine_similarity(embeddings[:sample_count])
    write_heatmap_svg(sim, out_dir / 'similarity_heatmap.svg')


if __name__ == '__main__':
    main()
