"""Convert UDCHealth pred_gold.jsonl to eval_overlay_*.json format
for compatibility with compute_ddi_pkl.py.

Usage:
    python scripts/convert_pred_gold_to_overlay.py \
        --input log/ckpt/REC/CUSTOM-Transformer-udc-0_t04/pred_gold.jsonl \
        --output-dir /tmp/overlay_t04
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=Path, required=True,
                    help='pred_gold.jsonl from UDCHealth inference')
    ap.add_argument('--output-dir', type=Path, required=True,
                    help='directory to write eval_overlay_*.json files')
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.input) as f:
        lines = [json.loads(l) for l in f if l.strip()]

    for idx, record in enumerate(lines):
        out = {
            'idx': idx,
            'subject_id': 'udc_p{}'.format(idx),
            'method': 'udchealth',
            'evaluated_pred_codes': record['pred'],
            'gold_codes': record['gold'],
        }
        out_path = args.output_dir / 'eval_overlay_{:04d}.json'.format(idx)
        with open(out_path, 'w') as f:
            json.dump(out, f)

    print('Converted {} records to {}'.format(len(lines), args.output_dir))


if __name__ == '__main__':
    main()
