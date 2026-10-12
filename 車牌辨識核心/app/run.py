"""Run the resident USB file-exchange core (no API server)."""
import argparse
from pathlib import Path
import sys
from settings import load_settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--max-jobs', type=int, default=0, help='Stop after N completed deliveries; 0 keeps running')
    args = parser.parse_args()
    if args.max_jobs < 0:
        parser.error('--max-jobs must be nonnegative')
    try:
        settings = load_settings(args.core_root)
        from backend import UsbRecognizer
        from service import FileCore
        return FileCore(settings, UsbRecognizer, args.max_jobs).run()
    except Exception as exc:
        print(f'[START_FAILED] {exc}', file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
