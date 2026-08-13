#!/usr/bin/env python3
"""Run one model across several devices and backends, then compare.

Answers two questions that a single-device benchmark cannot:

1. **How fast** is this model on each platform (latency, inferences/sec)?
2. **Do the platforms agree on the answer?** A model compiled for the Edge TPU
   can return nonsense while reporting excellent latency. Nothing else in this
   project checks that, so a comparison table can look great and be wrong.

Each target runs the same model with the same input seed, and reports an
output signature (top-k indices, dequantised checksum, L2 norm). Targets are
compared against a reference — by default the first one.

Targets are `name=host:backend[@model]`, where an empty host means this
machine and `@model` overrides the model for that target. The override is
required for Edge TPU, which needs its own compiled `*_edgetpu.tflite` build —
comparing it against the CPU build of the same network is the whole point:

    python scripts/platform_matrix.py \\
        --model data/models/mobilenetv1_int8_ptq_Fuzzy.tflite \\
        --target "x86=:cpu" \\
        --target "rpi-cpu=rpi:cpu" \\
        --target "rpi-tpu=rpi:edgetpu@~/models/mobilenetv1_int8_ptq_Fuzzy_edgetpu.tflite"

Remote targets need SSH access and a TFLite runtime on the far side. The
agent sources are copied to a temporary directory there and removed after.

This is a comparison harness, not a replacement for agent/benchmark_full.py:
keep the iteration count high if you intend to cite the numbers.
"""

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REMOTE_DIR = '/tmp/edge-bench-matrix'


class TargetError(RuntimeError):
    """A target could not be measured."""


def parse_target(spec: str) -> dict:
    """Parse `name=host:backend[@model]` into its parts."""
    if '=' not in spec:
        raise argparse.ArgumentTypeError(
            f'Target must look like name=host:backend[@model], got {spec!r}'
        )
    name, _, location = spec.partition('=')
    location, _, model_override = location.partition('@')
    host, _, backend = location.partition(':')
    if not name or not backend:
        raise argparse.ArgumentTypeError(
            f'Target must look like name=host:backend[@model], got {spec!r}'
        )
    return {
        'name': name,
        'host': host,
        'backend': backend,
        'model': model_override or None,
    }


def run_local(model: Path, backend: str, args) -> dict:
    cmd = [
        sys.executable,
        str(ROOT / 'scripts' / 'benchmark_smoke.py'),
        '--model',
        str(model),
        '--backend',
        backend,
        '--runs',
        str(args.runs),
        '--warmup',
        str(args.warmup),
        '--threads',
        str(args.threads),
        '--seed',
        str(args.seed),
        '--json-stdout',
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise TargetError(
            proc.stderr.strip().splitlines()[-1] if proc.stderr else 'failed'
        )
    return json.loads(proc.stdout)


def run_remote(host: str, model: str, backend: str, args) -> dict:
    """Copy the agent to the host, run one benchmark, return the result."""
    # benchmark_smoke.py resolves `agent/` relative to its parent's parent,
    # so the remote layout must mirror the repository: <dir>/scripts/<script>
    # next to <dir>/agent/.
    subprocess.run(
        ['ssh', '-o', 'BatchMode=yes', host, f'mkdir -p {REMOTE_DIR}/scripts'],
        check=True,
        capture_output=True,
    )
    # Only what the benchmark needs; no .git, no data, no results.
    rsync = subprocess.run(
        [
            'rsync',
            '-az',
            '--delete',
            '--exclude',
            '__pycache__',
            str(ROOT / 'agent') + '/',
            f'{host}:{REMOTE_DIR}/agent/',
        ],
        capture_output=True,
        text=True,
    )
    if rsync.returncode != 0:
        raise TargetError(f'rsync failed: {rsync.stderr.strip()}')
    subprocess.run(
        [
            'scp',
            '-q',
            str(ROOT / 'scripts' / 'benchmark_smoke.py'),
            f'{host}:{REMOTE_DIR}/scripts/benchmark_smoke.py',
        ],
        check=True,
        capture_output=True,
    )

    remote_cmd = (
        f'cd {REMOTE_DIR} && '
        f'{args.remote_python} scripts/benchmark_smoke.py '
        f'--model {shlex.quote(model)} --backend {backend} '
        f'--runs {args.runs} --warmup {args.warmup} '
        f'--threads {args.threads} --seed {args.seed} --json-stdout'
    )
    proc = subprocess.run(
        ['ssh', '-o', 'BatchMode=yes', host, remote_cmd],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        tail = (proc.stderr or proc.stdout).strip().splitlines()
        raise TargetError(tail[-1] if tail else f'exit {proc.returncode}')
    return json.loads(proc.stdout)


def cleanup_remote(host: str) -> None:
    subprocess.run(
        ['ssh', '-o', 'BatchMode=yes', host, f'rm -rf {REMOTE_DIR}'],
        capture_output=True,
    )


def compare_outputs(reference: dict, other: dict) -> dict:
    """Compare two output signatures.

    `agrees` is the honest verdict: identical top-1 AND identical checksum
    means bit-exact. Matching top-1 with a different checksum is normal when
    comparing an INT8 CPU model against its Edge TPU build — the verdict is
    the same, the arithmetic is not.
    """
    ref, oth = reference.get('output'), other.get('output')
    if not ref or not oth or 'error' in ref or 'error' in oth:
        return {'comparable': False, 'reason': 'no output signature'}

    ref_top = ref.get('top_k_indices') or []
    oth_top = oth.get('top_k_indices') or []
    top1_match = bool(ref_top and oth_top and ref_top[0] == oth_top[0])
    topk_match = ref_top == oth_top
    checksum_match = ref.get('checksum') == oth.get('checksum')

    ref_norm, oth_norm = ref.get('l2_norm') or 0.0, oth.get('l2_norm') or 0.0
    norm_delta_pct = abs(oth_norm - ref_norm) / ref_norm * 100 if ref_norm else None

    if checksum_match:
        verdict = 'identical'
    elif topk_match:
        verdict = 'same top-5, different arithmetic'
    elif top1_match:
        verdict = 'same top-1, different ranking'
    else:
        verdict = 'DISAGREES'

    return {
        'comparable': True,
        'verdict': verdict,
        'top1_match': top1_match,
        'topk_match': topk_match,
        'checksum_match': checksum_match,
        'l2_norm_delta_pct': round(norm_delta_pct, 3)
        if norm_delta_pct is not None
        else None,
    }


def summarise(name: str, result: dict) -> dict:
    latency = result.get('latency', {})
    throughput = result.get('throughput', {})
    device = result.get('device_info', {})
    runtime = result.get('runtime', {})
    return {
        'target': name,
        'device': device.get('device_model') or device.get('hostname'),
        'arch': device.get('platform', '').split('-')[-1] or None,
        'runtime': f'{runtime.get("tflite_source")} {runtime.get("tflite_version")}',
        'backend': result.get('params', {}).get('backend'),
        'mean_ms': latency.get('mean_ms'),
        'p50_ms': latency.get('p50_ms'),
        'p95_ms': latency.get('p95_ms'),
        'std_ms': latency.get('std_ms'),
        'fps': throughput.get('fps_from_mean'),
        'rss_mb': result.get('system', {}).get('process_rss_mb_max'),
        'model_load_ms': result.get('cold_start', {}).get('model_load_ms'),
        'warnings': result.get('warnings', []),
    }


def print_table(rows: list[dict], comparisons: dict, reference: str) -> None:
    def cell(v, digits=2):
        return '—' if v is None else f'{v:.{digits}f}'

    print()
    print('=== Performance ===')
    header = f'{"target":<10} {"device":<26} {"backend":<9} {"mean ms":>9} {"p95 ms":>9} {"fps":>9} {"RSS MB":>8}'
    print(header)
    print('-' * len(header))
    fastest = min((r['mean_ms'] for r in rows if r['mean_ms']), default=None)
    for r in rows:
        marker = ' *' if fastest and r['mean_ms'] == fastest else '  '
        print(
            f'{r["target"]:<10} {str(r["device"])[:26]:<26} {str(r["backend"]):<9} '
            f'{cell(r["mean_ms"], 3):>9} {cell(r["p95_ms"], 3):>9} '
            f'{cell(r["fps"], 1):>9} {cell(r["rss_mb"], 1):>8}{marker}'
        )
    if fastest:
        print('\n* fastest')

    print()
    print(f'=== Output agreement (reference: {reference}) ===')
    for name, cmp_result in comparisons.items():
        if not cmp_result.get('comparable'):
            print(f'  {name:<10} not comparable: {cmp_result.get("reason")}')
            continue
        flag = 'OK  ' if cmp_result['top1_match'] else 'FAIL'
        extra = ''
        if cmp_result['l2_norm_delta_pct'] is not None:
            extra = f', |L2| delta {cmp_result["l2_norm_delta_pct"]:.2f}%'
        print(f'  {flag} {name:<10} {cmp_result["verdict"]}{extra}')

    for r in rows:
        for w in r['warnings']:
            print(f'\n  warning [{r["target"]}]: {w}')


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--model', required=True, help='Local path to the .tflite model'
    )
    parser.add_argument(
        '--remote-model',
        help='Path to the same model on remote hosts (default: same basename under ~/models)',
    )
    parser.add_argument(
        '--target',
        action='append',
        required=True,
        type=parse_target,
        help='Repeatable. Format: name=host:backend (empty host = this machine)',
    )
    parser.add_argument('--runs', type=int, default=50)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--remote-python',
        default='~/edge-bench-agent/venv/bin/python',
        help='Python interpreter on remote hosts',
    )
    parser.add_argument(
        '--output-dir',
        default='results/platform_matrix',
        help='Where to write the comparison document',
    )
    parser.add_argument('--no-write', action='store_true')
    parser.add_argument(
        '--keep-remote',
        action='store_true',
        help='Do not delete the temporary directory on remote hosts',
    )
    args = parser.parse_args()

    model = Path(args.model).expanduser().resolve()
    if not model.is_file():
        print(f'Model not found: {model}', file=sys.stderr)
        return 2
    remote_model = args.remote_model or f'~/models/{model.name}'

    results: dict[str, dict] = {}
    failures: dict[str, str] = {}
    hosts_used = set()

    for target in args.target:
        label = f'{target["name"]} ({target["host"] or "local"}:{target["backend"]})'
        print(f'Running {label} ...', file=sys.stderr)
        try:
            if target['host']:
                hosts_used.add(target['host'])
                results[target['name']] = run_remote(
                    target['host'],
                    target['model'] or remote_model,
                    target['backend'],
                    args,
                )
            else:
                local_model = (
                    Path(target['model']).expanduser() if target['model'] else model
                )
                results[target['name']] = run_local(
                    local_model, target['backend'], args
                )
            print(f'  done {label}', file=sys.stderr)
        except (
            TargetError,
            json.JSONDecodeError,
            subprocess.CalledProcessError,
        ) as exc:
            failures[target['name']] = str(exc)
            print(f'  FAILED {label}: {exc}', file=sys.stderr)

    if not args.keep_remote:
        for host in hosts_used:
            cleanup_remote(host)

    if not results:
        print('No target produced a result.', file=sys.stderr)
        return 1

    reference = next(iter(results))
    rows = [summarise(name, res) for name, res in results.items()]
    comparisons = {
        name: compare_outputs(results[reference], res)
        for name, res in results.items()
        if name != reference
    }

    print_table(rows, comparisons, reference)

    if failures:
        print('\n=== Failed targets ===')
        for name, err in failures.items():
            print(f'  {name}: {err}')

    document = {
        'generated_at': datetime.now(UTC).isoformat(),
        'model': {
            'name': model.name,
            'local_path': str(model),
            'hash': next(iter(results.values())).get('model', {}).get('hash'),
        },
        'parameters': {
            'runs': args.runs,
            'warmup': args.warmup,
            'threads': args.threads,
            'input_seed': args.seed,
        },
        'reference_target': reference,
        'summary': rows,
        'output_agreement': comparisons,
        'failures': failures,
        'raw_results': results,
    }

    if not args.no_write:
        out_dir = Path(args.output_dir)
        if not out_dir.is_absolute():
            out_dir = ROOT / out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime('%Y-%m-%d_%H%M%S')
        out_path = out_dir / f'{stamp}_{model.stem}.json'
        out_path.write_text(json.dumps(document, indent=2, default=str))
        print(f'\nComparison written to {out_path}')

    # A disagreeing backend is a failure, not a footnote.
    disagreed = [
        n
        for n, c in comparisons.items()
        if c.get('comparable') and not c.get('top1_match')
    ]
    if disagreed:
        print(f'\nOutput disagreement on: {", ".join(disagreed)}', file=sys.stderr)
        return 1
    if failures:
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
