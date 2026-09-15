#!/usr/bin/env python3
"""Generate all AHA workflow artifacts using only this publication version."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from aha_publish.commands import entry, parser, run_stage, selected_tasks


def main():
    p = parser('Generate TTM context, evidence, descriptions, BTs, calibration, runs, and scores.', tasks=True)
    p.add_argument('--provider', choices=['openai', 'claude'], default='openai',
                   help='Provider for descriptions and BT generation; runtime VLM uses OpenAI.')
    p.add_argument('--generation-model', help='Model for descriptions and BT generation.')
    p.add_argument('--verification-model', help='OpenAI model for runtime verification.')
    p.add_argument('--episodes', type=int, default=10, help='Clean calibration episodes per task.')
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--failures', default='all', help='all, none, or comma-separated failure types.')
    p.add_argument('--checks', choices=['full', 'detectors'], default='full')
    args = p.parse_args()
    if args.episodes < 1 or args.workers < 1:
        p.error('--episodes and --workers must be positive')

    selection = [value for task in selected_tasks(args) for value in ('--task', task)]
    generation = ['--provider', args.provider]
    if args.generation_model:
        generation += ['--model', args.generation_model]

    run_stage('01_ttm_context', selection, args.dry_run)
    run_stage('02_descriptions', [*selection, *generation], args.dry_run)
    run_stage('03_behavior_trees', [*selection, *generation,
                                  '--description-provider', args.provider], args.dry_run)
    run_stage('calibration', ['all', *selection, '--episodes', args.episodes,
                             '--workers', args.workers, '--force'], args.dry_run)
    running = [*selection, '--failures', args.failures, '--checks', args.checks,
               '--workers', args.workers, '--redo']
    if args.verification_model:
        running += ['--model', args.verification_model]
    run_stage('04_running', running, args.dry_run)
    methods = ['A2'] if args.checks == 'detectors' else ['A1', 'A2', 'A3', 'A4']
    run_stage('05_scoring', [*selection, '--methods', *methods], args.dry_run)


if __name__ == '__main__':
    entry(main)
