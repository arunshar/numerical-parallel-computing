"""Compile and run four small native examples in a temporary directory."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import tempfile


def capture(command, timeout=40):
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    return {'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Save the verification JSON at this path.')
    args = parser.parse_args()
    source_dir = Path(__file__).resolve().parent
    cc = os.environ.get('CC', 'clang' if platform.system() == 'Darwin' else 'cc')
    mpicc = os.environ.get('MPICC', 'mpicc')
    launcher = os.environ.get('MPIEXEC', 'mpiexec')
    versions = {tool: capture([tool, '--version']) for tool in dict.fromkeys([cc, mpicc, launcher])}
    libomp = Path('/opt/homebrew/opt/libomp')
    omp_flags = ['-fopenmp']
    if platform.system() == 'Darwin' and libomp.exists() and 'clang' in cc:
        omp_flags = ['-Xpreprocessor', '-fopenmp', f'-I{libomp}/include',
                     f'-L{libomp}/lib', f'-Wl,-rpath,{libomp}/lib', '-lomp']
    if shutil.which('brew'):
        versions['homebrew_openmp_mpi'] = capture(['brew', 'list', '--versions', 'libomp', 'open-mpi'])
    mpi_flags = ['--oversubscribe'] if 'Open MPI' in versions[launcher]['stdout'] else []
    flags = ['-std=c11', '-O2', '-Wall', '-Wextra']
    cases = [('pthread_sum', [cc] + flags + ['-pthread'], []),
             ('openmp_sum', [cc] + flags + omp_flags, []),
             ('mpi_exchange', [mpicc] + flags, [launcher] + mpi_flags + ['-n', '4']),
             ('mpi_collectives', [mpicc] + flags, [launcher] + mpi_flags + ['-n', '4'])]
    records = []
    with tempfile.TemporaryDirectory(prefix='parallel-native-') as temporary:
        for name, compiler, prefix in cases:
            source = source_dir / f'{name}.c'
            executable = Path(temporary) / name
            compiled = capture(compiler + [str(source), '-o', str(executable)])
            row = {'example': name, 'source': f'labs/parallel/{name}.c',
                   'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                   'compile_command': compiler + [f'labs/parallel/{name}.c', '-o', f'<temporary>/{name}'],
                   'compile_returncode': compiled['returncode'], 'compile_stderr': compiled['stderr']}
            if compiled['returncode'] == 0:
                ran = capture(prefix + [str(executable)], timeout=30)
                row.update(run_command=prefix + [f'<temporary>/{name}'],
                           run_returncode=ran['returncode'], stdout=ran['stdout'], stderr=ran['stderr'])
            records.append(row)
    report = {'platform': platform.platform(), 'machine': platform.machine(),
              'tool_versions': versions, 'max_processes_or_threads': 4,
              'cases': records, 'passed': all(r.get('run_returncode') == 0 for r in records),
              'scope': 'Local native correctness checks only. No timing or speedup measurements.'}
    encoded = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + '\n')
    print(encoded)
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
