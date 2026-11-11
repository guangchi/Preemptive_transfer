#!/usr/bin/env python3
"""Check prerequisites without launching model servers or an experiment."""
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig

ROOT = Path(__file__).resolve().parents[1]


def main():
    errors = []
    if sys.version_info[:2] != (3, 11):
        errors.append('The archived experiment used Python 3.11.')
    for package, expected in [('vllm', '0.32.0'), ('torch', '2.13.0'),
                              ('nixl', '1.5.2'), ('nixl-cu13', '1.5.2'),
                              ('flashinfer-python', '0.7.0.post1')]:
        try:
            actual = metadata.version(package)
        except metadata.PackageNotFoundError:
            actual = 'missing'
        print(f'{package}: {actual}')
        if actual.split('+')[0] != expected:
            errors.append(f'Requires {package}=={expected}.')
    result = subprocess.run([sys.executable, str(ROOT/'scripts/restore_vllm.py'), '--check'])
    if result.returncode:
        errors.append('vLLM modifications are not fully installed.')
    model = Path(os.environ.get('MODEL', str(ROOT.parent/'models/Llama-3.1-8B-Instruct')))
    if not (model/'config.json').is_file() or not list(model.glob('*.safetensors')):
        errors.append(f'Set MODEL to the complete local model directory: {model}')
    elif 'LlamaForCausalLM' not in json.loads((model/'config.json').read_text()).get('architectures', []):
        errors.append('The recorded source-fence path was validated with LlamaForCausalLM.')
    vllm_bin = Path(os.environ.get('VLLM_BIN', str(Path(sys.executable).parent/'vllm')))
    if not vllm_bin.is_file() or not os.access(vllm_bin, os.X_OK):
        errors.append(f'vLLM executable is missing: {vllm_bin}')
    toolkit = Path(os.environ.get('CUDA_TOOLKIT_ROOT', str(Path(sysconfig.get_path('purelib'))/'nvidia/cu13')))
    nvcc = toolkit/'bin/nvcc'
    if not nvcc.is_file():
        errors.append(f'CUDA compiler missing; set CUDA_TOOLKIT_ROOT: {nvcc}')
    print(f'Model: {model}\nCUDA toolkit: {toolkit}')
    try:
        from nixl._api import nixl_agent
        import torch
        print(f'Torch CUDA: {torch.version.cuda}; visible GPUs: {torch.cuda.device_count()}')
        if not torch.version.cuda or not torch.version.cuda.startswith('13.'):
            errors.append('The archived experiment requires the CUDA 13 Torch build.')
        p_device = int(os.environ.get('P_DEVICE', '0'))
        d_device = int(os.environ.get('D_DEVICE', '1'))
        if p_device == d_device or not all(0 <= i < torch.cuda.device_count() for i in (p_device, d_device)):
            errors.append('P and D require two distinct visible GPUs (defaults: 0 and 1).')
        else:
            for i in (p_device, d_device):
                print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        errors.append(f'GPU/NIXL check failed: {error}')
    if errors:
        raise SystemExit('\n'.join('FAIL: ' + error for error in errors))
    print('Prerequisites passed. Model startup and JIT compilation still require a real run.')


if __name__ == '__main__':
    main()
