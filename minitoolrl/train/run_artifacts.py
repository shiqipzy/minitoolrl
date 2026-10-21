"""运行输入与交付产物：输入包打包、官方底座获取、回传清单。

这里只处理文件，不训练、不评测：pack 只读冻结输入生成上传包，fetch_base 在运行
目录建立唯一底座，package 列出可回传的文件与权重。已完成的运行产物不改写。
"""
from __future__ import annotations

import json
import shutil
import tarfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from minitoolrl import files
from minitoolrl.data.input_bundle import (CURRENT_MANIFEST_SHA256, DEFAULT_ROOT, MANIFEST,
                                          MARKER, REQUIRED_FILES, verify_bundle)
from minitoolrl.train.experiment_config import (BASE_REPO, BASE_REVISION, BASE_SHA256,
                                                BASE_TEMPLATE_SHA256)
from minitoolrl.train.run_session import LaunchError, checkpoint_template_sha256


def installed_torch_version():
    """计划与打包不加载训练运行时，只记录已安装版本（没有则为 None）。"""
    try:
        return version('torch')
    except PackageNotFoundError:
        return None


def source_files(package: Path):
    """只收集当前模块，不把数据、旧模型或残留缓存当成源码。"""
    return sorted([*package.glob('*.py'),
                   *(p for name in ('data', 'env', 'train', 'eval')
                     for p in (package / name).glob('*.py'))])


def verify_base_template(checkpoint: Path):
    """底座权重与 tokenizer 模板必须来自同一个固定 Qwen revision。"""
    if checkpoint_template_sha256(checkpoint) != BASE_TEMPLATE_SHA256:
        raise LaunchError('base tokenizer must use the pinned official Qwen3 chat template')


def stage_pack(S):
    """本地步骤：把冻结输入与当前源码打成上传包，不创建训练运行。"""
    input_root = Path(S.args.inputs_root) if S.args.inputs_root else DEFAULT_ROOT
    verify_bundle(input_root, expected_manifest_sha256=CURRENT_MANIFEST_SHA256)
    out = Path(S.args.pack_out or (S.repo / 'dist' / 'current' / 'full_chain_inputs.tar.gz'))
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest = {}
    bundled = files.load_json(input_root / MANIFEST)['files']
    extras = [MARKER, MANIFEST, 'validation.json']
    with tarfile.open(out, 'w:gz') as tar:
        for rel in sorted(REQUIRED_FILES):
            src = input_root / rel
            tar.add(str(src), arcname=rel)
            manifest[rel] = dict(source=str(src), bytes=src.stat().st_size,
                                 sha256=bundled[rel]['sha256'])
        for rel in extras:
            tar.add(str(input_root / rel), arcname=rel)
    source_archive = out.parent / 'minitoolrl_source.tar.gz'
    package = S.repo / 'minitoolrl'
    sources = [*source_files(package), *package.joinpath('docs').glob('*.md'), package / 'README.md',
               *(S.repo / name for name in ('README.md', 'AGENTS.md', 'pyproject.toml', 'LICENSE',
                                            'NOTICE', 'requirements.txt', 'requirements-gpu.txt'))]
    with tarfile.open(source_archive, 'w:gz') as tar:
        for path in sorted(sources):
            tar.add(path, arcname=str(path.relative_to(S.repo)))
    files.dump_json(out.with_suffix('.manifest.json'),
                    dict(run_id=S.run_id, inputs_manifest_sha256=CURRENT_MANIFEST_SHA256,
                         tool_order='registry', files=manifest, created_local=files.now(),
                         archives={p.name: files.sha256_file(p) for p in (out, source_archive)}))
    print(json.dumps(dict(tarball=str(out), source_archive=str(source_archive),
                          files=len(manifest), bytes=out.stat().st_size), ensure_ascii=False))
    return dict(tarball=str(out), source_archive=str(source_archive), files=manifest)


def stage_fetch_base(S):
    """在运行目录建立唯一官方底座，核对权重与模板哈希后记录 pin。"""
    dest = S.run / 'models' / 'Qwen3-0.6B'
    weight = dest / 'model.safetensors'
    if S.args.base_dir:
        src = Path(S.args.base_dir)
        if src.resolve() != dest.resolve():
            shutil.copytree(src, dest, dirs_exist_ok=True)
    elif not weight.exists() or files.sha256_file(weight) != BASE_SHA256:
        # 使用公开下载 API，在 GPU 的 huggingface_hub 1.x 上同样按固定 revision 下载。
        code = ('from huggingface_hub import snapshot_download; import sys; '
                'snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3])')
        hf = [S.python, '-c', code, BASE_REPO, BASE_REVISION, str(dest)]
        S.launch('fetch_base_hf', hf, S.args.timeout or 1800, 'fetch_base.log',
                 env_extra={'HF_ENDPOINT': S.args.hf_endpoint})
    if not weight.exists():
        raise LaunchError('base_weights_not_downloaded')
    if files.sha256_file(weight) != BASE_SHA256:
        raise LaunchError('base_sha_mismatch')
    verify_base_template(dest)
    record = S.pin('base_qwen', dest, 'fetch_base', repo=BASE_REPO, revision=BASE_REVISION)
    return dict(base=record)


def stage_package(S):
    """列出可回传的运行文件与权重清单；只读运行目录，不改写产物。"""
    entries = []
    weights = []
    for path in sorted(S.run.rglob('*')):
        if not path.is_file():
            continue
        rel = path.relative_to(S.run).as_posix()
        if path.suffix in ('.safetensors', '.pt', '.pth') or '.part.' in path.name \
                or rel.startswith('models/'):
            if path.name == 'model.safetensors':
                weights.append(dict(path=rel, bytes=path.stat().st_size))
            continue
        entries.append(dict(path=rel, bytes=path.stat().st_size,
                            sha256=files.sha256_file(path)))
    manifest = dict(run_id=S.run_id, files=entries, weight_artifacts=weights, pins=S.pins,
                    total_bytes=sum(e['bytes'] for e in entries), local=files.now())
    files.dump_json(S.run / 'transfer_manifest.json', manifest)
    return dict(files=len(entries), weights=len(weights))
