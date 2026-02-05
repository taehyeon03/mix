#!/usr/bin/env python
"""
머지된 SmolVLA 모델 검증 스크립트

사용 예시:
1. 로컬 디렉토리에서 검증:
   python verify_merged_smolvla.py --path outputs/merged_ta_tallmask

2. Hugging Face Hub에서 검증:
   python verify_merged_smolvla.py --repo_id KTH03/merged-smolvla_ta

3. 원본 모델과 비교:
   python verify_merged_smolvla.py --repo_id KTH03/merged-smolvla_ta \
     --compare_with lerobotForScienceEdu/flametest-v1-smolVLA,lerobotForScienceEdu/Rock_Acid-v4-130-merged-smolVLA
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

import torch
from tqdm import tqdm

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

logger = logging.getLogger(__name__)


def verify_model_structure(policy: SmolVLAPolicy) -> dict:
    """모델 구조와 파라미터 통계를 확인."""
    stats = {
        "total_params": 0,
        "trainable_params": 0,
        "param_shapes": {},
        "param_norms": {},
    }

    for name, param in policy.named_parameters():
        num_params = param.numel()
        stats["total_params"] += num_params
        if param.requires_grad:
            stats["trainable_params"] += num_params

        # 주요 레이어만 샘플링해서 저장
        if any(key in name for key in ["vision_backbone", "language_model", "action_head", "projector"]):
            stats["param_shapes"][name] = list(param.shape)
            stats["param_norms"][name] = param.norm().item()

    return stats


def compare_models(
    merged_policy: SmolVLAPolicy,
    original_repos: List[str],
    sample_keys: Optional[List[str]] = None,
) -> dict:
    """머지된 모델과 원본 모델들의 파라미터를 비교."""
    print(f"원본 모델 {len(original_repos)}개 로딩 중...", flush=True)
    original_policies = []
    for i, repo in enumerate(original_repos, 1):
        print(f"  [{i}/{len(original_repos)}] {repo} 로딩 중...", flush=True)
        original_policies.append(SmolVLAPolicy.from_pretrained(repo))
    print("원본 모델 로딩 완료.", flush=True)
    
    merged_sd = merged_policy.state_dict()

    comparison = {}
    all_keys = set(merged_sd.keys())
    for repo, orig_policy in zip(original_repos, original_policies):
        orig_sd = orig_policy.state_dict()
        common_keys = all_keys & set(orig_sd.keys())

        if sample_keys:
            keys_to_check = [k for k in sample_keys if k in common_keys]
        else:
            # trainable 파라미터 중 일부만 샘플링
            trainable_keys = {
                name
                for name, param in merged_policy.named_parameters()
                if param.requires_grad
            }
            keys_to_check = sorted(list(trainable_keys & common_keys))[:100]  # 상위 10개만

        print(f"  비교할 파라미터 키 개수: {len(keys_to_check)}", flush=True)
        if not keys_to_check:
            print(f"  ⚠️  {repo}와 비교할 공통 trainable 파라미터가 없습니다.", flush=True)
            comparison[repo] = {}
            continue
            
        diffs = {}
        for key in tqdm(keys_to_check, desc=f"  {repo} 비교 중", leave=False):
            merged_val = merged_sd[key]
            orig_val = orig_sd[key]
            if merged_val.shape == orig_val.shape:
                diff = (merged_val - orig_val).abs().max().item()
                diff_norm = (merged_val - orig_val).norm().item()
                merged_norm = merged_val.norm().item()
                orig_norm = orig_val.norm().item()
                diffs[key] = {
                    "max_diff": diff,
                    "diff_norm": diff_norm,
                    "merged_norm": merged_norm,
                    "orig_norm": orig_norm,
                    "relative_diff": diff_norm / (orig_norm + 1e-8),
                }
        comparison[repo] = diffs

    return comparison


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Verify merged SmolVLA model")
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="로컬 디렉토리 경로 (예: outputs/merged_ta_tallmask)",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default=None,
        help="Hugging Face Hub repo ID (예: KTH03/merged-smolvla_ta)",
    )
    parser.add_argument(
        "--compare_with",
        type=str,
        default=None,
        help="원본 모델 repo ID들을 콤마로 구분 (비교용)",
    )
    parser.add_argument(
        "--sample_keys",
        type=str,
        default=None,
        help="비교할 특정 파라미터 키들을 콤마로 구분 (선택사항)",
    )

    args = parser.parse_args()

    if not args.path and not args.repo_id:
        raise ValueError("--path 또는 --repo_id 중 하나는 반드시 지정해야 합니다.")

    # 모델 로드
    print("=" * 60, flush=True)
    print("머지된 모델 로딩 중...", flush=True)
    print("=" * 60, flush=True)
    try:
        if args.path:
            merged_policy = SmolVLAPolicy.from_pretrained(args.path)
            print(f"✓ 로컬 경로에서 로드 완료: {args.path}", flush=True)
        else:
            merged_policy = SmolVLAPolicy.from_pretrained(args.repo_id)
            print(f"✓ Hugging Face Hub에서 로드 완료: {args.repo_id}", flush=True)
        merged_policy.eval()
        print("모델 로딩 완료.\n", flush=True)
    except Exception as e:
        print(f"ERROR: 모델 로딩 실패: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise

    # 1. 모델 구조 검증
    print("\n" + "=" * 60, flush=True)
    print("1. 모델 구조 및 파라미터 통계", flush=True)
    print("=" * 60, flush=True)
    stats = verify_model_structure(merged_policy)
    print(f"총 파라미터 수: {stats['total_params']:,}", flush=True)
    print(f"학습 가능 파라미터 수: {stats['trainable_params']:,}", flush=True)
    print(f"학습 가능 비율: {stats['trainable_params']/stats['total_params']*100:.2f}%", flush=True)

    print("\n주요 레이어 파라미터 샘플:", flush=True)
    for name, shape in list(stats["param_shapes"].items())[:5]:
        norm = stats["param_norms"].get(name, 0)
        print(f"  {name}: shape={shape}, norm={norm:.6e}", flush=True)

    # 2. 원본 모델과 비교 (옵션)
    if args.compare_with:
        print("\n" + "=" * 60, flush=True)
        print("2. 원본 모델과 파라미터 비교", flush=True)
        print("=" * 60, flush=True)
        original_repos = [r.strip() for r in args.compare_with.split(",") if r.strip()]
        sample_keys_list = (
            [k.strip() for k in args.sample_keys.split(",") if k.strip()]
            if args.sample_keys
            else None
        )

        print("원본 모델과 비교 중...", flush=True)
        try:
            comparison = compare_models(merged_policy, original_repos, sample_keys_list)
            print("비교 완료.\n", flush=True)
        except Exception as e:
            print(f"ERROR: 비교 중 오류 발생: {e}", flush=True)
            import traceback
            traceback.print_exc()
            raise

        for repo, diffs in comparison.items():
            print(f"\n원본 모델: {repo}", flush=True)
            print(f"비교한 파라미터 개수: {len(diffs)}", flush=True)
            if diffs:
                print("\n상위 5개 파라미터 비교:", flush=True)
                sorted_diffs = sorted(
                    diffs.items(), key=lambda x: x[1]["relative_diff"], reverse=True
                )[:5]
                for key, diff_info in sorted_diffs:
                    print(
                        f"  {key}:"
                        f" max_diff={diff_info['max_diff']:.6e},"
                        f" relative_diff={diff_info['relative_diff']:.4f},"
                        f" merged_norm={diff_info['merged_norm']:.6e},"
                        f" orig_norm={diff_info['orig_norm']:.6e}",
                        flush=True
                    )

    # 3. 간단한 forward 테스트 (선택사항)
    print("\n" + "=" * 60, flush=True)
    print("3. Forward pass 테스트", flush=True)
    print("=" * 60, flush=True)
    try:
        # 더미 입력으로 forward 테스트
        dummy_image = torch.zeros(1, 3, 224, 224)
        dummy_text = ["test instruction"]
        with torch.no_grad():
            # 실제 forward 시그니처에 맞게 조정 필요 (SmolVLA의 실제 forward 확인 필요)
            print("Forward pass 테스트는 SmolVLA의 실제 forward 시그니처에 맞게 구현 필요", flush=True)
            print("모델 구조는 정상적으로 로드되었습니다.", flush=True)
    except Exception as e:
        print(f"WARNING: Forward pass 테스트 중 오류 (무시 가능): {e}", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("검증 완료!", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
