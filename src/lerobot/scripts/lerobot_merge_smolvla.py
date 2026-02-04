#!/usr/bin/env python
"""
SmolVLA 모델 병합 스크립트

두 개의 SmolVLA 정책을 다양한 알고리즘으로 병합합니다.
지원 알고리즘: weighted_average, TATallMask, ties_TallMask

사용 예시:

1. Weighted Average (가중 평균):
   python lerobot_merge_smolvla.py merge \
     --repos user/model1,user/model2 \
     --algo_name weighted_average \
     --weight 0.5 \
     --output_dir outputs/merged_wa \
     --debug

2. TallMask Task Arithmetic:
   python lerobot_merge_smolvla.py merge \
     --repos user/model1,user/model2 \
     --algo_name TATallMask \
     --tall_mask_lambda 0.6 \
     --output_dir outputs/merged_ta_tallmask \
     --debug

3. TallMask Ties Merging:
   python lerobot_merge_smolvla.py merge \
     --repos user/model1,user/model2 \
     --algo_name ties_TallMask \
     --tall_mask_lambda 0.6 \
     --output_dir outputs/merged_ties_tallmask \
     --debug

4. 병합 후 Hugging Face Hub에 업로드 (항상 public):
   python lerobot_merge_smolvla.py merge \
     --repos user/model1,user/model2 \
     --algo_name weighted_average \
     --weight 0.5 \
     --output_dir outputs/merged_wa \
     --push_to_hub \
     --repo_id user/merged-smolvla
"""

import argparse
import logging
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

from huggingface_hub import HfApi, hf_hub_download
import torch
from torch import Tensor, nn
from tqdm import tqdm

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


logger = logging.getLogger(__name__)


@dataclass
class MergeSmolVLAConfig:
    """머지 스크립트 기본 설정 모음."""

    default_algo_name: str = "weighted_average"
    default_weight: float = 0.5
    debug_sample_params: int = 5
    tall_mask_lambda: float = 0.6  # TallMask 계열에서 사용하는 lambda


CFG = MergeSmolVLAConfig()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge two SmolVLA policies (trainable params only) and optionally push to Hugging Face Hub."
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    merge_parser = subparsers.add_parser("merge", help="Merge two SmolVLA models.")
    merge_parser.add_argument(
        "--repos",
        type=str,
        required=True,
        help="두 개의 허깅페이스 모델 repo id를 콤마로 구분해서 입력 (예: user/model1,user/model2).",
    )
    merge_parser.add_argument(
        "--algo_name",
        type=str,
        default=CFG.default_algo_name,
        help="머지 알고리즘 이름 (현재 weighted_average만 지원).",
    )
    merge_parser.add_argument(
        "--weight",
        type=float,
        default=CFG.default_weight,
        help="첫 번째 모델 가중치 w (두 번째는 1-w). 기본값 0.5. weighted_average 알고리즘에서만 사용.",
    )
    merge_parser.add_argument(
        "--tall_mask_lambda",
        type=float,
        default=CFG.tall_mask_lambda,
        help="TallMask 알고리즘에서 사용하는 lambda 하이퍼파라미터. 기본값 0.6. TATallMask, ties_TallMask에서 사용.",
    )
    merge_parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="머지된 SmolVLA 정책을 저장할 로컬 경로.",
    )
    def str_to_bool(v):
        """--push_to_hub true/false만 받음. 플래그만 주면 True."""
        if isinstance(v, bool):
            return v
        if v is None:
            return True  # 플래그만 주면 True
        v_lower = v.lower()
        if v_lower == "true":
            return True
        elif v_lower == "false":
            return False
        else:
            raise argparse.ArgumentTypeError(f"--push_to_hub는 'true' 또는 'false'만 받습니다. 받은 값: {v}")

    merge_parser.add_argument(
        "--push_to_hub",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=False,
        help="허깅페이스 허브에 업로드 여부. --push_to_hub 또는 --push_to_hub true/false (기본값: false).",
    )
    merge_parser.add_argument(
        "--repo_id",
        type=str,
        default=None,
        help="업로드할 허깅페이스 repo id (예: user/merged-smolvla). 지정하지 않으면 첫 번째 모델의 repo_id를 사용하려 시도.",
    )
    merge_parser.add_argument(
        "--debug",
        action="store_true",
        help="머지 전후 일부 파라미터 노름/차이를 출력하여 실제로 가중치가 변했는지 확인.",
    )

    return parser.parse_args()


def _get_trainable_param_names(model: torch.nn.Module) -> set[str]:
    return {name for name, p in model.named_parameters() if p.requires_grad}


# ---- state_dict helper 함수들 (fusion_bench 최소 구현) ----

StateDict = Dict[str, Tensor]


def _sd_keys_intersection(dicts: Iterable[StateDict]) -> List[str]:
    keys = None
    for sd in dicts:
        k = set(sd.keys())
        keys = k if keys is None else keys & k
    return sorted(keys or [])


def _state_dict_sub(a: StateDict, b: StateDict, exclude_keys: List[str] | None = None) -> StateDict:
    ek = set(exclude_keys or [])
    out: StateDict = {}
    for k in a.keys():
        if k in ek:
            continue
        out[k] = a[k] - b[k]
    return out


def _state_dict_add(a: StateDict, b: StateDict, exclude_keys: List[str] | None = None) -> StateDict:
    """a + b, exclude_keys는 a의 값을 그대로 사용. b에 없는 키는 a만 사용."""
    ek = set(exclude_keys or [])
    out: StateDict = {}
    for k in a.keys():
        if k in ek:
            out[k] = a[k]
        elif k in b:
            out[k] = a[k] + b[k]
        else:
            # b에 없는 키는 a만 사용 (exclude_keys가 아니면 경고)
            out[k] = a[k]
    return out


def _state_dict_sum(dicts: List[StateDict]) -> StateDict:
    if not dicts:
        return {}
    keys = dicts[0].keys()
    out: StateDict = {}
    for k in keys:
        out[k] = sum(sd[k] for sd in dicts)
    return out


def _state_dict_diff_abs(a: StateDict, b: StateDict) -> StateDict:
    out: StateDict = {}
    for k in a.keys():
        out[k] = (a[k] - b[k]).abs()
    return out


def _state_dict_binary_mask(a: StateDict, b: StateDict) -> Dict[str, Tensor]:
    """a > b 위치에서 True 인 mask."""
    out: Dict[str, Tensor] = {}
    for k in a.keys():
        out[k] = (a[k] > b[k])
    return out


def _state_dict_mul(a: StateDict, scalar: float) -> StateDict:
    out: StateDict = {}
    for k, v in a.items():
        out[k] = v * scalar
    return out


def _state_dict_hadamard(mask: Dict[str, Tensor], tv: StateDict) -> StateDict:
    out: StateDict = {}
    for k in tv.keys():
        out[k] = tv[k] * mask[k].to(tv[k].dtype)
    return out


def _generate_task_masks(
    multi_task_vector: OrderedDict,
    ft_task_vector: OrderedDict,
    tall_mask_lambda: float,
) -> OrderedDict:
    """
    MergeVLA tall_mask.utils.generate_task_masks 의 축약 구현.
    mask_t = |theta_t| > |theta_mt - theta_t| * lambda
    """
    # |theta_t|
    diff_pt_ft = OrderedDict((key, tensor.abs()) for key, tensor in ft_task_vector.items())
    # |theta_mt - theta_t|
    diff_multi_ft = _state_dict_diff_abs(multi_task_vector, ft_task_vector)
    # 비교 후 binary mask
    final_mask = _state_dict_binary_mask(
        diff_pt_ft,
        _state_dict_mul(diff_multi_ft, tall_mask_lambda),
    )
    return final_mask


# ---- TIES merging 최소 구현 (두 모델만) ----


def _state_dict_to_vector(state_dict: StateDict, remove_keys: List[str] | None = None) -> Tensor:
    shared = OrderedDict(
        (k, v)
        for k, v in sorted(state_dict.items())
        if not remove_keys or k not in remove_keys
    )
    return nn.utils.parameters_to_vector([v.reshape(-1) for v in shared.values()])


def _vector_to_state_dict(vector: Tensor, reference: StateDict, remove_keys: List[str] | None = None) -> StateDict:
    ref = OrderedDict(
        (k, v.clone())
        for k, v in sorted(reference.items())
        if not remove_keys or k not in remove_keys
    )
    nn.utils.vector_to_parameters(vector, ref.values())
    return ref


def _topk_values_mask(M: Tensor, K: float) -> Tensor:
    """MergeVLA의 topk_values_mask 요약 구현: 작은 값들 reset."""
    if K > 1:
        K = K / 100.0
    original_shape = M.shape
    if M.dim() == 1:
        M = M.unsqueeze(0)
    n, d = M.shape
    k = int(d * K)
    if k == d:
        return M.view(original_shape)
    k = d - k
    kth_values, _ = M.abs().kthvalue(k, dim=1, keepdim=True)
    mask = M.abs() >= kth_values
    final = (M * mask).view(original_shape)
    return final


def _resolve_sign(v: Tensor) -> Tensor:
    sign_to_mult = torch.sign(v.sum(dim=0))
    # majority rule: 0 은 전체 부호 합을 따른다
    majority_sign = torch.sign(v.sum())
    sign_to_mult[sign_to_mult == 0] = majority_sign
    return sign_to_mult


def _disjoint_merge(v: Tensor, merge_func: str, sign_to_mult: Tensor) -> Tensor:
    merge_func = merge_func.split("-")[-1]
    rows_to_keep = torch.where(sign_to_mult.unsqueeze(0) > 0, v > 0, v < 0)
    selected = v * rows_to_keep
    if merge_func == "mean":
        non_zero_counts = (selected != 0).sum(dim=0).float().clamp(min=1)
        return selected.sum(dim=0) / non_zero_counts
    if merge_func == "sum":
        return selected.sum(dim=0)
    if merge_func == "max":
        vals = selected.abs().max(dim=0)[0]
        return vals * sign_to_mult
    raise ValueError(f"Unknown merge_func={merge_func}")


def _ties_merging(flat_task_checks: Tensor, reset_thresh: float, merge_func: str) -> Tensor:
    updated = _topk_values_mask(flat_task_checks.clone(), K=reset_thresh)
    signs = _resolve_sign(updated)
    merged = _disjoint_merge(updated, merge_func, signs)
    return merged


# ---- 알고리즘 선택 및 병합 ----


def _merge_with_algo(
    policies: List[SmolVLAPolicy],
    algo_name: str,
    weight: float,
    tall_mask_lambda: float = 0.6,
    debug: bool = False,
) -> SmolVLAPolicy:
    """
    Merge 두 SmolVLA 정책을 fusion_bench 알고리즘(TA TallMask / ties TallMask / weighted_average)으로 병합.
    - weighted_average: fusion_bench WeightedAverageAlgorithm 사용
    - TATallMask / ties_TallMask: MergeVLA의 get_algo를 그대로 재사용
    """
    if len(policies) < 2 or len(policies) > 4:
        raise ValueError(f"policies 길이는 2~4개만 지원합니다. 받은 개수: {len(policies)}")

    # ---------------------------------------------------------
    # 1) trainable 파라미터 교집합 계산 (모든 모델 공통 trainable만 병합)
    # ---------------------------------------------------------
    trainable_sets = [_get_trainable_param_names(p) for p in policies]
    trainable_keys = set.intersection(*trainable_sets)

    sd_list = [p.state_dict() for p in policies]
    sd_pre = sd_list[0]  # baseline: 첫 번째 모델 (향후 필요 시 별도 pretrained로 확장 가능)

    all_keys = list(sd_pre.keys())
    exclude_keys = [k for k in all_keys if k not in trainable_keys]

    # ---------------------------------------------------------
    # 2) weighted_average: 현재는 2개 모델만 지원 (명확한 의미를 위해)
    # ---------------------------------------------------------
    if algo_name == "weighted_average":
        if len(policies) != 2:
            raise ValueError(
                "weighted_average 알고리즘은 현재 2개 모델 병합만 지원합니다. "
                f"받은 모델 수: {len(policies)}"
            )

        sd1, sd2 = sd_list

        w1 = float(weight)
        w2 = 1.0 - w1
        merged_sd: StateDict = {}
        debug_infos = []
        for name, t1 in tqdm(sd1.items(), desc="Merging (weighted_average)", total=len(sd1)):
            if name in trainable_keys and name in sd2:
                t2 = sd2[name]
                if t1.shape == t2.shape and t1.dtype == t2.dtype:
                    mt = w1 * t1 + w2 * t2
                    merged_sd[name] = mt
                    if debug and len(debug_infos) < CFG.debug_sample_params:
                        with torch.no_grad():
                            diff1 = (mt - t1).abs().max().item()
                            diff2 = (mt - t2).abs().max().item()
                            n1 = t1.norm().item()
                            n2 = t2.norm().item()
                            nm = mt.norm().item()
                        debug_infos.append((name, diff1, diff2, n1, n2, nm))
                else:
                    logger.warning("Skip key %s due to shape/dtype mismatch", name)
                    merged_sd[name] = t1
            else:
                merged_sd[name] = t1

        if debug and debug_infos:
            logger.info("==== Debug (WA): sample merged parameter stats ====")
            for name, diff1, diff2, n1, n2, nm in debug_infos:
                logger.info(
                    "param=%s | max|m-θ1|=%.6e, max|m-θ2|=%.6e, ||θ1||=%.6e, ||θ2||=%.6e, ||m||=%.6e",
                    name,
                    diff1,
                    diff2,
                    n1,
                    n2,
                    nm,
                )
            logger.info("===================================================")

        out = SmolVLAPolicy(policies[0].config)
        out.load_state_dict(merged_sd)
        out.to("cpu")
        return out

    # ---------------------------------------------------------
    # 3) Task vector 기반 알고리즘: TATallMask / ties_TallMask
    #    - 원 MergeVLA 스타일로 N(2~4)개 모델까지 일반화
    # ---------------------------------------------------------
    tv_list = [
        _state_dict_sub(sd, sd_pre, exclude_keys=exclude_keys)
        for sd in sd_list
    ]
    task_vectors = {f"model_{i}": tv for i, tv in enumerate(tv_list)}
    keys = _sd_keys_intersection(tv_list)

    # 3-1) TallMask Task Arithmetic (TATallMask)
    # trainable 파라미터만 merge하도록 exclude_keys 구성
    if algo_name == "TATallMask":
        # multi-task vector = sum of all task vectors (원 MergeVLA 스타일)
        multi_tv = _state_dict_sum(tv_list)

        # 각 task에 대해 TallMask를 계산하고, 해당 task vector를 마스킹한 뒤
        # 모든 masked task vector를 합산하여 최종 multi-task vector를 구성.
        #   - mask_t = |θ_t| > λ |θ_mt - θ_t|
        #   - masked_tv_t = θ_t * mask_t
        #   - multi_tv_masked = sum_t masked_tv_t
        tall_masks = {}
        masked_tvs: List[StateDict] = []
        for name, tv in tqdm(task_vectors.items(), desc="Generating TallMasks (TA)", total=len(task_vectors)):
            mask = _generate_task_masks(
                OrderedDict((k, multi_tv[k]) for k in keys),
                OrderedDict((k, tv[k]) for k in keys),
                tall_mask_lambda=tall_mask_lambda,
            )
            tall_masks[name] = mask
            masked_tv = _state_dict_hadamard(mask, tv)
            masked_tvs.append(masked_tv)

        multi_tv_masked = _state_dict_sum(masked_tvs)

        # baseline(sd_pre)에 TallMask가 적용된 multi-task vector를 더해 최종 weight 구성
        final_sd = _state_dict_add(sd_pre, multi_tv_masked, exclude_keys=exclude_keys)

        # 모든 키가 포함되었는지 검증
        if set(final_sd.keys()) != set(sd_pre.keys()):
            missing = set(sd_pre.keys()) - set(final_sd.keys())
            logger.warning("TATallMask: final_sd에 누락된 키가 있습니다: %s", missing)
            # 누락된 키는 sd_pre에서 복사
            for k in missing:
                final_sd[k] = sd_pre[k]

        if debug:
            logger.info("TallMaskTaskArithmetic: generated masks for models: %s", list(tall_masks.keys()))
            logger.info("TATallMask: final_sd 키 개수=%d, sd_pre 키 개수=%d", len(final_sd), len(sd_pre))

        out = SmolVLAPolicy(policies[0].config)
        out.load_state_dict(final_sd)
        out.to("cpu")
        return out

    if algo_name == "ties_TallMask":
        # TIES: 벡터 공간에서 task vector 합성을 수행
        flat_ft = torch.vstack(
            [
                _state_dict_to_vector(sd, remove_keys=exclude_keys)
                for sd in sd_list
            ]
        )
        flat_ptm = _state_dict_to_vector(sd_pre, remove_keys=exclude_keys)
        tv_flat = flat_ft - flat_ptm

        merged_tv_flat = _ties_merging(tv_flat, reset_thresh=1.0, merge_func="sum")
        merged_tv = _vector_to_state_dict(merged_tv_flat, sd_pre, remove_keys=exclude_keys)

        # cast Parameter → Tensor
        for k, v in merged_tv.items():
            if isinstance(v, nn.Parameter):
                merged_tv[k] = v.detach().clone()

        # eval_task 없이 merged_tv 자체를 최종 weight로 사용
        final_sd = _state_dict_add(sd_pre, merged_tv, exclude_keys=exclude_keys)

        # 모든 키가 포함되었는지 검증
        if set(final_sd.keys()) != set(sd_pre.keys()):
            missing = set(sd_pre.keys()) - set(final_sd.keys())
            logger.warning("ties_TallMask: final_sd에 누락된 키가 있습니다: %s", missing)
            # 누락된 키는 sd_pre에서 복사
            for k in missing:
                final_sd[k] = sd_pre[k]

        out = SmolVLAPolicy(policies[0].config)
        out.load_state_dict(final_sd)
        out.to("cpu")
        return out

    raise ValueError(f"Unknown algo_name: {algo_name}")


def _copy_preprocessor_postprocessor_from_repo(
    source_repo_id: str,
    output_dir: Path,
) -> None:
    """원본 모델에서 preprocessor/postprocessor 파일들을 다운로드해서 output_dir에 복사."""
    processor_files = [
        "policy_preprocessor.json",
        "policy_preprocessor_step_5_normalizer_processor.safetensors",
        "policy_postprocessor.json",
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    ]

    logger.info("Copying preprocessor/postprocessor from %s", source_repo_id)
    for filename in processor_files:
        try:
            local_path = hf_hub_download(
                repo_id=source_repo_id,
                filename=filename,
                local_dir=str(output_dir),
                local_dir_use_symlinks=False,
            )
            logger.info("Copied %s to %s", filename, output_dir)
        except Exception as e:
            logger.warning("Failed to copy %s from %s: %s", filename, source_repo_id, e)


def _push_merged_to_hub(
    policy: SmolVLAPolicy,
    output_dir: Path,
    repo_id: str | None,
    private: bool | None,
) -> None:
    """머지된 모델을 Hugging Face Hub에 업로드 (항상 public)."""
    api = HfApi()

    if repo_id is None:
        # config에 repo_id가 있으면 사용, 없으면 에러
        repo_id = getattr(policy.config, "repo_id", None)
        if repo_id is None:
            raise ValueError(
                "repo_id를 지정하지 않았고 policy.config.repo_id도 없습니다. "
                "--repo_id를 명시적으로 전달하세요."
            )

    created = api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
    logger.info("Using Hugging Face Hub repo_id=%s", created.repo_id)

    api.upload_folder(
        repo_id=created.repo_id,
        repo_type="model",
        folder_path=str(output_dir),
        commit_message="Upload merged SmolVLA policy",
        allow_patterns=["*.safetensors", "*.json", "*.md"],
        ignore_patterns=["*.tmp", "*.log"],
    )


def _ensure_model_card(
    output_dir: Path,
    repos: List[str],
    algo_name: str,
    weight: float,
    tall_mask_lambda: float,
) -> None:
    """
    Hugging Face Hub에 업로드할 때 사용할 기본 모델 카드(README.md)를 생성.
    - 이미 README.md 가 있으면 건드리지 않음.
    - lerobot / SmolVLA / 머지 설정 정도만 간단히 기록.
    """
    readme_path = output_dir / "README.md"
    if readme_path.exists():
        return

    base_models = ", ".join(repos)
    # base_model 필드는 단일 모델 ID만 허용하므로 제거 (여러 모델은 README 본문에만 기록)
    lines = [
        "---",
        "tags:",
        "  - robotics",
        "  - lerobot",
        "  - smolvla",
        "  - merged-model",
        "library_name: lerobot",
        "license: apache-2.0",
        "language:",
        "  - en",
        "  - ko",
        "---",
        "",
        "# Merged SmolVLA",
        "",
        "여러 SmolVLA 정책을 병합 알고리즘으로 합쳐 만든 모델입니다.",
        "",
        "## Base models",
    ]
    # 각 모델을 별도 리스트 아이템으로 표시
    for repo in repos:
        lines.append(f"- {repo}")
    lines.extend([
        "",
        "## Merge config",
        f"- algo_name: `{algo_name}`",
        f"- weight: `{weight:.3f}`",
        f"- tall_mask_lambda: `{tall_mask_lambda:.3f}`",
        "",
        "_This README was auto-generated by `lerobot_merge_smolvla.py`. 수정해서 사용해도 됩니다._",
        "",
    ])

    readme_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    if args.command != "merge":
        raise ValueError(f"Unknown command: {args.command}")

    if args.algo_name not in {"weighted_average", "TATallMask", "ties_TallMask"}:
        raise NotImplementedError(
            f"algo_name='{args.algo_name}' 는 아직 지원하지 않습니다. "
            "지원 알고리즘: weighted_average, TATallMask, ties_TallMask"
        )

    repos = [r.strip() for r in args.repos.split(",") if r.strip()]
    if not (2 <= len(repos) <= 4):
        raise ValueError(
            f"--repos 는 2~4개의 repo id를 콤마로 구분해서 전달해야 합니다. 받은 값: {args.repos}"
        )

    logger.info("Loading SmolVLA policies from: %s", ", ".join(repos))
    policies = [SmolVLAPolicy.from_pretrained(r) for r in repos]

    logger.info(
        "Merging trainable parameters only with algo=%s (w=%.3f, tall_mask_lambda=%.2f)",
        args.algo_name,
        args.weight,
        args.tall_mask_lambda,
    )
    merged_policy = _merge_with_algo(
        policies,
        algo_name=args.algo_name,
        weight=args.weight,
        tall_mask_lambda=args.tall_mask_lambda,
        debug=args.debug,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Saving merged policy to %s", output_dir)
    merged_policy.save_pretrained(output_dir)

    # 원본 모델에서 preprocessor/postprocessor 복사 (lerobot 형식)
    # 여러 모델을 병합하는 경우, 첫 번째 모델의 processor를 기준으로 사용
    logger.info("Copying preprocessor/postprocessor from source model (first repo: %s)", repos[0])
    _copy_preprocessor_postprocessor_from_repo(repos[0], output_dir)

    # 기본 모델 카드(README.md)가 없으면 자동 생성
    _ensure_model_card(
        output_dir=output_dir,
        repos=repos,
        algo_name=args.algo_name,
        weight=args.weight,
        tall_mask_lambda=args.tall_mask_lambda,
    )

    if args.push_to_hub:
        logger.info("Pushing merged policy to Hugging Face Hub (public)")
        _push_merged_to_hub(
            merged_policy,
            output_dir=output_dir,
            repo_id=args.repo_id,
            private=None,  # 항상 public으로 업로드
        )


if __name__ == "__main__":
    main()

