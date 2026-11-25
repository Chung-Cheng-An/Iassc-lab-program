"""Export a trained rsl_rl (PPO ActorCritic) policy for real-robot deployment.

No Isaac Sim needed, only torch:

    python scripts/export_policy.py --run logs/rsl_rl/openarm_yolo_reach/2026-10-02_11-46-37 --out E:/openarm_deploy
    python scripts/export_policy.py --checkpoint logs/rsl_rl/openarm_lift/<run>/model_4498.pt --out E:/openarm_deploy

--run picks the newest model_*.pt in that run folder. Without --run/--checkpoint the newest run under logs/rsl_rl is used.

Output folder:
    policy.pt / policy.onnx  raw observation -> action (observation normalization baked in)
    deploy_config.yaml       obs/action dims, activation, normalization, dt/decimation (if params/env.yaml exists)
    params/                  copy of env.yaml / agent.yaml (obs order, action scale, default joint pos ...)
    source_checkpoint.pt     the checkpoint used
"""

import argparse
import glob
import os
import re
import shutil
from datetime import datetime

import torch
import torch.nn as nn


def latest_model(run_dir):
    fs = glob.glob(os.path.join(run_dir, "model_*.pt"))
    if not fs:
        raise FileNotFoundError(f"No model_*.pt in {run_dir}")
    return max(fs, key=lambda f: int(re.findall(r"model_(\d+)", os.path.basename(f))[0]))


def latest_run(root="logs/rsl_rl"):
    runs = [d for d in glob.glob(os.path.join(root, "*", "*")) if glob.glob(os.path.join(d, "model_*.pt"))]
    if not runs:
        raise FileNotFoundError(f"No runs under {root}")
    return max(runs, key=os.path.getmtime)


class DeployPolicy(nn.Module):
    def __init__(self, linears, activation, mean, std, eps=1e-2):
        super().__init__()
        act = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh, "selu": nn.SELU, "lrelu": nn.LeakyReLU}[activation]
        layers = []
        for i, (w, b) in enumerate(linears):
            lin = nn.Linear(w.shape[1], w.shape[0])
            lin.weight.data.copy_(w)
            lin.bias.data.copy_(b)
            layers.append(lin)
            if i < len(linears) - 1:
                layers.append(act())
        self.mlp = nn.Sequential(*layers)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.eps = eps

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # rsl_rl EmpiricalNormalization: (x - mean) / (std + eps)
        return self.mlp((obs - self.mean) / (self.std + self.eps))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=None, help="run folder (uses its newest model_*.pt)")
    p.add_argument("--checkpoint", default=None, help="specific model_XXX.pt")
    p.add_argument("--out", default=None, help="output dir (e.g. USB). default: <run>/exported_deploy")
    p.add_argument("--activation", default=None, help="elu/relu/tanh... (default: read agent.yaml, else elu)")
    args = p.parse_args()

    if args.checkpoint:
        ckpt_path = args.checkpoint
    else:
        ckpt_path = latest_model(args.run or latest_run())
    run_dir = os.path.dirname(os.path.abspath(ckpt_path))
    print(f"[INFO] checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"]

    # actor linear layers, in order
    keys = sorted(
        [k for k in sd if k.startswith("actor.") and k.endswith(".weight") and sd[k].ndim == 2],
        key=lambda k: int(k.split(".")[1]),
    )
    linears = [(sd[k].float(), sd[k[:-6] + "bias"].float()) for k in keys]
    obs_dim, act_dim = linears[0][0].shape[1], linears[-1][0].shape[0]
    print(f"[INFO] actor: {[tuple(w.shape) for w, _ in linears]}  obs_dim={obs_dim} action_dim={act_dim}")

    # activation
    activation = args.activation
    agent_yaml = os.path.join(run_dir, "params", "agent.yaml")
    if activation is None and os.path.exists(agent_yaml):
        m = re.search(r"^\s*activation:\s*(\w+)", open(agent_yaml, encoding="utf-8").read(), re.M)
        activation = m.group(1) if m else None
    activation = activation or "elu"

    # observation normalizer (only present if empirical_normalization / obs_normalization was on)
    mean_k = [k for k in sd if "obs_normalizer" in k and k.endswith("_mean") and ("actor" in k or k.startswith("obs"))]
    if mean_k:
        base = mean_k[0][: -len("_mean")]
        mean = sd[base + "_mean"].float().flatten()
        std = sd[base + "_std"].float().flatten() if (base + "_std") in sd else torch.sqrt(sd[base + "_var"].float().flatten())
        print(f"[INFO] observation normalizer found ({base}) -> baked in.")
    else:
        mean, std = torch.zeros(obs_dim), torch.ones(obs_dim)
        # with identity: (x-0)/(1+eps) would be wrong -> cancel eps
        std = std - 1e-2
        print("[INFO] no observation normalizer in checkpoint -> identity.")
    print(f"[INFO] activation: {activation}")

    model = DeployPolicy(linears, activation, mean, std).eval()

    out_dir = args.out or os.path.join(run_dir, "exported_deploy")
    os.makedirs(out_dir, exist_ok=True)
    dummy = torch.randn(1, obs_dim)
    torch.jit.trace(model, dummy).save(os.path.join(out_dir, "policy.pt"))
    torch.onnx.export(
        model, dummy, os.path.join(out_dir, "policy.onnx"),
        input_names=["obs"], output_names=["action"], opset_version=17,
        dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
    )
    shutil.copy(ckpt_path, os.path.join(out_dir, "source_checkpoint.pt"))
    params_src = os.path.join(run_dir, "params")
    if os.path.isdir(params_src):
        shutil.copytree(params_src, os.path.join(out_dir, "params"), dirs_exist_ok=True)

    sim_dt, decim = None, None
    env_yaml = os.path.join(params_src, "env.yaml")
    if os.path.exists(env_yaml):
        txt = open(env_yaml, encoding="utf-8").read()
        m1, m2 = re.search(r"^\s{2}dt:\s*([\d.eE-]+)", txt, re.M), re.search(r"^decimation:\s*(\d+)", txt, re.M)
        sim_dt = float(m1.group(1)) if m1 else None
        decim = int(m2.group(1)) if m2 else None
    hz = round(1.0 / (sim_dt * decim), 2) if sim_dt and decim else "FILL_IN"

    with open(os.path.join(out_dir, "deploy_config.yaml"), "w", encoding="utf-8") as f:
        f.write(f"""# Generated {datetime.now().isoformat(timespec='seconds')}
source_checkpoint: {ckpt_path}
obs_dim: {obs_dim}
action_dim: {act_dim}
activation: {activation}
sim_dt: {sim_dt}
decimation: {decim}
control_hz: {hz}
note: policy.pt takes RAW observations (normalization included). Output = raw action.
# Obs order / action scale / default joint pos / joint order: see params/env.yaml
#   target_joint_pos = default_joint_pos + action_scale * action   (for JointPositionAction, use_default_offset=True)
""")
    print(f"[DONE] exported to {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    main()
