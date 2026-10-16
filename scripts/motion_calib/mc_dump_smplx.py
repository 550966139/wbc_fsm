"""mc_dump_smplx.py - one-shot GVHMR output dump (RUNS ON 224, gmr conda env).

Loads a GVHMR hmr4d_results.pt through GMR's own smplx utilities and writes a
plain-numpy npz (joints/body_pose/global_orient/transl/betas/human_height)
that measure_smplx.py consumes on any machine. Paths below are 224 site
defaults; the GVHMR world is Y-up.

Usage (224): conda activate gmr && python mc_dump_smplx.py
"""
import numpy as np, torch
sys.path.insert(0, "/home/hnsjb/dsh-workspace/GMR")
from general_motion_retargeting.utils.smpl import load_gvhmr_pred_file
from smplx.joint_names import JOINT_NAMES

BM = pathlib.Path("/home/hnsjb/dsh-workspace/GMR/assets/body_models")
os.makedirs("/home/hnsjb/dsh-workspace/mc_dump", exist_ok=True)
for tag, pt in [
    ("take6", "/home/hnsjb/dsh-workspace/GVHMR/outputs/demo/take6_1080x1920_30fps/hmr4d_results.pt"),
    ("take2", "/home/hnsjb/dsh-workspace/GVHMR/outputs/demo/take2_1080x1920_30fps/hmr4d_results.pt"),
]:
    smplx_data, body_model, smplx_output, h = load_gvhmr_pred_file(pt, BM)
    nj = len(body_model.parents)
    names = list(JOINT_NAMES[:nj])
    J = smplx_output.joints.detach().numpy()[:, :nj, :]
    P = torch.load(pt, map_location="cpu")["smpl_params_global"]
    np.savez_compressed(
        f"/home/hnsjb/dsh-workspace/mc_dump/{tag}_smplx.npz",
        joints=J.astype(np.float32), names=np.array(names),
        body_pose=P["body_pose"].numpy(), global_orient=P["global_orient"].numpy(),
        transl=P["transl"].numpy(), betas=P["betas"].numpy(), human_height=float(h),
    )
    feet = [i for i, n in enumerate(names) if any(k in n for k in ("foot", "ankle", "toe", "heel"))]
    print(tag, "joints", J.shape, "human_height", float(h), "feet_idx", feet, [names[i] for i in feet])
