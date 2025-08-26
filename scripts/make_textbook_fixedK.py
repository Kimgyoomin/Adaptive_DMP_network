import os, argparse, numpy as np
from pathlib import Path

from network_pkg.admp_core.core.gen.bezier      import rand_bezier_fixed_dist
from network_pkg.admp_core.core.gen.test_curves import circle_arc, s_curve
from network_pkg.admp_core.core.basis.make_rbf  import make_rbf
from network_pkg.admp_core.core.dmp.dmp2d       import fit_weights_2d


def resample_norm_safe(y, T, eps=1e-6):
    """일관된 정규화 함수 - 학습 코드와 동일"""
    idx = np.linspace(0, len(y) - 1, T).astype(int)
    z = y[idx]
    d = np.linalg.norm(z[-1] - z[0])
    if d < eps:
        diffs = np.diff(z, axis=0)
        L = np.sum(np.linalg.norm(diffs, axis=1))
        d = max(L, eps)
    return ((z - z[0]) / d).astype(np.float64)


def get_random_fixed(T=400, D=5.0, jitter=0.25, n_ctrl=2):
    return rand_bezier_fixed_dist(T=T, D=D, jitter=jitter, n_ctrl=n_ctrl)


def gen_testbank(T=400):
    return [circle_arc(T=T), s_curve(T=T)]      # only for open curve


def fit_trajectory_fixed_K(traj, K_fix=128, T_out=600):
    """궤적을 고정 K로 피팅하는 함수"""
    
    # 1. 정규화 좌표로 변환
    y_norm = resample_norm_safe(traj, T_out)
    
    # 2. 고정 K의 RBF 기저 생성
    c, h = make_rbf(K_fix)
    dt = 1.0 / (T_out - 1)
    
    # 3. 가중치 피팅
    wx, wy, y0, g = fit_weights_2d(y_norm, dt, c, h)
    
    # 4. 피팅 품질 계산 (rollout으로 검증)
    try:
        from network_pkg.admp_core.core.dmp.dmp2d import rollout_2d
        y_recon = rollout_2d(T_out, dt, c, h, wx, wy, y0, g, tau=1.0)
        
        # nRMSE 계산
        rmse = np.sqrt(np.mean(np.sum((y_norm - y_recon) ** 2, axis=1)))
        d_norm = np.linalg.norm(y_norm[-1] - y_norm[0])
        nrmse = rmse / d_norm if d_norm > 1e-6 else rmse
        
    except Exception as e:
        print(f"Warning: Rollout failed: {e}")
        nrmse = 0.0
        rmse = 0.0
    
    return {
        'K': K_fix,
        'wx': wx,
        'wy': wy,
        'c': c,
        'h': h,
        'y0': y0,
        'g': g,
        'nrmse': nrmse,
        'rmse': rmse,
        'dt': dt,
        'y_norm': y_norm  # 정규화된 궤적도 저장 (디버깅용)
    }


def main():
    ap = argparse.ArgumentParser(description="Generate fixed K=128 DMP dataset")
    ap.add_argument("--outdir",     type=str,   default="dataset/train_K128")
    ap.add_argument("--num",        type=int,   default=500)
    ap.add_argument("--T_out",      type=int,   default=600)
    ap.add_argument("--K_fix",      type=int,   default=128)
    ap.add_argument("--D",          type=float, default=5.0)
    ap.add_argument("--seed",       type=int,   default=42)
    args = ap.parse_args()

    # 재현성을 위한 시드 설정
    np.random.seed(args.seed)
    
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {args.num} trajectories with fixed K={args.K_fix}")
    print(f"Output directory: {outdir}")
    print(f"T_out: {args.T_out}, D: {args.D}")
    print("-" * 50)

    # 통계 수집용
    nrmse_list = []
    failed_count = 0
    
    for i in range(args.num):
        try:
            # 랜덤 파라미터 생성
            jitter = np.clip(0.2 + 0.15*np.random.randn(), 0.05, 0.5)
            n_ctrl = np.random.choice([1, 2, 3], p=[0.2, 0.6, 0.2])
            
            # 궤적 생성
            traj = get_random_fixed(T=400, D=args.D, jitter=jitter, n_ctrl=n_ctrl)
            
            # 고정 K로 피팅
            out = fit_trajectory_fixed_K(traj, K_fix=args.K_fix, T_out=args.T_out)
            
            # 결과 저장 (원본 궤적 + 피팅 결과)
            np.savez_compressed(outdir/f"{i:06d}.npz",
                traj=traj,           # 원본 궤적 (월드 좌표)
                K=out["K"],          # 고정 K
                wx=out["wx"],        # X축 가중치
                wy=out["wy"],        # Y축 가중치
                c=out["c"],          # RBF 센터
                h=out["h"],          # RBF 폭
                y0=out["y0"],        # 시작점 (정규화)
                g=out["g"],          # 목표점 (정규화)
                nrmse=out["nrmse"],  # 피팅 오차
                rmse=out["rmse"],
                dt=out["dt"],
                y_norm=out["y_norm"],  # 정규화된 궤적
                # 메타데이터
                jitter=jitter,
                n_ctrl=n_ctrl,
                K_fix=args.K_fix,
                T_out=args.T_out
            )
            
            nrmse_list.append(out["nrmse"])
            
            if i % 50 == 0:
                avg_nrmse = np.mean(nrmse_list[-50:]) if len(nrmse_list) >= 50 else np.mean(nrmse_list)
                print(f"[{i:4d} / {args.num}] K={out['K']}, nRMSE={out['nrmse']:.4f}, avg_nRMSE={avg_nrmse:.4f}")
                
        except Exception as e:
            print(f"Failed to generate trajectory {i}: {e}")
            failed_count += 1
            continue

    # 통계 출력
    if nrmse_list:
        print(f"\nGeneration completed!")
        print(f"Successfully generated: {len(nrmse_list)}/{args.num}")
        print(f"Failed: {failed_count}")
        print(f"nRMSE statistics:")
        print(f"  Mean: {np.mean(nrmse_list):.4f}")
        print(f"  Std:  {np.std(nrmse_list):.4f}")
        print(f"  Min:  {np.min(nrmse_list):.4f}")
        print(f"  Max:  {np.max(nrmse_list):.4f}")
        print(f"  95th percentile: {np.percentile(nrmse_list, 95):.4f}")

    # Test bank 생성 (고정 K로)
    tb_dir = outdir.parent / "testbank_K128"
    tb_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating test bank in {tb_dir}...")
    
    for j, t in enumerate(gen_testbank(T=800)):
        try:
            out = fit_trajectory_fixed_K(t, K_fix=args.K_fix, T_out=max(args.T_out, 800))
            np.savez_compressed(tb_dir/f"tb_{j:02d}.npz", 
                traj=t, **out, 
                test_name=f"testbank_{j}")
            print(f"[test {j}] K={out['K']}, nRMSE={out['nrmse']:.4f}")
        except Exception as e:
            print(f"Failed to generate test {j}: {e}")

    print(f"\nDataset generation completed!")
    print(f"Training data: {outdir}")
    print(f"Test data: {tb_dir}")


if __name__ == "__main__":
    main()