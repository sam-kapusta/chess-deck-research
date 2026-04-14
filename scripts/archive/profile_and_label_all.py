#!/usr/bin/env python3
"""Profile + label all SAE variants that have weights but no labels yet.

Meant to run on SAIS (needs GPU for encoder forward passes + profiling).

Steps:
  1. Scan output/k_sweep/ for sae_btk_*.pt files
  2. Check which already have profiles_*.json
  3. Profile missing ones (profile_sae.py)
  4. Upload all profiles to S3
  5. Print batch labeling command to run locally

Usage (on SAIS):
    cd /home/ec2-user/SageMaker/chess-stage-a
    python3 /home/ec2-user/SageMaker/chess-deck-research/scripts/sae/profile_and_label_all.py

Then locally:
    python3 scripts/evaluation/batch_label_and_score.py label --profiles-dir /tmp/profiles/
"""
import os, sys, subprocess, json, glob

BASE = "/home/ec2-user/SageMaker/chess-stage-a"
K_SWEEP = BASE + "/output/k_sweep"
PROFILE_SCRIPT = BASE + "/profile_sae.py"
S3_BUCKET = "chess-stage-a-140023406996"

def main():
    # Find all SAE weight files
    weight_files = sorted(glob.glob(f"{K_SWEEP}/sae_btk_*.pt"))
    print(f"Found {len(weight_files)} SAE weight files:")
    for wf in weight_files:
        print(f"  {os.path.basename(wf)}")

    # Find existing profiles
    profile_files = sorted(glob.glob(f"{K_SWEEP}/profiles_btk_*.json"))
    profiled = set()
    for pf in profile_files:
        # profiles_btk_2048_k64.json → btk_2048_k64
        name = os.path.basename(pf).replace("profiles_", "").replace(".json", "")
        profiled.add(name)

    print(f"\nAlready profiled: {len(profiled)}")
    for p in sorted(profiled):
        print(f"  {p}")

    # Find which need profiling
    needs_profile = []
    for wf in weight_files:
        # sae_btk_2048_k32_aux.pt → btk_2048_k32_aux → btk_2048_k32
        name = os.path.basename(wf).replace("sae_", "").replace(".pt", "")
        # Normalize: btk_2048_k32_aux → btk_2048_k32 (profiles don't include _aux suffix)
        profile_name = name.replace("_aux", "")
        if profile_name not in profiled:
            needs_profile.append((wf, profile_name))

    if not needs_profile:
        print("\nAll SAEs are profiled!")
    else:
        print(f"\nNeed profiling ({len(needs_profile)}):")
        for wf, name in needs_profile:
            print(f"  {os.path.basename(wf)} → profiles_{name}.json")

        # Profile each
        for wf, name in needs_profile:
            print(f"\n{'='*60}")
            print(f"Profiling {os.path.basename(wf)}...")
            sys.stdout.flush()
            cmd = ["python3", PROFILE_SCRIPT, "--checkpoint", wf]
            result = subprocess.run(cmd, capture_output=False)
            if result.returncode != 0:
                print(f"  FAILED (exit {result.returncode})")
            else:
                print(f"  Done → profiles_{name}.json")

    # Upload all profiles to S3
    print(f"\n{'='*60}")
    print("Uploading profiles to S3...")
    profile_files = sorted(glob.glob(f"{K_SWEEP}/profiles_btk_*.json"))
    for pf in profile_files:
        s3_key = f"sae-eval/{os.path.basename(pf)}"
        cmd = ["aws", "s3", "cp", pf, f"s3://{S3_BUCKET}/{s3_key}"]
        subprocess.run(cmd, capture_output=True)
        print(f"  Uploaded {os.path.basename(pf)}")

    # Summary
    print(f"\n{'='*60}")
    print("READY FOR LABELING")
    print(f"Profiles on S3: s3://{S3_BUCKET}/sae-eval/profiles_btk_*.json")
    print(f"\nTo label all variants:")
    print(f"  1. Download profiles: aws s3 sync s3://{S3_BUCKET}/sae-eval/ /tmp/profiles/ --exclude '*' --include 'profiles_*.json'")
    print(f"  2. Run: python3 scripts/evaluation/batch_label_and_score.py label --profiles-dir /tmp/profiles/")


if __name__ == "__main__":
    main()
