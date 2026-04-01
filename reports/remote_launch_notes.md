# Remote Launch Notes

The server is reachable and the hardware is correct: root access worked through JumpServer, the host exposes two `NVIDIA A800-SXM4-80GB` GPUs, and the uploaded research bundle is present on the remote side. The main blocker in the first boot attempts was not compute or code, but Python packaging state: `python3.10-venv` was missing initially, then installed, and the first long bootstrap command timed out before the venv + dependency install + background training chain could finish.

Recommended recovery steps are straightforward:

1. Reuse the staged bootstrap script instead of a single long remote one-liner.
2. Always split setup from training so log/pid files are created before the background job starts.
3. Verify with `cat /root/lunwen_run/train.pid`, `tail -n 80 /root/lunwen_run/train.log`, and `nvidia-smi`.
4. Keep the JumpServer credential window in mind because the password material observed here appeared short-lived.

High-risk point: if the remote shell stays on a system interpreter without `pip`, the bootstrap will fail again unless the venv is recreated and `python -m pip` is used inside the venv. The runbook now makes that order explicit.

