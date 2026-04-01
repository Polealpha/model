# Remote Training Runbook

## Preconditions

- SSH target: `10.101.0.36:2222`
- Login pattern that worked in this session: `plink -ssh 10.101.0.36 -P 2222 -l "v6yvdcnv#root#bec2604c-ae04-4222-85f3-b399f6ab2e51" -pw "Qingbei36974!"`
- Remote state verified in this session:
  - `root`
  - two `NVIDIA A800-SXM4-80GB`
  - `Python 3.10.12`
  - repo unpacked to `/root/lunwen_run/lunwen`
  - `/root/lunwen_run/lunwen/.venv` exists
  - system `python3` had no `pip` module before repair

## Recommended pip recovery order

1. `python3 -m venv .venv` after confirming `python3.10-venv` is installed.
2. Activate the venv and run `python -m pip install --upgrade pip setuptools wheel`.
3. Install the repo requirements with `python -m pip install -r requirements.txt`.
4. Use `apt install python3-pip` only if you need system-level packaging for unrelated tools. It is not the preferred fix for this training path because it does not isolate dependencies and does not directly solve the repo venv bootstrap.
5. Use `python -m ensurepip` only as a fallback if the venv was created without pip and the interpreter exposes `ensurepip`.

## Final command chain

The most reliable path is to stage the steps explicitly instead of forcing one long shell command to do everything:

```bash
cd /root/lunwen_run/lunwen
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
nohup bash -lc '. .venv/bin/activate && python scripts/generate_synthetic_dataset.py --force && python scripts/train_structured_baseline.py && python scripts/train_multitask_model.py' > /root/lunwen_run/train.log 2>&1 < /dev/null &
echo $! > /root/lunwen_run/train.pid
```

If you need a single reusable entrypoint, use `/tmp/server_bootstrap_train.sh` with:

```bash
/tmp/server_bootstrap_train.sh /tmp/lunwen_upload.zip /root/lunwen_run
```

## Progress checks

```bash
cat /root/lunwen_run/train.pid
tail -n 80 /root/lunwen_run/train.log
ps -ef | grep -E 'generate_synthetic_dataset|train_structured_baseline|train_multitask_model' | grep -v grep
nvidia-smi
```

## Common failures

- `ensurepip not available`: install `python3.10-venv` and recreate the venv.
- `No module named pip`: activate the venv and run `python -m pip install --upgrade pip setuptools wheel`.
- `timeout` during bootstrap: split the process into setup and background training, then check `/root/lunwen_run/train.log`.
- `Access denied` on SSH: the JumpServer password is short-lived; re-fetch a fresh credential before retrying.

