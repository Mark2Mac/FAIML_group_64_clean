#!/usr/bin/env python3
"""
start-work.py — One-shot pre-work sync for FAIML Group 64.

Lancia questo script ogni volta che inizi una sessione di lavoro, prima
di aprire l'IDE / cominciare a editare. Cross-OS (Windows / Linux / macOS),
non-blocking: gira 30-60 secondi, sincronizza, esce.

Cosa fa (in ordine):
  1. git pull --rebase origin main          allinea il tuo locale con GitHub
  2. rclone copy <repo>  gdrive:.../code/   pusha il codice locale al Drive
                                            cosi' i Colab vedono l'ultima versione
  3. (opzionale, con --pull-results)
     rclone copy gdrive:.../{models,plots,tensorboard}  ./drive-mirror/
     per analizzare risultati training in locale

Cosa NON fa:
  - non e' un daemon: lo lanci e basta
  - non resta in background: non polls, non watches
  - non automatizza il git push: tu decidi quando committare

Setup one-shot (5 minuti, ognuno sulla propria macchina):
  1. install rclone:
      Windows:  choco install rclone
      macOS:    brew install rclone
      Linux:    apt install rclone   (o equivalente)
  2. rclone config
      - n (new remote)
      - name: gdrive
      - storage: drive (Google Drive)
      - lascia client_id e client_secret vuoti per usare i default rclone
      - scope: drive
      - root_folder_id: lascia vuoto (uso MyDrive)
      - service_account_file: vuoto
      - edit advanced config: n
      - use auto config: y  (apre browser, fai login Google)
      - configure as team drive: n
  3. verifica:
      rclone lsd gdrive:
      (deve elencare le cartelle del tuo MyDrive)

Usage:
  python tools/start-work.py                    # sync solo codice repo -> Drive
  python tools/start-work.py --pull-results     # sync codice + scarica risultati
  python tools/start-work.py --no-git           # skip git pull (es. branch detached)
  python tools/start-work.py --dry-run          # mostra cosa farebbe, no exec
"""

from __future__ import annotations
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DRIVE_FOLDER = "FAIML_group_64"  # nome cartella in MyDrive (lo shortcut)
DRIVE_REMOTE = f"gdrive:{DRIVE_FOLDER}"

# File / cartelle da NON pushare al Drive (gitignored o irrilevanti per Colab)
RCLONE_EXCLUDES = [
    ".git/**",
    ".github/**",
    ".gitignore",
    "*.pth",
    "*.zip",
    "models/**",
    "runs/**",
    "wandb/**",
    "tensorboard/**",
    "__pycache__/**",
    "*.pyc",
    ".venv/**",
    "venv/**",
    "rag-venv/**",
    ".vscode/**",
    ".idea/**",
    # Niente exclude *.png/*.mp4/*.gif: alcuni asset di panda-gym
    # (es. colored_cube.png) servono al rendering dell'env.
    "drive-mirror/**",
    # NB: NON escludiamo part2/panda-gym/ — la versione PyPI non ha i kwarg
    # `type="source"/"target"` che servono per Task 5/6. Va il custom del prof.
    "part2/panda-gym/.github/**",
    "part2/panda-gym/docs/**",
    "part2/panda-gym/test/**",
    "part2/panda-gym/examples/**",
]


def fail(msg: str, code: int = 1):
    print(f"\n[start-work] ERRORE: {msg}", file=sys.stderr)
    sys.exit(code)


def check_tool(name: str, install_hint: str):
    if shutil.which(name) is None:
        fail(f"'{name}' non e' in PATH. Installalo: {install_hint}")


def check_rclone_remote():
    """Verifica che il remote 'gdrive' esista in rclone config."""
    try:
        out = subprocess.check_output(["rclone", "listremotes"], text=True)
    except subprocess.CalledProcessError as e:
        fail(f"rclone listremotes ha fallito: {e}")
    if "gdrive:" not in out:
        fail("Remote 'gdrive:' non configurato. Lancia: rclone config "
             "(vedi setup nel docstring di questo script).")


def run(cmd: list[str], dry: bool) -> int:
    print(f"$ {' '.join(cmd)}")
    if dry:
        return 0
    return subprocess.call(cmd)


def git_pull(dry: bool) -> int:
    return run(["git", "-C", str(REPO_ROOT), "pull", "--rebase", "origin", "main"], dry)


def rclone_push_code(dry: bool) -> int:
    args = ["rclone", "copy", "--update", "--progress"]
    for pat in RCLONE_EXCLUDES:
        args += ["--exclude", pat]
    args += [str(REPO_ROOT), f"{DRIVE_REMOTE}/code"]
    return run(args, dry)


def rclone_pull_results(dry: bool) -> int:
    """Scarica models/, plots/, tensorboard/ dal Drive a ./drive-mirror/."""
    local_mirror = REPO_ROOT / "drive-mirror"
    local_mirror.mkdir(exist_ok=True)
    rc = 0
    for sub in ("models", "plots", "tensorboard"):
        args = [
            "rclone", "copy", "--update", "--progress",
            f"{DRIVE_REMOTE}/{sub}",
            str(local_mirror / sub),
        ]
        rc |= run(args, dry)
    return rc


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pull-results", action="store_true",
                    help="Scarica anche models/plots/tensorboard dal Drive in ./drive-mirror/")
    ap.add_argument("--no-git", action="store_true",
                    help="Skip git pull (utile se sei su branch detached o conflitti)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Mostra i comandi senza eseguirli")
    args = ap.parse_args()

    print(f"[start-work] repo root: {REPO_ROOT}")
    print(f"[start-work] Drive remote: {DRIVE_REMOTE}")

    # Sanity check tool
    check_tool("git", "https://git-scm.com/downloads")
    check_tool("rclone", "vedi setup nel docstring di questo script")
    check_rclone_remote()

    # Step 1: git pull
    if not args.no_git:
        print("\n[1/3] git pull --rebase origin main")
        rc = git_pull(args.dry_run)
        if rc != 0:
            fail("git pull ha fallito. Risolvi prima di continuare.")
    else:
        print("\n[1/3] git pull   (SKIPPED via --no-git)")

    # Step 2: push code to Drive
    print(f"\n[2/3] rclone copy <repo>  ->  {DRIVE_REMOTE}/code")
    rc = rclone_push_code(args.dry_run)
    if rc != 0:
        fail("rclone push del codice ha fallito.")

    # Step 3: optional pull results
    if args.pull_results:
        print(f"\n[3/3] rclone copy  {DRIVE_REMOTE}/(models,plots,tensorboard)  ->  ./drive-mirror/")
        rc = rclone_pull_results(args.dry_run)
        if rc != 0:
            fail("rclone pull dei risultati ha fallito.")
    else:
        print("\n[3/3] pull results   (SKIPPED, usa --pull-results se vuoi scaricare)")

    print("\n[start-work] OK. Buon lavoro!")


if __name__ == "__main__":
    main()
