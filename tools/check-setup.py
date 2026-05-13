#!/usr/bin/env python3
"""
check-setup.py — Verifica che il tuo ambiente sia pronto per lavorare al progetto.

Lancia questo PRIMA di start-work.py la prima volta che configuri la macchina.
Esce con codice 0 se tutto OK, > 0 se manca qualcosa. Per ogni problema spiega
come risolverlo.

Usage:
    python tools/check-setup.py
"""

from __future__ import annotations
import shutil
import subprocess
import sys
from pathlib import Path


GREEN = "\033[32m" if sys.stdout.isatty() else ""
RED = "\033[31m" if sys.stdout.isatty() else ""
YELLOW = "\033[33m" if sys.stdout.isatty() else ""
RESET = "\033[0m" if sys.stdout.isatty() else ""


def ok(msg: str):
    print(f"{GREEN}[OK]{RESET}  {msg}")


def warn(msg: str):
    print(f"{YELLOW}[!!]{RESET}  {msg}")


def fail(msg: str):
    print(f"{RED}[X]{RESET}   {msg}")


issues: list[str] = []


def need(name: str, install_hint: dict[str, str]):
    """Verifica che `name` sia in PATH; altrimenti registra un issue."""
    if shutil.which(name):
        try:
            v = subprocess.check_output([name, "--version"], text=True, timeout=5,
                                        stderr=subprocess.STDOUT).strip().split("\n")[0]
        except Exception:
            v = "(installato, versione non disponibile)"
        ok(f"{name}: {v}")
        return True
    fail(f"{name} non in PATH")
    print(f"      Install:")
    for os_name, cmd in install_hint.items():
        print(f"        {os_name:7s}  {cmd}")
    issues.append(f"manca {name}")
    return False


def check_git_identity():
    try:
        email = subprocess.check_output(["git", "config", "--get", "user.email"],
                                        text=True).strip()
        name = subprocess.check_output(["git", "config", "--get", "user.name"],
                                       text=True).strip()
    except subprocess.CalledProcessError:
        email = name = ""
    if not email or not name:
        warn("git user.name / user.email non impostati")
        print("      Fix:  git config --global user.name 'TuoNome'")
        print("            git config --global user.email '<id>+<user>@users.noreply.github.com'")
        issues.append("git identity mancante")
        return
    if not email.endswith("users.noreply.github.com"):
        warn(f"git email non noreply ({email}). Consigliato usare la noreply GitHub per privacy.")
        print("      Find yours: https://github.com/settings/emails")
        # Non bloccante
    ok(f"git identity: {name} <{email}>")


def check_rclone_remote():
    try:
        out = subprocess.check_output(["rclone", "listremotes"], text=True)
    except Exception as e:
        fail(f"rclone listremotes ha fallito: {e}")
        issues.append("rclone non funziona")
        return
    if "gdrive:" not in out:
        fail("Remote 'gdrive:' non configurato in rclone")
        print("      Fix: rclone config (vedi tools/README.md)")
        issues.append("remote gdrive: mancante")
        return
    ok(f"rclone remotes: {out.strip().split(chr(10))}")

    # Test mirato sulla cartella del progetto (no lista root MyDrive che puo'
    # essere lentissima se hai molte cartelle).
    try:
        out = subprocess.check_output(
            ["rclone", "lsd", "gdrive:FAIML_group_64"],
            text=True, stderr=subprocess.STDOUT, timeout=30,
        )
    except subprocess.TimeoutExpired:
        fail("rclone lsd gdrive:FAIML_group_64 timeout (> 30s). Internet lento o auth scaduta.")
        print("      Fix: rclone config reconnect gdrive: (o rilancia tra qualche minuto)")
        issues.append("rclone gdrive: timeout")
        return
    except subprocess.CalledProcessError as e:
        fail(f"rclone lsd gdrive:FAIML_group_64 ha fallito\n      {e.output.strip() if e.output else e}")
        print("      Possibili cause:")
        print("        - Token OAuth scaduto: rclone config reconnect gdrive:")
        print("        - Shortcut MyDrive mancante: apri https://drive.google.com/drive/folders/1oWQ04jeGPfCCc9MA8PWKABnhlUtTwhmE")
        print("                                     tasto destro -> Add shortcut to Drive -> MyDrive")
        issues.append("rclone gdrive:FAIML_group_64 non raggiungibile")
        return
    ok("rclone gdrive:FAIML_group_64 raggiungibile")

    # Verifica sottocartelle attese
    expected_dirs = {"code", "checkpoints", "locks", "models", "tensorboard", "plots"}
    found = {line.split()[-1] for line in out.strip().splitlines() if line.strip()}
    missing = expected_dirs - found
    if missing:
        warn(f"Sottocartelle attese mancanti nel Drive: {sorted(missing)}")
        print("      (potrebbero non essere ancora create — chiedi al coordinatore)")
    else:
        ok(f"Drive folder layout OK (trovate: {sorted(expected_dirs & found)})")


def check_github_access():
    """Verifica che possa pullare/pushare dalla repo del gruppo."""
    try:
        out = subprocess.check_output(["git", "remote", "get-url", "origin"],
                                      text=True).strip()
        ok(f"git remote origin: {out}")
    except subprocess.CalledProcessError:
        fail("git remote 'origin' non configurato (non sei in una repo clonata?)")
        issues.append("git remote mancante")
        return

    # ls-remote come smoke test di auth
    try:
        subprocess.check_call(["git", "ls-remote", "origin", "HEAD"],
                              stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL,
                              timeout=15)
        ok("GitHub: pull access OK")
    except subprocess.CalledProcessError:
        fail("git ls-remote origin ha fallito (no access o auth scaduta)")
        print("      Fix: gh auth login")
        issues.append("GitHub auth")


def check_repo_layout():
    """Verifica che siamo nella repo giusta e i file chiave esistano."""
    repo_root = Path(__file__).resolve().parent.parent
    expected = [
        "part1/agent.py",
        "part2/train_sb3.py",
        "part2/rand_wrapper.py",
        "part2/eval_sb3.py",
        "requirements.txt",
        "tools/start-work.py",
    ]
    missing = [p for p in expected if not (repo_root / p).exists()]
    if missing:
        fail(f"File attesi mancanti nella repo: {missing}")
        issues.append("repo layout incompleto")
    else:
        ok(f"Repo layout OK ({len(expected)} file chiave presenti)")


def main():
    print(f"=== check-setup.py — {Path.cwd()} ===\n")

    print("[tool] base tools")
    need("python", {"all": "https://www.python.org/downloads/ (>=3.10)"})
    need("git", {"all": "https://git-scm.com/downloads"})
    need("rclone", {
        "Windows": "choco install rclone",
        "macOS":   "brew install rclone",
        "Linux":   "apt install rclone   (o equivalente)",
    })
    need("gh", {
        "Windows": "winget install GitHub.cli",
        "macOS":   "brew install gh",
        "Linux":   "https://github.com/cli/cli#installation",
    })
    print()

    print("[tool] git identity")
    check_git_identity()
    print()

    print("[tool] GitHub access")
    check_github_access()
    print()

    print("[tool] rclone + Drive")
    check_rclone_remote()
    print()

    print("[tool] repo layout")
    check_repo_layout()
    print()

    if issues:
        print(f"{RED}{'=' * 60}{RESET}")
        print(f"{RED}{len(issues)} problemi da risolvere prima di iniziare:{RESET}")
        for i in issues:
            print(f"  - {i}")
        sys.exit(1)
    else:
        print(f"{GREEN}{'=' * 60}{RESET}")
        print(f"{GREEN}Tutto pronto. Lancia: python tools/start-work.py{RESET}")
        sys.exit(0)


if __name__ == "__main__":
    main()
