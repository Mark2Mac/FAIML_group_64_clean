# tools/

Script di infrastruttura del gruppo. Cross-OS (Windows, macOS, Linux), in Python
puro per portabilita'.

## start-work.py

Sincronizzazione one-shot da lanciare PRIMA di ogni sessione di lavoro.

```bash
python tools/start-work.py                  # git pull + push del codice al Drive
python tools/start-work.py --pull-results   # scarica anche models/plots/tensorboard
python tools/start-work.py --dry-run        # mostra i comandi senza eseguire
python tools/start-work.py --no-git         # skip git pull (utile su branch detached)
```

Non e' un daemon: gira 30-60s, sincronizza, esce.

## Setup rclone (one-shot, 5 minuti per ognuno)

### 1. Install rclone

| OS | Comando |
|---|---|
| Windows | `choco install rclone` (o scaricare da rclone.org) |
| macOS | `brew install rclone` |
| Linux Debian/Ubuntu | `sudo apt install rclone` |
| Linux Arch | `sudo pacman -S rclone` |

Verifica: `rclone version`

### 2. Configura il remote Google Drive

```bash
rclone config
```

Risposte interattive:

```
n           # New remote
gdrive      # name (IMPORTANTE: deve essere esattamente "gdrive")
drive       # Google Drive
            # client_id: vuoto (uso default)
            # client_secret: vuoto
1           # scope: full access (scelta 1)
            # service_account_file: vuoto
n           # edit advanced config: no
y           # use auto config: apre browser, fai login Google
n           # configure as team drive: no
y           # yes, confirma
q           # quit
```

### 3. Aggiungi shortcut della cartella condivisa al tuo MyDrive

Apri https://drive.google.com/drive/folders/1oWQ04jeGPfCCc9MA8PWKABnhlUtTwhmE
→ tasto destro sulla cartella `FAIML_group_64` → "Add shortcut to Drive"
→ scegli MyDrive.

Lo script si aspetta che la cartella sia raggiungibile come
`gdrive:FAIML_group_64/...`. Se hai un nome di shortcut diverso, edita
`DRIVE_FOLDER` nella prima riga di `start-work.py`.

### 4. Verifica setup

```bash
rclone lsd gdrive:FAIML_group_64
```

Devi vedere elencate le cartelle del Drive condiviso (code, locks, checkpoints,
models, tensorboard, plots).

## Quando eseguire start-work.py

- **Inizio giornata**: prima di aprire l'IDE e committare nuove modifiche.
- **Prima di lanciare un Colab**: cosi' la cartella `code/` nel Drive ha
  l'ultima versione e i Colab leggono codice fresh.
- **Quando vuoi vedere risultati training dei colleghi**: con `--pull-results`,
  popola `./drive-mirror/` (gitignored) con models/plots/tensorboard.

## Workflow tipico

```bash
# Mattina, inizio lavoro
cd <repo>
python tools/start-work.py

# ...editi i file in VS Code o IDE preferito...
git commit -am "wip ppo skeleton"
git push

# Vuoi aggiornare il Drive con il nuovo commit (cosi' i Colab lo vedono):
python tools/start-work.py
```

Workflow Colab cloud-first (per chi vuole programmare direttamente in Colab):

```python
# In una nuova cella nel notebook (prima volta serve auth):
!gh auth login
!git clone https://github.com/Mark2Mac/FAIML_group_64_clean.git
%cd FAIML_group_64_clean
# ...editi i file via Colab editor...
!git add -A && git commit -m "fix sigma" && git push
```

Niente `start-work.py` da Colab: GitHub e' gia' la source of truth, e quando
qualcuno (anche tu) lancia `start-work.py` da locale, il Drive si aggiorna.
