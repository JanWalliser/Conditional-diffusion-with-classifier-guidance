# Conditional Diffusion with Classifier Guidance

Dieses Projekt trainiert ein DDPM-Diffusionsmodell auf CIFAR-10 und nutzt optional einen separaten Noisy-Image-Classifier für Classifier Guidance beim Sampling.

Das Projekt besteht aktuell aus drei Hauptteilen:

1. Training eines Classifiers auf verrauschten CIFAR-10-Bildern
2. Training eines DDPM-Denoising-U-Nets
3. Sampling von Bildern mit oder ohne Classifier Guidance

---

## Setup

Virtuelle Umgebung erstellen:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

Dependencies installieren:

```powershell
pip install -r requirements.txt
```

Die Python-Befehle werden aus dem Projekt-Root ausgeführt, also aus dem Ordner, in dem `src/`, `configs/` und `requirements.txt` liegen.

---

## Projektstruktur

```text
configs/
  classifier_cifar10.yaml
  ddpm_cifar10.yaml
  ddpm_cifar10_finetune.yaml
  guidance_sweep.yaml

src/
  data/
    cifar10.py

  diffusion/
    schedule.py
    ddpm.py
    sampler.py

  models/
    classifier.py
    unet.py
    unet_classifier.py

  training/
    train_classifier.py
    train_ddpm.py

  sampling/
    sample_guided.py
```

---

## Dateien im Projekt

### `src/data/cifar10.py`

Erstellt die CIFAR-10-DataLoader für Training und Validation.

Die Bilder werden auf den Bereich `[-1, 1]` normalisiert, damit sie direkt für DDPM-Training verwendet werden können.

Wichtige Funktion:

```python
get_cifar10_loaders(...)
```

Diese Funktion gibt zwei Loader zurück:

```python
loaders.train
loaders.val
```

---

### `src/diffusion/schedule.py`

Definiert den Diffusion-Noise-Schedule.

Der Schedule enthält unter anderem:

- `betas`
- `alphas`
- `alpha_bars`
- `sqrt_alpha_bars`
- `sqrt_one_minus_alpha_bars`

Außerdem enthält die Datei die Funktion:

```python
q_sample(x_0, t, noise)
```

Diese erzeugt aus einem sauberen Bild `x_0` ein verrauschtes Bild `x_t`.

---

### `src/diffusion/ddpm.py`

Enthält den DDPM-Wrapper um das Denoising-U-Net.

Die Datei übernimmt:

- Forward-Noising
- Training-Loss
- Vorhersage von `x_0` aus vorhergesagtem Noise
- Berechnung von Mean und Variance für Reverse-Diffusion-Schritte

Wichtige Klasse:

```python
DDPM
```

Wichtige Funktionen:

```python
q_sample(...)
training_loss(...)
p_mean_variance(...)
predict_x0_from_eps(...)
```

---

### `src/diffusion/sampler.py`

Enthält den DDPM-Sampler für die Bilderzeugung.

Die Datei startet mit normalverteiltem Rauschen und führt die Reverse-Diffusion-Schritte aus, bis ein Bild entsteht.

Optional kann ein Classifier für Classifier Guidance verwendet werden.

Wichtige Klasse:

```python
DDPMSampler
```

Wichtige Funktionen:

```python
sample(...)
p_sample(...)
classifier_gradient(...)
```

---

### `src/models/unet.py`

Enthält das Denoising-U-Net für das DDPM-Modell.

Das Modell bekommt:

```text
x_t, t
```

und sagt das hinzugefügte Rauschen vorher:

```text
epsilon_pred
```

Wichtige Klasse:

```python
DenoisingUNet
```

Wichtige Bausteine:

- Residual Blocks
- Timestep Embeddings
- Attention Blocks
- Encoder/Decoder mit Skip Connections

---

### `src/models/classifier.py`

Enthält den einfachen Noisy-Image-Classifier.

Der Classifier bekommt ein verrauschtes Bild `x_t` und den Timestep `t` und gibt CIFAR-10-Klassenlogits zurück.

Wichtige Klasse:

```python
NoisyImageClassifier
```

Wichtige Funktion:

```python
build_classifier_from_config(...)
```

---

### `src/models/unet_classifier.py`

Enthält den U-Net-artigen Classifier für Noisy-Image-Classification.

Diese Variante ist näher an der Architekturidee aus Classifier Guidance: Der Classifier arbeitet direkt auf verrauschten Bildern und nutzt Timestep Conditioning.

Wichtige Klasse:

```python
NoisyUNetClassifier
```

---

### `src/training/train_classifier.py`

Trainiert den Noisy-Image-Classifier.

Ablauf:

1. CIFAR-10 laden
2. zufällige Timesteps sampeln
3. Bilder verrauschen
4. Classifier trainieren, die richtige CIFAR-10-Klasse vorherzusagen
5. Validation Accuracy berechnen
6. Checkpoints speichern

Startbefehl:

```powershell
python -m src.training.train_classifier --config configs/classifier_cifar10.yaml
```

Resume von einem Checkpoint:

```powershell
python -m src.training.train_classifier --config configs/classifier_cifar10.yaml --resume checkpoints/last_classifier_cifar10.pt
```

---

### `src/training/train_ddpm.py`

Trainiert das eigentliche DDPM-Diffusionsmodell.

Ablauf:

1. CIFAR-10 laden
2. zufällige Timesteps sampeln
3. Bilder verrauschen
4. U-Net sagt das hinzugefügte Rauschen vorher
5. Loss ist MSE zwischen vorhergesagtem Noise und echtem Noise
6. Validation Loss berechnen
7. Checkpoints, Logs und Sample-Grids speichern

Startbefehl:

```powershell
python -m src.training.train_ddpm --config configs/ddpm_cifar10.yaml
```

Resume von einem Checkpoint:

```powershell
python -m src.training.train_ddpm --config configs/ddpm_cifar10.yaml --resume checkpoints/last_ddpm_cifar10.pt
```

---

### `src/sampling/sample_guided.py`

Erzeugt Bilder mit dem trainierten DDPM-Modell.

Optional wird ein Classifier geladen, um Classifier Guidance zu verwenden.

Startbefehl mit Classifier Guidance:

```powershell
python -m src.sampling.sample_guided `
  --ddpm-checkpoint checkpoints/ddpm_cifar10.pt `
  --ddpm-config configs/ddpm_cifar10.yaml `
  --classifier-checkpoint checkpoints/classifier_cifar10.pt `
  --classifier-config configs/classifier_cifar10.yaml `
  --class-labels all `
  --guidance-scales 0,1,2,4 `
  --num-images 9 `
  --clip-denoised
```

Startbefehl ohne Classifier:

```powershell
python -m src.sampling.sample_guided `
  --ddpm-checkpoint checkpoints/ddpm_cifar10.pt `
  --ddpm-config configs/ddpm_cifar10.yaml `
  --class-labels all `
  --guidance-scales 0 `
  --num-images 9 `
  --clip-denoised
```

Die erzeugten Bilder werden standardmäßig gespeichert unter:

```text
outputs/guided_samples/
```

---

## Config-Dateien

### `configs/classifier_cifar10.yaml`

Config für das Classifier-Training.

Typische Bereiche:

```yaml
seed:
data:
diffusion:
model:
training:
checkpoint:
```

Bedeutung:

- `seed`: Zufallsseed für reproduzierbare Runs
- `data`: CIFAR-10-Pfad, Batch Size, Worker, Validation Split
- `diffusion`: Noise-Schedule für das Verrauschen der Bilder
- `model`: Architektur des Classifiers
- `training`: Epochs, Learning Rate, Weight Decay, AMP, Gradient Clipping
- `checkpoint`: Speicherort und Dateiname des Classifier-Checkpoints

Start:

```powershell
python -m src.training.train_classifier --config configs/classifier_cifar10.yaml
```

---

### `configs/ddpm_cifar10.yaml`

Config für das normale DDPM-Training.

Typische Bereiche:

```yaml
seed:
data:
diffusion:
model:
training:
checkpoint:
logging:
sampling:
samples:
```

Bedeutung:

- `seed`: Zufallsseed
- `data`: CIFAR-10-Daten und Batch Size
- `diffusion`: Anzahl Timesteps und Beta-Schedule
- `model`: U-Net-Konfiguration
- `training`: Optimizer- und Trainingsparameter
- `checkpoint`: Speicherort für Checkpoints
- `logging`: Logdatei und CSV-Metriken
- `sampling`: feste Sample-Erzeugung während des Trainings
- `samples`: Ausgabeordner für Sample-Bilder

Start:

```powershell
python -m src.training.train_ddpm --config configs/ddpm_cifar10.yaml
```

---

### `configs/ddpm_cifar10_finetune.yaml`

Config für weiteres Training oder Finetuning eines DDPM-Modells.

Start mit Resume:

```powershell
python -m src.training.train_ddpm --config configs/ddpm_cifar10_finetune.yaml --resume checkpoints/last_ddpm_cifar10.pt
```

---

### `configs/guidance_sweep.yaml`

Config für Experimente mit mehreren Guidance Scales.

Diese Datei beschreibt normalerweise, welche Klassen, Seeds und Guidance-Skalen beim Sampling getestet werden sollen.

Der direkte Sampling-Befehl läuft über:

```powershell
python -m src.sampling.sample_guided ...
```

---

## Typischer Ablauf auf einem neuen Rechner

### 1. Repository klonen

```powershell
git clone <repository-url>
cd <projektordner>
```

### 2. Virtuelle Umgebung erstellen

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

### 3. Dependencies installieren

```powershell
pip install -r requirements.txt
```

### 4. Classifier trainieren

```powershell
python -m src.training.train_classifier --config configs/classifier_cifar10.yaml
```

### 5. DDPM trainieren

```powershell
python -m src.training.train_ddpm --config configs/ddpm_cifar10.yaml
```

### 6. Samples erzeugen

```powershell
python -m src.sampling.sample_guided `
  --ddpm-checkpoint checkpoints/ddpm_cifar10.pt `
  --ddpm-config configs/ddpm_cifar10.yaml `
  --classifier-checkpoint checkpoints/classifier_cifar10.pt `
  --classifier-config configs/classifier_cifar10.yaml `
  --class-labels all `
  --guidance-scales 0,1,2,4 `
  --num-images 9 `
  --clip-denoised
```

---

## Outputs

Während Training und Sampling entstehen typischerweise diese Ordner:

```text
checkpoints/
logs/
outputs/
data/
```

### `checkpoints/`

Enthält gespeicherte Modelle, z. B.:

```text
classifier_cifar10.pt
ddpm_cifar10.pt
last_ddpm_cifar10.pt
best_ddpm_cifar10.pt
```

### `logs/`

Enthält Trainingslogs und CSV-Metriken.

### `outputs/`

Enthält generierte Sample-Bilder und Vergleichsgrids.

### `data/`

Enthält CIFAR-10 nach dem Download.

---

## CIFAR-10 Klassen

```text
0 airplane
1 automobile
2 bird
3 cat
4 deer
5 dog
6 frog
7 horse
8 ship
9 truck
```

Beim Sampling können Klassen einzeln oder zusammen ausgewählt werden:

```powershell
--class-labels 0,1,3,5,8
```

oder:

```powershell
--class-labels all
```

---

## Guidance Scales

Beim Sampling legt `--guidance-scales` fest, welche Classifier-Guidance-Stärken getestet werden.

Beispiel:

```powershell
--guidance-scales 0,1,2,4
```

Dabei bedeutet:

```text
0   ohne Guidance
1   schwache Guidance
2   mittlere Guidance
4   stärkere Guidance
```

Für jede Klasse und jede Guidance Scale wird ein eigenes Grid gespeichert.
