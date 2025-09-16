# SoccerHigh Baseline Model 💻

This code provides the implementation of the Baseline Model presented in the [paper](https://arxiv.org/pdf/2509.01439).

---

## 📂 Code Structure

The code for the baseline model is organized as follows:

```text
code/
├── augmentations/           
│   ├── mixup.py                          # Mixup augmentation implementation
├── configs/                              # Hydra configuration files
|   ├── datamodules/
│   |   ├── default.yaml
|   ├── datasets/
│   |   ├── default.yaml
|   ├── models/
│   |   ├── default.yaml
|   ├── scripts/
│   |   ├── trim_summary.yaml
|   ├── trainer/
│   |   ├── default.yaml
|   ├── predict.yaml
|   ├── test.yaml
│   ├── train.yaml
├── datamodules/             
│   ├── datamodule.py                     # Data module handling datasets and dataloaders
├── datasets/                
│   ├── soccernet_games.py                # Handles SoccerNet game-level data
│   ├── soccernet_summarization.py        # Handles video summarization data
│   ├── utils.py                          # Utility functions for dataset processing
├── evaluation/              
│   ├── evaluate.py                       # Compute metrics and summarize model performance
├── inference/               
│   ├── inference.py                      # Key shot selection for inference
│   ├── utils.py                          # Helper functions for inference
├── models/                  
│   ├── classifier.py                     # Baseline model implementation
│   ├── dino.py                           # DINO implementation
│   ├── heads.py                          # Model heads definitions
│   ├── transnetv2.py                     # TransNetv2 implementation
├── scripts/                 
│   ├── trimm_summary.py                  # Script to compute new summary annotations
├── weights/
│   ├── Baseline_VideoMAEv2-Giant.ckpt    # Baseline checkpoint using the VideoMAEv2 giant backbone 
│   ├── Baseline_VideoMAEv2-Small.ckpt    # Baseline checkpoint using the VideoMAEv2 small backbone       
├── predict.py                            # Inference script
├── test.py                               # Testing script
├── train.py                              # Training script
```

---

## ⚙️ Preparation

Before running the code, set up the environment and the dataset path.

**1. Create the Conda environment**:

We provide an `environment.yml` file to install all dependencies:

```bash
conda env create -f environment.yml
conda activate soccerhigh
```

**2. Create a symbolic link to the dataset**:

The code expects a symbolic link named data pointing to the dataset directory, which should contain the files from [SoccerHigh](https://github.com/IPCV/SoccerHigh/tree/main/dataset).

```bash
ln -s /path/to/your/dataset data
```

Replace `/path/to/your/dataset` with the actual path to your dataset folder.

---


## 🚀 Usage

The code is designed using [PyTorch Lightning](https://lightning.ai/) for cleaner training loops and easier scaling.
All configurations are managed using a [Hydra](https://hydra.cc/), making it easy to modify parameters.

### 1. Training

Train the model using the default configuration:

```bash
python3 train.py
```

### 2. Testing/Evaluation

Evaluate a trained model checkpoint:

```bash
python3 test.py checkpoint='weights/checkpoint.ckpt'
```

Replace `checkpoint.ckpt` with the actual checkpoint name. Default: [`Baseline_VideoMAEv2-Giant.ckpt`](https://github.com/IPCV/SoccerHigh/blob/main/code/weights/Baseline_VideoMAEv2-Giant.ckpt)

### 3. Inference

Run inference on new games:

```bash
python3 predict.py checkpoint='weights/checkpoint.ckpt' output_path='file.json' datamodule.predict.dataset.game_list='/data/games.txt'
```

Replace `checkpoint.ckpt`, `file.json` and `games.txt` with the actual file names. Default: [`Baseline_VideoMAEv2-Giant.ckpt`](https://github.com/IPCV/SoccerHigh/blob/main/code/weights/Baseline_VideoMAEv2-Giant.ckpt), `output.json` and `test.txt`

---

## 🔗 Notes

- Make sure the symbolic link `data` points correctly to your dataset.

- All configuration files are in `configs/` for easy experiment management.

- Logs and checkpoints will be saved in the default generated directory `lightning_logs/`.

---

## 📖 Citation

If you use this code for a scientific publication, please reference the original paper:

```bibtex
@article{diaz2025soccerhigh,
  title={SoccerHigh: A Benchmark Dataset for Automatic Soccer Video Summarization},
  author={D{\'\i}az-Juan, Artur and Ballester, Coloma and Haro, Gloria},
  journal={arXiv preprint arXiv:2509.01439},
  year={2025}
}
```
---

## 🛡️ License

This code is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) license.