# SoccerHigh Dataset ⚽️

SoccerHigh is a benchmark dataset designed for automatic soccer video summarization. It follows the [SoccerNet](https://www.soccer-net.org/) directories structure of `league -> season -> game`. 
For each game, a `Labels-summary.json` file is provided, which extends SoccerNet’s `Labels-v2.json` with new fields tailored to highlight detection and summary generation.

At the dataset root, three split files — `train.txt`, `validation.txt`, and `test.txt` — define the games included in each subset used in the paper.

⚠️ Note: This repository contains only the annotation files.
To access the complete dataset (features + annotations), download it [here](https://drive.google.com/file/d/1lMVbjCAvbYpoLJ6F2tnZgbfhvGWHWhpf/view?usp=sharing).

---

## 📂 Dataset Structure

The dataset is organized hierarchically:

```text
dataset/
├── train.txt
├── validation.txt
├── test.txt
├── <league>/
│   ├── <season>/
│   │   ├── <game>/
│   │   │   ├── 1_HQ_224p_VideoMAEv2_Giant_K710_1408.npy
│   │   │   ├── 1_HQ_224p_VideoMAEv2_SmallFromGiant_K710_384.npy
│   │   │   ├── 1_intervals.srt
│   │   │   ├── 2_HQ_224p_VideoMAEv2_Giant_K710_1408.npy
│   │   │   ├── 2_HQ_224p_VideoMAEv2_SmallFromGiant_K710_384.npy
│   │   │   ├── 2_intervals.srt
│   │   │   ├── Labels-summary.json
```

### 📝 Files per game

- **`1_HQ_224p_VideoMAEv2_Giant_K710_1408.npy`**  
  Frame features from the game's first half, extracted with the [VideoMAEv2-Giant](https://huggingface.co/OpenGVLab/VideoMAE2/tree/main/mae-g) backbone.

- **`1_HQ_224p_VideoMAEv2_SmallFromGiant_K710_384.npy`**  
  Frame features from the game's first half, extracted with the [VideoMAEv2-SmallFromGiant](https://huggingface.co/OpenGVLab/VideoMAE2/tree/main/distill) backbone.

- **`1_intervals.srt`**  
  Annotated temporal segments for the first half in `.srt` format.  

- **`2_HQ_224p_VideoMAEv2_Giant_K710_1408.npy`**  
  Frame features from the game's second half, extracted with the [VideoMAEv2-Giant](https://huggingface.co/OpenGVLab/VideoMAE2/tree/main/mae-g) backbone.

- **`2_HQ_224p_VideoMAEv2_SmallFromGiant_K710_384.npy`**  
  Frame features from the game's second half, extracted with the [VideoMAEv2-SmallFromGiant](https://huggingface.co/OpenGVLab/VideoMAE2/tree/main/distill) backbone.

- **`2_intervals.srt`**  
  Annotated temporal segments for the second half in `.srt` format.

- **`Labels-summary.json`**  
  Metadata describing the game (teams, date, score, video URLs, and annotations).

---

## 🏷️ Labels Metadata

| Key | Description |
|-----|-------------|
| `UrlLocal` | Path of the game directory. |
| `UrlYoutube` | URL to the **official YouTube video** of the game summary. |
| `annotations` | List of annotated segments, each with start and end times (`half - hh:mm:ss.ms`) extracted from SoccerNet `*_HQ.mkv` and a segment ID. |
| `gameHomeTeam` | Name of the home team. |
| `gameAwayTeam` | Name of the away team. |
| `gameDate` | Date and time of the game (`DD/MM/YYYY - HH:MM`). |
| `gameScore` | Final score of the game (`home_score - away_score`). |
| `summaryLength` | Duration of the summary annotations in `mm:ss`. |

---

## ⏱️ Annotation Format

The `.srt` files define summary segments:

- **First line**: ignored (temporal index).  
- **Second line**: `start_time --> end_time` with millisecond precision.  
- **Third line**: `Segment <id>` (annotation label).

### 🧾 Example
Supposing a first half segment:

```text
2
00:09:50,500 --> 00:10:01,000
Segment 3
```

In `JSON`, this becomes:

```json
{
    "id": 3,
    "start": "1 - 00:09:50.500",
    "end": "1 - 00:10:01.000"
}
```
---

## 📖 Citation

If you use this dataset in your research, please cite:

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

This dataset is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) license.