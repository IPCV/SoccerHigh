# SoccerHigh: A Benchmark Dataset for Automatic Soccer Video Summarization

Repository containing the code and the dataset presented in [`SoccerHigh: A Benchmark Dataset for Automatic Soccer Video Summarization`](https://arxiv.org/pdf/2509.01439)

Paper accepted at the **8th International ACM Workshop on Multimedia Content Analysis in Sports** ([ACM MMSports 2025](http://mmsports.multimedia-computing.de/mmsports2025/index.html)), part of **ACM Multimedia 2025**, held on **October 28th, 2025 in Dublin, Ireland**.  


---

## 📋 Abstract

Video summarization aims to extract key shots from longer videos to produce concise and informative summaries. One of its most common applications is in sports, where highlight reels capture the most important moments of a game, along with notable reactions and specific contextual events. Automatic summary generation can support video editors in the sports media industry by reducing the time and effort required to identify key segments. However, the lack of publicly available datasets poses a challenge in developing robust models for sports highlight generation. In this paper, we address this gap by introducing a curated dataset for soccer video summarization, designed to serve as a benchmark for the task. The dataset includes shot boundaries for 237 matches from the Spanish, French, and Italian leagues, using broadcast footage sourced from the SoccerNet dataset. Alongside the dataset, we propose a baseline model specifically designed for this task, which achieves an F1 score of 0.3956 in the test set. Furthermore, we propose a new metric constrained by the length of each target summary, enabling a more objective evaluation of the generated content.

👉 **SoccerHigh** is the first large-scale benchmark dataset for **automatic soccer video summarization**.

---

## 📌 Content

- 💻[**Code**](https://github.com/IPCV/SoccerHigh/tree/main/code): Implementation of the baseline model and training, testing and inference pipelines.


- ⚽️[**Dataset**](https://github.com/IPCV/SoccerHigh/tree/main/dataset): Open dataset files and pre-extracted features for experiments.

---

## ⚡ Quick Start

To run the baseline model, check the code [README](https://github.com/IPCV/SoccerHigh/tree/main/code/README.md).  

---

## 📖 Citation

If you use this code or dataset in your research, please cite:

```bibtex
@inproceedings{10.1145/3728423.3759410,
  author = {D\'{\i}az-Juan, Artur and Ballester, Coloma and Haro, Gloria},
  title = {SoccerHigh: A Benchmark Dataset for Automatic Soccer Video Summarization},
  year = {2025},
  isbn = {9798400711985},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3728423.3759410},
  doi = {10.1145/3728423.3759410},
  booktitle = {Proceedings of the 8th International ACM Workshop on Multimedia Content Analysis in Sports},
  pages = {121–130},
  numpages = {10},
  location = {Dublin, Ireland},
  series = {MMSports '25}
}
```
---

## 🛡️ License

This dataset is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) license.

---

## 🙏 Acknowledgements

Funded by the **European Union (GA 101119800 - EMERALD)**.

<p align="center">
  <img src="https://european-union.europa.eu/sites/default/files/styles/oe_theme_medium_no_crop/public/2025-04/LOGO%20CE_RGB_MUTE_POS.png?itok=2xxOY8gh" alt="EU Logo" width="200"/>
  &nbsp;&nbsp;&nbsp;
  <img src="https://www.upf.edu/documents/279393550/290803695/emerald.png/45473326-80ed-2da6-207b-c3acc6f5e2b9?t=1725622949880" alt="EMERALD Logo" width="300"/>
</p>