# ReX-GoT
The implementation of our AAAI 2024 paper "Reverse Multi-Choice Dialogue Commonsense Inference with Graph-of-Thought"

## Abstract
With the proliferation of dialogic data across the Internet, the Dialogue Commonsense Multi-choice Question Answering (DC-MCQ) task has emerged as a response to the challenge of comprehending user queries and intentions.
Although prevailing methodologies exhibit effectiveness in addressing single-choice questions, they encounter difficulties in handling multi-choice queries due to the heightened intricacy and informational density. 
In this paper, inspired by the human cognitive process of progressively excluding options, we propose a three-step Reverse Exclusion Graph-of-Thought (ReX-GoT) framework, including Option Exclusion, Error Analysis, and Combine Information.
Specifically, our ReX-GoT mimics human reasoning by gradually excluding irrelevant options and learning the reasons for option errors to choose the optimal path of the GoT and ultimately infer the correct answer.
By progressively integrating intricate clues, our method effectively reduces the difficulty of multi-choice reasoning and provides a novel solution for DC-MCQ.
Extensive experiments on the CICERO and CICERO_v2 datasets validate the significant improvement of our approach on DC-MCQ task.
On zero-shot setting, our model outperform the best baseline by 17.67% in terms of F1 score for the multi-choice task.
Most strikingly, our GPT3.5-based ReX-GoT framework achieves a remarkable 39.44% increase in F1 score.


## Setup
- **Build environment**
```
cd ReX-GoT
# use anaconda to build environment 
conda create -n ReX-GoT python=3.10
conda activate ReX-GoT
# install packages
pip install transformers==4.11.3
pip install numpy==1.23.3
```

## Quick Start

```
python main.py
```

## BibTeX 

If you find ReX-GoT both interesting and helpful, please consider citing us in your research or publications:

```bibtex
@inproceedings{zheng2024reverse,
  title={Reverse Multi-Choice Dialogue Commonsense Inference with Graph-of-Thought},
  author={Zheng, Li and Fei, Hao and Li, Fei and Li, Bobo and Liao, Lizi and Ji, Donghong and Teng, Chong},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence (AAAI 24)},
  year={2024}
}
```



