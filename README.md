
### Minimind Repo
https://jingyaogong.github.io/minimind/


### Running and evaluating the model.

To run the model used in base repo:
```bash
git clone https://huggingface.co/jingyaogong/MiniMind2 # or https://www.modelscope.cn/models/gongjy/MiniMind2
```

```bash
python eval_llm.py --load_from ./MiniMind2
```

### Training and evaluating the model.

Datasets:
The notebook to build the dataset is added under `dataset` folder. Running the notebook will save the `pretrain_en.jsonl` and `sft_en.jsonl` under data folder. To use the original chinese dataset, please refer below.

Base repo reference: 
https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files
https://www.modelscope.cn/datasets/gongjy/minimind_dataset/file/view/master/README.md?id=68909&status=1

### Basic Run 

```bash
cd trainer
```
Run from the `trainer` as working dir cause the train_*.py files uses the relative paths.

```bash
python train_pretrain.py
```
> Perform pre-training to obtain pretrain_*.pththe output weights as pre-training values ​​(where * represents the model's dimension, which defaults to 512).

```bash 
python train_full_sft.py
```
> Perform supervised fine-tuning to obtain full_sft_*.pththe output weights as instructions for fine-tuning (where the weights fullrepresent the full parameter fine-tuning).

By default, all training processes save the parameters to a file every 100 steps ./out/***.pth(overwriting the old weight file each time).

For simplicity, only the two-stage training process is described here. For other training methods (LoRA, distillation, reinforcement learning, fine-tuning inference, etc.), please refer to the detailed explanation in the "Experiments" section of original readme [here](https://github.com/jingyaogong/minimind/blob/master/README_en.md).


### Evaluation

Ensure the model file you need to test is located in ./out/the directory. 

```bash
python eval_llm.py --weight full_sft # pretrain/dpo/ppo/grpo...
```

