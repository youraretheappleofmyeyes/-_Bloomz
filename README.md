# 学习目标：

`基于LLaMA-Factory 实现Bloomz-7B1医疗模型微调流程

## 微调大模型的基本思路

 - [ ] 第一阶段：(Continue PreTraining)增量预训练
	一般垂直大模型是基于通用大模型进行二次的开发。为了给模型注入领域知识，就需要用领域内的语料进行继续的预训练。
 - [ ] 第一阶段：SFT( Supervised Finetuning,有监督微调)
 	通过SFT可以激发大模型理解领域内的各种问题并进行回答的能力
 - [ ] 第三阶段：直接偏好优化DPO
 	1）：RLHF(奖励建模、强化学习训练): 通过RLHF可以让大模型的回答对齐人们的偏好，比如行文的风格。
 	2）：DPO(直接偏好优化) 
 - [ ] 量化

 由于大模型的参数量巨大，在解码阶段需要占用大量的显存资源，因而在实际应用中的部署代价非常高。通常使用模型量化（Model Quantization），来减少大模型的显存占用，从而使得能够在资源有限的环境下使用大模型。
 - [ ] 部署

---

# 模型选择：
![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/09db8c5704164dc88c95d611adac3a8d.jpeg#pic_center)

 - [ ] 显存计算：
 
 ![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/698dd29eaec14a469ba31fdc4ede27cd.jpeg#pic_center)

 - [ ] 显存估计：
全量训练所需显存一般来说是同样规模LLM推理的4-5倍。
 
 ![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/2320937adcc44a758d356c02204bd311.jpeg#pic_center)

根据自己的资源和领域选择开源的大模型，最好是训练预料中包含该领域的数据。比如BLOOMZ 模型系列使用了PILE语料库进行训练，该语料库包含各种医学文本，包括PubMed Central 和 PubMed Abstracts等。BLOOMZ模型的医学知识体系比较丰富。
## 开源数据集整理：
|SFT数据集|  |	
|--|--|
| HuatuoGPT-sft-data-v1 | [地址](https://huggingface.co/datasets/FreedomIntelligence/HuatuoGPT-sft-data-v1) |
|Chinese-medical-dialogue-data|[中文医疗科室问答数据](https://github.com/Toyhom/Chinese-medical-dialogue-data)|
| Huatuo26M-Lite | [Huatuo-Lite](https://huggingface.co/datasets/FreedomIntelligence/Huatuo26M-Lite) |
|DISC-Med-SFT|[中文医疗多轮对话数据](https://huggingface.co/datasets/Flmc/DISC-Med-SFT)|
|sft-20k.json|[疾病、药品知识问答](https://github.com/CMKRG/QiZhenGPT/blob/main/data/train/sft-20k.json)|
|ShenNong_TCM_Dataset|[中文医疗问答数据](https://huggingface.co/datasets/michaelwzhu/ShenNong_TCM_Dataset)|
|Medical-Dialogue-System|[中/英医疗多轮对话数据集](https://github.com/UCSD-AI4H/Medical-Dialogue-System)|
|DPO||
|dpo_zh_500.jsonl|[地址](https://github.com/shibing624/MedicalGPT/tree/main/data/reward)|
|DPO-En-Zh-20k-Preference|[地址](https://huggingface.co/datasets/shibing624/DPO-En-Zh-20k-Preference)|

## 微调数据集配比：
参考：
贝壳团队做的关于大模型在垂域上继续finetune时的一些实验情况
[ChatHome: Development and Evaluation of a Domain-Specific Language Model for Home Renovation](https://arxiv.org/abs/2307.15290)

预训练数据配比：

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/abc56dadd48343cfae8bd6d55063e306.png#pic_center)


SFT数据配比：

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/64c2707976d64a93bdf04b4fb1aa5652.png#pic_center)



## 高质量微调数据集制作：
参考：
1. [IFD](https://github.com/MingLiiii/Cherry_LLM)

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/36ee75acca674ed380d4abebbe809061.png#pic_center)

2. [MODS](https://github.com/CASIA-LM/MoDS)

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/4210116de0e14b43bdd59b2d4d732a91.png#pic_center)

3. [CaR](https://github.com/IronBeliever/CaR/tree/main)

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/b65b5e164a0c44b381aa83399e6ea03f.png#pic_center)

4. [GPT-4/ChatGPT模型蒸馏医学数据](https://huggingface.co/spaces/wangrongsheng/DataMaker)

---

# 模型微调：

 - [ ] 增量预训练：
 
 

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train \
    --stage pt \
    --do_train True \
    --model_name_or_path bigscience/bloom-7b1 \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --template default \
    --flash_attn auto \
    --dataset_dir data \
    --dataset medical_book_zh,train_encyclopedia \
    --cutoff_len 1024 \
    --learning_rate 5e-05 \
    --num_train_epochs 3.0 \
    --max_samples 100000 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 100 \
    --warmup_steps 0 \
    --optim adamw_torch \
    --packing True \
    --report_to none \
    --output_dir saves/BLOOM-7B1/lora/train_2024-06-05-11-42-15 \
    --bf16 True \
    --plot_loss True \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0 \
    --lora_target query_key_value \
    --val_size 0.05 \
    --evaluation_strategy steps \
    --eval_steps 100 \
    --per_device_eval_batch_size 2
```

 - [ ] SFT：
 

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train \
    --stage sft \
    --do_train \
    --model_name_or_path /root/autodl-tmp/bloomz-7b1/ \
    --dataset firefly,identity \
    --dataset_dir ./data \
    --template default \
    --finetuning_type lora \
    --lora_target query_key_value \
    --output_dir /root/autodl-tmp/LLaMA-Factory/saves/bloomz-7b1/lora_sft \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --logging_steps 2000 \
    --warmup_steps 1000 \
    --save_steps 2000 \
    --eval_steps 2000 \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --learning_rate 5e-5 \
    --num_train_epochs 1.0 \
    --val_size 0.1 \
    --plot_loss \
    --bf16 \
    --max_samples 20000
```

 - [ ] 合并LoRA：
 

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli export examples/merge_lora/llama3_lora_sft.yaml
```

 - [ ] DPO训练：
 需要两倍的显存
 

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train \
    --stage dpo \
    --do_train True \
    --model_name_or_path /root/autodl-tmp/bloomz-7b1 \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --template default \
    --flash_attn auto \
    --dataset_dir data \
    --dataset dpo_en_10k,dpo_zh_10k \
    --cutoff_len 1024 \
    --learning_rate 5e-05 \
    --num_train_epochs 1.0 \
    --max_samples 30000 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 100 \
    --save_steps 100 \
    --warmup_steps 50 \
    --optim adamw_torch \
    --output_dir saves/bloomz-7b1/lora/dpo \
    --bf16 True \
    --plot_loss True \
    --lora_rank 4 \
    --lora_alpha 8 \
    --lora_dropout 0 \
    --lora_target query_key_value \
    --pref_beta 0.1 \
    --pref_ftx 0 \
    --val_size 0.01 \
    --per_device_eval_batch_size 1 \
    --evaluation_strategy steps \
    --pref_loss sigmoid 
```

 - [ ] AutoGPTQ 量化模型：
 

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli export \
    --model_name_or_path /root/autodl-tmp/bloomz-7b1 \
    --template default \
    --export_dir models/bloomz-7b1_gptq \
    --export_quantization_bit 4 \
    --export_quantization_dataset data/c4_demo.json \
    --export_size 2 \
    --export_device cpu \
    --export_legacy_format false \
    --export_quantization_maxlen 512 \
```

 - [ ] vLLM部署推理API：
 
 vLLM 专门针对解码效率进行了大量优化。vLLM 通过对键值缓存进行分页存储，并结合 PagedAttention 技术，显著提升了解码时注意力的计算效率。同时它还支持多种解码策略，引入了批次管理优化技术，能够很好地在真实场景中部署大模型。
 此外，vLLM 也可以像 ChatGPT 的网页端一样启动网络服务功能，将模型一直挂载运行。

```bash
CUDA_VISIBLE_DEVICES=0 API_PORT=8000 llamafactory-cli api examples/inference/llama3_vllm.yaml
```
 - [ ] 测试API：
 

```bash
curl -X 'POST' \
    'localhost:8000/v1/chat/completions' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
    "model": "string",
    "messages": [
      {
        "role": "user",
        "content": "我身体疼痛，想请问该使用什么方剂?要求:1.请考成所有症状。2.调据中医知识输出一步步的推理过程"
      }
    ],
    "temperature": 0,
    "top_p": 0,
    "max_new_tokens": 0,
    "stream": false    
  }'
```


项目参考：
[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory/tree/main)
 - 关于LLaMA-Factory的安装使用，以及数据格式和微调流程脚本参数使用查询上述项目。

---

# 微调效果：


微调后模型效果

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/4e755061a0b949cd9dfbae1e7106dce7.jpeg#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/f793c577fd8b493c8af1064be288cd8c.jpeg#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/da6a44b768cc44d19a8f903cf4d880ce.jpeg#pic_center)


Bloomz-7B

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/8aaf511623704d58953b79867c68da18.jpeg#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/a37f31ddd6164edd98477489ca18f29f.jpeg#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/f1b64193d3824af4a8d8643b885a3bf9.jpeg#pic_center)
