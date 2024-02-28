# :whale: Model-Compression-Paper-Note
Model compression paper note including pruning, distillation.
Feel free to drop a comment if you want to address any issues, and please don't hesitate to correct me if I've misunderstood the papers. 
Hope we all find our own path in the research life. :seedling:

- **PERP: Rethinking the Prune-Retrain Paradigm in the Era of LLMs**
    - Author: **Max Zimmer, Megi Andoni,** etc
    - Institute: Zuse Institute Berlin, Germany and Technische Universita ̈t Berlin, Germany
    - Link: https://arxiv.org/abs/2312.15230
    - Code: Not available
    - Pub: 2024
    - Tag:  `Unstructured Pruning` `Efficient Retraining`
    - Summary: After apply unstrctured pruning like WANDA, SparseGPT, we can use their method PERP to only retrain BN-Recalibrate, bias, BN-parameters and last linear layer with LoRA among all layers of model and achieve good performance. Retrain 0.27% - 0.35% OPT parameter in 1000 iteration is sufficient to recover performance (better than SparseGPT, WANDA)
    - Comment: Instead of retraining all parameters in model, retrain part of it to get benefit of memory, speed and accuracy is brilliant. But they did not give the specific number of retraining time, only say within minutes.

- **SliceGPT: Compress Large Language Models by Deleting Rows and Columns**
    - Author: Saleh Ashkboos, Maximilian L. Croci, etc
    - Institute: Microsoft, ETH Zurich
    - Link: https://arc.net/l/quote/ztditidv
    - Code: https://github.com/microsoft/TransformerCompression
    - Pub: ICLR 2024
    - Tag:  `Unstructured Pruning` `No Retraining`
    - Summary: Compute PCA on each layer’s input X and use computational invariance to add eigen-vectors Q to each transformer block weight then apply deletion on each weight’s row or columns. Perform better than SparseGPT and can with or w/o retraining. Prune 30% while maintain > 90% of performance.
    - Comment: It is more brilliant method to use PCA than DNN pruning in JSA 2021, but it still need 3.5 hours on LLAMA-2 70B model on single H100 GPU to compute PCA.

- **SparseGPT: Massive Language Models Can be Accurately Pruned in One-Shot**
    - Author: **Elias Frantar, Dan Alistarh**
    - Institute: Institute of Science and Technology Austria (ISTA), Neural Magic Inc.
    - Link: https://arxiv.org/abs/2301.00774
    - Code: https://github.com/IST-DASLab/sparsegpt
    - Pub: ICML 2023
    - Tag:  `Unstructured Pruning` `One-shot Pruning` `No Retraining` `LLM`
    - Summary: Reuse hessian inverse matrix to prune each weight locally, perserve input-output relationship. It maintain row-wise pruning ratio and adaptively choose the mask while running the reconstruction to maintain performance. It can prune 175B model in 4 hours on single NVIDIA A100 GPU with little accuracy degradation.
    - Comment: The math inside this paper is quite complex. Better to know Row-Hessian challenge first before reading it.

- **A Simple and Effective Pruning Approach for Large Language Models (WANDA)**
    - Author: Mingjie Sun, Zhuang Liu, Anna Bair, J. Zico Kolter
    - Institute: Carnegie Mellon University, Meta AI Research, Bosch Center for AI
    - Link: https://arxiv.org/abs/2306.11695
    - Code: https://github.com/locuslab/wanda
    - Pub: ICLR 2024
    - Tag:  `Unstructured Pruning` `LLM` `No Retraining`
    - Summary: Compute efficient local pruning using metric of the product of parameter magnitude and the norm of the corresponding input activation. The speed is 300 times faster than SparseGPT while attaining good performance tradeoff.
    - Comment: Incredibly simple but effective, it suggest the new metric of pruning.

- **LLM-Pruner: On the Structural Pruning of Large Language Models**
    - Author: Xinyin Ma, Gongfan Fang, Xinchao Wang
    - Institute: National University of Singapore
    - Link: https://arxiv.org/abs/2305.11627
    - Code: https://github.com/horseee/LLM-Pruner
    - Pub: NeurIPS 2023
    - Tag:  `Structured Pruning` `LLM` `Task-agnostic`
    - Summary: Identify each structure group using in and out degree of neurons and estimate each group’s importance with gradient and hessian matrix. After pruning, use LoRA for recovery.
    - Comment: Compress LLaMA-7B model in 3 hour on single GPU but not using other promising baseline. The part of finding each group is worth to consider.

- **ZipLM: Inference-Aware Structured Pruning of Language Models**
    - Author: Eldar Kurtic, Elias Frantar, Dan Alistarh
    - Institute: IST Austria, Neural Magic
    - Link: https://arxiv.org/abs/2302.04089
    - Code: https://github.com/ist-daslab/ziplm
    - Pub: NeurIPS 2023
    - Tag: `Structured Pruning` `Iterative Pruning` `Unstructured Pruning`
    - Summary: Iteravely prune structure like attention heads or FFN intermediate neurons. In each pruning step, it prune one sub-structure and update corresponding weight with the help of hessian matrix then update hessian matrix w.r.t new mask. Use layer-wise token distillation as recovery.
    - Comment: It can perfom the entire family of compressed models at one time that is different from other approach which needs one process for one compressed model.

- **Language Model Compression with Weighted Low-rank Factorization**
    - Author: Yen-Chang Hsu, Ting Hua, etc
    - Institute: Samsung Research America , Northeastern University
    - Link: https://arxiv.org/abs/2207.00112
    - Code: Not available
    - Pub: ICLR 2022
    - Tag: `Structured Pruning` `SVD`
    - Summary: They found that after SVD, some vectors with smaller singular values have bigger impact on task accuracy than vectors with bigger singular value. Therefore, they use SVD on weights multiplied with row-wise fisher information to compress model.
    - Comment: The property they found in SVD of task is interesting, that’s why standard SVD does not perform well on model compression when benchmarking.

- **A Fast Post-Training Pruning Framework for Transformers**
    - Author: Woosuk Kwon, Sehoon Kim, etc
    - Institute: UC Berkeley, etc.
    - Link: https://arxiv.org/abs/2204.09656
    - Code: https://github.com/WoosukKwon/retraining-free-pruning
    - Pub: NeurIPS 2022
    - Tag: `Structured Pruning` `No Retraining`
    - Summary: Prune transformer heads and filters based on fisher information approximation. View layer reconstruction as linear least square problem to adjust mask value in a short time so that it can prune Transformer in less than 3 minutes on single GPU.
    - Comment: To the best of my knowledge, this is the first paper to reduce transformer-based structured pruning time from hours to minutes.

- **Gradient-Free Structured Pruning with Unlabeled Data**
    - Author: Azade Nova, Hanjun Dai, Dale Schuurmans
    - Institute: Google DeepMind
    - Link: https://arxiv.org/abs/2303.04185
    - Code: Not available
    - Pub: ICML 2023
    - Tag: `Structured Pruning` `No Retraining`
    - Summary: A structured pruning method that does not require retraining nor labeled data. Introduce kernelized convex masking on FFN’s weight to decide the importance of each neuron.
    - Comment: This paper is based on “A Fast Post-Training Pruning Framework for Transformers” that the only difference is they use KCM and activated output to decide importance and get rid of mask rearrangement. The paper requires lots of math knowledge to understand KCM.

- **Accurate Retraining-free Pruning for Pretrained Encoder-based Language Models**
    - Author: Seungcheol Park, Hojun Choi, U Kang
    - Institute: Seoul National University
    - Link: https://openreview.net/forum?id=s2NjWfaYdZ
    - Code: Supplementary material of above link
    - Pub: ICLR 2024
    - Tag: `Structured Pruning` `No Retraining`
    - Summary: Use KL divergence of student’s and teacher’s logits as predictive knowledge and values of heads and activation output as representative knowledge. Introduce knowledge-preserving pruning (KPP) that prune and adjust weight while taking one structure once a time from input layer to output layer to avoid accuracy degradation.
    - Comment: KPP gives their benchmark metric good scores while pruning BERT within 10 minutes. It performs way more better than its series (A fast post-training pruning framework, KCM).

### Acknowledgement
Thanks for [Awesome-LLM-Prune](https://github.com/pprp/Awesome-LLM-Prune) that inspires me for this note, the template is based on their work.
