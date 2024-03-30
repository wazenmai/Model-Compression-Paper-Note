# :whale: Paper Note
Feel free to drop a comment if you want to address any issues, and please don't hesitate to correct me if I've misunderstood the papers. 
Hope we all find our own path in the research life. :seedling:

1. [Model-Compression-Paper-Note](#model-compression)
2. [Mixture-of-Experts-Paper-Note](#moe)

## <a name="model-compression"></a>Model-Compression-Paper-Note
Model compression paper note including pruning, distillation.

- **Merge, Then Compress: Demystify Efficient SMoE with Hints from Its Routing Policy**
    - Author: Pingzhi Li, Zhenyu Zhang, etc.
    - Institute: The University of North Carolina at Chapel Hill, The University of Texas at Austin, etc.
    - Link: https://arc.net/l/quote/kogdwhcf
    - Code: https://github.com/UNITES-Lab/MC-SMoE
    - Pub: ICLR 2024 Spotlight
    - Tag: `MoE` `Structured Pruning`
    - Summary: This paper proposed a method to first repermute expert weights then using the similarity of each experts to group and merge experts. After merging experts, they found a low-dimensionality in weights and further compress them to reduce memory usage and experts redundancy in SMoE. They also provide detailed ablation study of each methods.
    - Comment: Great paper with nice observations and ablation studies.

    - **Merging Experts into One: Improving Computational Efficiency of Mixture of Experts**
    - Author: **Shwai He, Run-Ze Fan, etc.**
    - Institute: University of Maryland, College Park, The University of Sydney, etc.
    - Link: https://arxiv.org/abs/2310.09832
    - Code: https://github.com/Shwai-He/MEO
    - Pub: EMNLP 2023 Oral
    - Tag: `MoE`
    - Summary: It proposed to merge selected experts while training and inference by adding experts weights into single weight and forward it. It boost training efficiency and performs better than MoE.
    - Comment: The expert structure is single linear layer with an activation function, so if the expert is normal FFN (linear-act-linear), we can’t just simply summarize it.

- **Not All Experts are Equal: Efficient Expert Pruning and Skipping for Mixture-of-Experts Large Language Models**
    - Author: Xudong Lu, Qi Liu, etc.
    - Institute: CUHK MMlab, Shanghai Jiao Tong University, Shanghai Artificial Intelligence Laboratory
    - Link: https://arc.net/l/quote/iqwsjght
    - Code: https://github.com/Lucky-Lance/Expert_Sparsity
    - Pub: ACL 2024 submission
    - Tag: `MoE` `Structure Pruning` `LLM` `No Retraining`
    - Summary: Propose expert-level pruning and dynamic expert-skipping methods for MoE LLM on Mixtral 8x7B. It can effectively reduce memory usage and increase inference speed of MoE models.
    - Comment: Simple but effective pruning method for MoE, while the method is still naive and did not compare to “Task-Specific Expert Pruning for Sparse Mixture-of-Experts” in task-specific tasks, probably due to the retraining in that paper.

- **Task-Specific Expert Pruning for Sparse Mixture-of-Experts**
    - Author: Tianyu Chen, Shaohan Huang, etc.
    - Institute: BDBC, Beihang University, China, SKLSDE Lab, Beihang University, China, etc.
    - Link: https://arxiv.org/abs/2206.00277
    - Code: Not available
    - Pub: Arxiv 2022
    - Tag: `MoE` `MoE pruning`
    - Summary: A task-specific MoE model pruning that progressively drop the experts below certain threshold and only left one expert after half of training step. It shows that single-expert fine-tuned model can even beats MoE fine-tuned model on some tasks like MRPC.
    - Comment: The criterion they use is naive and the single expert model’s inference speed is still slower than dense model, which suggest that there’s still ways to improve in this area.

- **Parameter-Efficient Mixture-of-Experts Architecture for Pre-trained Language Models**
    - Author: Ze-Feng Gao, Peiyu Liu, etc.
    - Institute: Gaoling School of Artificial Intelligence, Renmin University of China, Department of Physics, Renmin University of China, etc.
    - Link: https://arxiv.org/abs/2203.01104
    - Code: https://github.com/RUCAIBox/MPOE
    - Pub: Coling 2022
    - Tag: `MoE`
    - Summary: They use MPO to reduce expert weight matrix to one central tensor and four auxiliary tensors while sharing central tensor among experts in same layer. By doing this, they can effectively reduce MoE parameters to 0.19 of original scale. They also propose gradient masking so that central tensor won’t get too frequently update. Experiment results show that MPOE beats other baseline of GPT-2 and T5 on GLUE and NLG benchmark.
    - Comment: It’s the first time to use MPO decomposition on MoE architecture, and it really performs well, while authors did not provide the training efficiency analysis.

- **ShortGPT: Layers in Large Language Models are More Redundant Than You Expect**
    - Author: Xin Men, Mingyu Xu, etc.
    - Institute: Baichuan Inc., etc.
    - Link: https://arxiv.org/abs/2403.03853
    - Code: Not available
    - Pub: Arxiv
    - Tag: `Structured Pruning` `No Retraining` `LLM`
    - Summary: It proposes a simple metric to measure the difference of layer’s input and output and does simple layer removal on LLM. The performance is better than LLM-Pruner, SliceGPT and LaCo, revealing a significant redundancy on depth-level of LLM.
    - Comment: The method is so simple and they even compare the paper which posted on arxiv only two weeks before them. The redundancy in LLM is also a good topic to research.

- **LaCo: Large Language Model Pruning via Layer Collapse**
    - Author: Yifei Yang, Zouying Cao, Hai Zhao
    - Institute: Shanghai Jiao Tong University
    - Link: https://paperswithcode.com/paper/laco-large-language-model-pruning-via-layer
    - Code: https://github.com/deamme/laco
    - Pub: Arxiv
    - Tag: `Structured Pruning` `LLM`
    - Summary: It proposed a simple and effective method to collapse LLM layers with similarity beyond a threshold T, the collapse is just simply add the parameters of collapsed layers together. It achieved SOTA structured LLM pruning performance, which beats LLM-Pruner and SliceGPT.
    - Comment: It open the era of layer pruning in LLM, while the work after it about two weeks, ShortGPT even surpass its performance with similar layer pruning method.


- **ONE-SHOT SENSITIVITY-AWARE MIXED SPARSITY PRUNING FOR LARGE LANGUAGE MODELS**
    - Author: *Hang Shao, Bei Liu, Yanmin Qia*
    - Institute: Auditory Cognition and Computational Acoustics Lab, etc.
    - Link: https://arxiv.org/abs/2310.09499
    - Code: Not available
    - Pub: ICASSP 2024
    - Tag:  `Unstructured Pruning` `No Retraining` `LLM`
    - Summary: They combine OBD and OBS’s saliency score as their metric for evaluating the sensitivity of weights. They also compute the sensitivity level of each weight to give them different sparsity ratio s.t. the overall sparsity is satisfied. The performance is better than SparseGPT and works well with quantization.
    - Comment: They use lots of approximation when computing the sensitivity level of each weight matrices, but give intersting insights on the property of each layer and each weight matrices. (refer to Fig.1, Fig.2)

- **THE LLM SURGEON**
    - Author: Tycho F.A. van der Ouderaa, Markus Nagel, etc.
    - Institute: Imperial College London , Qualcomm AI Research, etc.
    - Link: https://openreview.net/pdf?id=DYIIRgwg2i
    - Code: Not available
    - Pub: ICLR 2024
    - Tag: `Structured Pruning` `Unstructured Pruning` `LLM` 
    - Summary: Propose algorithm for unstructured (SOTA), semi-structured (SOTA) and structured compression of LLMs. The algorithm can dynamically allocate weight across layers with global threshold. Utilise Fisher approximation with KFAC approximation to expand the curvature of the loss landscape, consider correlated weight, “prune and update” the model in multiple shots.
    - Comment: The math they use is very complex, I spend a few days to fully understand the whole background. While the method they propose is very strong to deal with any pruning condition of LLMs.

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

## <a name="moe"></a>Mixture-of-Experts-Paper-Note

- **Scaling Vision with Sparse Mixture of Experts**
    - Author: Carlos Riquelme, Joan Puigcerver, etc
    - Institute: Google Brain
    - Link: https://arxiv.org/abs/2106.05974
    - Code: https://github.com/google-research/vmoe
    - Pub: NeurIPS 2021
    - Tag: `MoE`
    - Summary: The first work to apply MoE on Vision Transformer. They also propose batched prioritized routing to further improve inference speed and provide lots of analysis on V-MoE.
    - Comment: The analysis on V-MoE is thorough with great visualization which helps people to know more about the attribute of MoE routing and experts.

- **AdaMV-MoE: Adaptive Multi-Task Vision Mixture-of-Experts**
    - Author: Tianlong Chen, Xuxi Chen, etc.
    - Institute: University of Texas at Austin, Apple, Google
    - Link: https://www.semanticscholar.org/paper/a4382a9b1edba07e0af4df2ff6a4bce22f4e55b5
    - Code: https://github.com/google-research/google-research/tree/master/moe_mtl
    - Pub: ICCV 2023
    - Tag: `MoE`
    - Summary: For vision multi-task learning, they use task-dependent router and differnt top-k number for each task in MoE layer. They use task’s validation set loss to determine k, and outperform recent SOTA MTL MoE appraoch (TAPS).
    - Comment: The method is simple and performs well on image classification, object detection and instance segmentation. However, the training and inference resources need 16-64 and 8 TPU-v3.

- **From Sparse to Soft Mixtures of Experts**
    - Author: Joan Puigcerver∗ Carlos Riquelme∗, Basil Mustafa Neil Houlsby
    - Institute: Google DeepMind
    - Link: https://openreview.net/forum?id=jxpsAj7ltE
    - Code: Algorithm code at https://github.com/google-research/vmoe
    - Pub: ICLR 2024 spotlight
    - Tag: `MoE`
    - Summary: Instead of choosing top-k experts in MoE, they use softmax combination to transform input tokens into slots (Soft MoE). Through adjusting # of slots, they can achieve same training time complexity as MoE but performs better than other MoE baseline on vision tasks.
    - Comment: It still has challenge on implementing auto-regressive decoder. However, the transformation for hard assignment to soft address experts imbalance and token dropping problem.

- **Adaptive Gating in Mixture-of-Experts based Language Models**
    - Author: Jiamin Li, Qiang Su, etc
    - Institute: City University of Hong Kong, The Chinese University of Hong Kong
    - Link: https://arxiv.org/abs/2310.07188
    - Code: Not available
    - Pub: EMNLP 2023
    - Tag: `MoE`
    - Summary: Propose adaptive gating in MoE that enables tokens to be processed by a flexible number of experts depending on the gating decision andlLeverage the idea of curriculum learning by strategically adjusting the order of training data samples
    - Comment: The paper is well-written, the methods is simple and beautiful and they provides interesting insights of tokens routing to more experts, worthy to read it.

- **Mixtral of Experts**
    - Author: Albert Q. Jiang, Alexandre Sablayrolles, etc.
    - Institute: Mixtral.AI
    - Link: [https://arxiv.org/abs/2401.04088#:~:text=Mixtral of Experts](https://arxiv.org/abs/2401.04088#:~:text=Mixtral%20of%20Experts)
    - Code: https://github.com/mistralai/mistral-src
    - Pub: Arxiv 2024
    - Tag: `LLM` `MoE`
    - Summary: A huge Sparse Mixture of Experts language model which outperforms or matches Llama2-70B and GPT-3.5 across all evaluated benchmarks with 7B activated parameters. It did not use special structure nor new routing network but did a lot of interesting routing analysis.
    - Comment: They opensource their code and use Appache 2.0 lincense, so it’s friendly to the community, while it does not provide details about the datasets.

### Acknowledgement
Thanks for [Awesome-LLM-Prune](https://github.com/pprp/Awesome-LLM-Prune) that inspires me for this note, the template is based on their work.
