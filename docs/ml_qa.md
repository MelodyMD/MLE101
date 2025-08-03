# ML Q&A

---

## Fundamental Concepts

- What is bias vs variance?
- What is overfitting vs underfitting?
- How does regularization help prevent overfitting?
- What is the difference between L1 and L2 regularization?
- What is the difference between generative and discriminative models?
- What is the difference between supervised and unsupervised learning?
- What is the difference between LDA and PCA?
- What is the curse of dimensionality?

---

## Evaluation Metrics

- What are precision, recall, and F1 score?
- What is an ROC curve, and how is it used?
- What is AUC and how is it interpreted?
- What is log loss / cross-entropy loss?
- What are confusion matrices and how do you interpret them?
- What is the difference between micro, macro, and weighted averaging in classification metrics?
- How do you evaluate a model for multi-class vs multi-label classification?
- What metrics are suitable for ranking models? 

---

## Optimization

- How does gradient descent work?
- What is stochastic gradient descent, and how is it different from full-batch gradient descent?
- What is backward propagation, and how does it relate to neural networks?
- What are common variants of gradient descent (SGD, Momentum, Adam, RMSProp)?
- What are vanishing and exploding gradients?
- How does learning rate affect training?
- What is gradient clipping and why is it used?
- How do optimizers differ in convergence speed and stability?

---

## ML Algorithms

- What is logistic regression? (forward calculation, loss function)
- What is K-Nearest Neighbors (KNN)?
- What is decision tree learning?
- What is random forest, and how does it differ from gradient boosting?
- What is the difference between bagging and boosting?
- What is K-Means clustering, and how does it work?
- What is Support Vector Machine (SVM)?
- What is Bayesian learning?
- What is the difference between MAP and MLE?
- What is Naive Bayes and when is it effective?
- What are the assumptions behind linear regression?
- How does ridge regression differ from lasso regression?
- What are the strengths and weaknesses of tree-based methods?
- What are ensemble methods, and why do they work?

---

## Data Issues

- How does imbalanced data affect model performance?
- What are common strategies to handle imbalanced data?
- What is data drift and how do you handle it?
- How to handle missing data in ML pipelines?
- How do you split your data into train/validation/test?
- What are the benefits and risks of oversampling and undersampling?

---

## Deep Learning

- What is the ReLU activation function?
- How to deal with gradient vanishing?
- What are common activation functions and when to use them (ReLU, sigmoid, tanh, GELU, etc.)?
- What are fully connected layers?
- What is the role of initialization in deep learning?
- What is dropout and how does it prevent overfitting?
- What is an epoch, batch, and iteration?
- What is early stopping and how does it help generalization?
- What is the difference between feedforward networks and recurrent networks?

---

## Transformers

- Why do we divide the attention score by √dₖ in the Transformer?
- Why are different weight matrices used to compute Q, K, and V?
- Why do we use Multi-Head Attention?
- What is the time complexity of attention?
- What is KV cache and why is it used?
- What is Multi-Query Attention (MQA)?
- What is Grouped Query Attention (GQA)?
- What are the shapes of Q, K, V in MHA, MQA, and GQA?
- What is Flash Attention and how does it work?
- How to optimize memory usage in attention mechanisms?

---

## LLM Architecture & Inference

- What are the key factors that affect LLM inference latency?
- What are common LLM inference optimization techniques?
- What is KV cache and how does it improve inference?
- What is smart batching and how does it affect performance?
- What is quantization and how does it help inference?
- What is the tradeoff in using MQA or GQA?

---

## LLM Fine-Tuning Techniques

- What are typical fine-tuning methods for LLMs?
- What is LoRA and how does it work?
- How is the loss calculated in supervised fine-tuning (SFT)?
- What is the difference between prefix tuning and prompt tuning?
- What is RLHF and how does it work?

---

## LLM Training Optimization

- What is mixed-precision training and how does it work?
- What are the benefits and tradeoffs of FP16 vs BF16?
- What is gradient checkpointing?
- What is Distributed Data Parallel (DDP)?
- What is Fully Sharded Data Parallel (FSDP)?
- What is ZeRO (Zero Redundancy Optimizer) and its stages?
- How does ZeRO enable large-scale model training?

---

## Retrieval-Augmented Generation (RAG)

- Why use RAG for LLMs?
- How does RAG architecture work?
- What are the steps to build a RAG-based chatbot?

---

---

## Learning Resources

### GitHub Repositories
- [andrewekhalel / MLQuestions](https://github.com/andrewekhalel/MLQuestions)  
  A curated list of ML interview questions categorized by topic, with some helpful explanations.
- [khangich / machine-learning-interview](https://github.com/khangich/machine-learning-interview)  
  Interview prep notebook covering theory and code implementations for ML algorithms.

---

### Blogs & Articles

#### General ML
- [Multicollinearity in Regression Analysis](https://statisticsbyjim.com/regression/multicollinearity-in-regression-analysis/) – Statistics by Jim  
  Explains the impact of multicollinearity on model interpretability and stability.
- [Feature Scaling Overview](https://sebastianraschka.com/Articles/2014_about_feature_scaling.html) – Sebastian Raschka  
  Covers different scaling techniques and their effect on ML models.
- [Gradient Boosting vs Random Forest](https://medium.com/@aravanshad/gradient-boosting-versus-random-forest-cfa3fa8f0d80)  
  An intuitive side-by-side comparison with examples.
- [L1 vs L2 Regularization (Visual Explanation)](https://www.linkedin.com/pulse/intuitive-visual-explanation-differences-between-l1-l2-xiaoli-chen/)  
  Illustrates geometric differences and impact on sparsity.
- [PCA Explained](https://towardsdatascience.com/a-one-stop-shop-for-principal-component-analysis-5582fb7e0a9c)  
  A step-by-step walkthrough of PCA with visuals.
- [Logistic Regression Detailed Overview](https://towardsdatascience.com/logistic-regression-detailed-overview-46c4da4303bc)  
  Covers math, intuition, and loss functions.

#### Clustering
- [K-Means Intuition (Stanford CS221)](https://stanford.edu/~cpiech/cs221/handouts/kmeans.html)  
  Visual and interactive explanation of the K-Means algorithm.

#### Transformers & DL
- [Cross-Attention in Transformers](https://medium.com/@sachinsoni600517/cross-attention-in-transformer-f37ce7129d78)  
  Cross Attention Explanation.
- [BatchNorm vs LayerNorm](https://www.linkedin.com/pulse/understanding-batch-normalization-layer-group-implementing-pasha-s/)  
  Explains when and why to use each normalization technique.
- [Bagging vs Boosting](https://www.geeksforgeeks.org/bagging-vs-boosting-in-machine-learning/)  
  Good summary for tree-based ensemble learners.

---

### Research Papers

- *[Attention Is All You Need](https://arxiv.org/abs/1706.03762)* – Vaswani et al.  
  The original paper that introduced the Transformer architecture.
- *CLIP: Learning Transferable Visual Models From Natural Language*  
  [CLIP Paper (OpenAI)](https://openai.com/research/clip) – Vision-language model combining image and text representations.

---

### Courses

- [Deep Learning Specialization (Coursera by Andrew Ng)](https://www.coursera.org/specializations/deep-learning)  
  A five-course series covering neural networks, CNNs, RNNs, optimization, and structuring deep learning projects. Ideal for building strong foundational intuition.

- [Generative AI with LLMs (DeepLearning.AI)](https://www.deeplearning.ai/courses/generative-ai-with-llms/)  
  Covers modern LLM workflows, from prompt engineering to RAG and fine-tuning.

