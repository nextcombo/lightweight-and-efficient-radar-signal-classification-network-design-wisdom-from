# Lightweight and Efficient Radar Signal Classification Network Design Wisdom: From Coordinate Attention to Mixed Enhancement Training [with Python code].

Lightweight and Efficient Radar Signal Classification Network Design: From Coordinate Attention to Mixed Enhancement Training
In the field of modern electronic reconnaissance and spectrum perception, the automatic modulation recognition of radar communication signals has become one of the core technologies in cognitive radio and electromagnetic situation analysis. As the electromagnetic environment becomes increasingly complex, traditional expert feature extraction methods have gradually revealed their limitations when faced with noise interference and diverse signal styles. In recent years, deep learning, with its end-to-end feature learning capability, has shown great potential in the task of radio frequency signal classification. However, how to achieve high accuracy and strong robustness in classification performance within the constraints of limited computational resources and model size remains a key challenge in engineering applications.
This paper is based on the design of a small convolutional neural network with a limited parameter quantity not exceeding 100,000, integrating advanced techniques such as coordinate attention mechanism, mixed sample enhancement, weighted exponential moving average, etc., achieving excellent classification results on a radar signal dataset. We will systematically review these technical points from a theoretical level, providing a reference framework for lightweight model design for researchers involved in related studies.
I. Enhancing Generalization Data Augmentation Strategies
1.1 Traditional image domain enhancement: Given the characteristic of radio frequency signals often converted into images through time-frequency diagrams, random horizontal flipping and color jitter are used to simulate possible mirror spectrum offsets and minor distortions during signal reception. These classic augmentation operations expand data diversity without altering the signal semantics, suppressing overfitting.
1.2 Mixup: To further regularize the decision boundary of the model, Mixup is introduced. Its core idea is to perform convex combinations between training samples, forcing the model to learn linear interpolation between samples, thereby smoothing the input space. For a pair of samples and, construct virtual training samples:
where the mixing coefficient is sampled from a beta distribution, typically generating mixtures biased towards the original samples. The loss function is adjusted accordingly:
Mixup effectively improves the model's consistency, reduces susceptibility to adversarial examples, and provides a smoother prediction distribution in the out-of-label regions.
II. Theoretical Innovation in Model Architecture The BasicCNN network designed in this paper integrates the Mobile Flip-Through Bottleneck Convolution (MBConv) with Coordinate Attention, and employs a dual pooling strategy at the end to integrate features.
2.1 Mobile Flip-Flops Convolution (MBConv) MBConv originates from the EfficientNet series and is an efficient variant of deep separable convolutions. Its structure includes three key steps:
1  
Point by point expansion: Utilize convolution to expand the number of input channels by a factor (typically between 2 and 6), enhancing feature representation in high-dimensional spaces.
2  
Depthwise Convolution: Independent spatial convolution on each channel, significantly reducing the number of parameters.
3  
Compression and Projection: Through convolution, the number of channels is projected back to the target output dimension, and optionally, residual connections are added.
The parameter count of deep convolution is only that of standard convolution, making MBConv the cornerstone for building lightweight models while maintaining strong feature extraction capabilities.
2.2 Coordinate Attention Mechanism (Coordinate Attention) Traditional channel attention mechanisms, such as SENet, model channel dependencies solely through global average pooling, completely disregarding spatial position information, which is particularly disadvantageous for position-sensitive radio frequency signal time-frequency plots. Coordinate attention decomposes spatial attention into two parallel one-dimensional feature encoding processes.
For the input feature map, firstly perform global average pooling along both horizontal and vertical directions:
The pair of direction-aware feature vectors are obtained. They are then concatenated, followed by shared convolution, batch normalization, and non-linear activation:
The SiLU (Sigmoid Linear Unit) activation function is employed. Subsequently, the spatial dimensions are divided into and separately processed through convolution and target Sigmoid gating to generate attention weights:
The final output is the element-wise multiplication of the original features and two attention weights:
This design not only captures inter-channel dependencies but also precisely encodes both horizontal and vertical spatial structure information, significantly enhancing the model's ability to distinguish signal texture and frequency point locations in time-frequency images. After being embedded into MBConv's deep convolution, i.e., forming the MBConvCA module, each stage of the network possesses spatial-channel joint attention.
2.3 Dual pooling feature aggregation at the end of convolution feature extraction, traditional global average pooling merely retains the average energy of the data while discarding peak information. This paper adopts both global average pooling and global max pooling:
By stitching two feature vectors along the channel dimension, an is obtained, which is then input to a fully connected layer with Dropout regularization for classification. The dual pooling strategy can complementarily capture both the overall statistical quantities of signals and the instantaneous strong response features, especially beneficial for signal detection under broadband noise conditions.
III. Training Strategies with Regularization and Optimization Techniques 3.1 Label Smoothing (Label Smoothing) Traditional cross-entropy loss guided by hard-coded one-hot labels can lead to overconfidence in the model, resulting in decreased generalization performance. Label smoothing softens the distribution of true labels from rigid to flexible:
where n is the number of classes and σ is the smoothing coefficient (in this paper, set to 0.1). The cross-entropy loss becomes, encouraging the model to learn a more uniform prediction distribution, effectively preventing overfitting, and working in conjunction with Mixup.
3.2 Weighted Exponential Moving Average (EMA) During training, stochastic gradient descent of model parameters often involves large fluctuations, especially when using smaller batch sizes. Weighted exponential moving average maintains a smooth set of model parameters updated after each iteration:
The decay factor is set to 0.999, making the EMA model equivalent to taking a weighted average of recent parameter trajectories. Using EMA parameters during validation and testing phases can eliminate random noise-induced jitter, typically achieving higher and more stable accuracy than the original training model without additional inference overhead.
3.3 Optimizer selection and learning rate scheduling: The AdamW optimizer is used, which decouples weight decay from adaptive learning rates:
It is a bias correction value for momentum and second moment, and a decoupling weight decay coefficient. Compared with traditional L2 regularization in Adam, AdamW can more precisely control the decay of weights, thereby improving the performance of the model after convergence.
Learning rate strategy adopts cosine annealing scheduling:
During 80 training cycles, the learning rate smoothly decreases from its initial value to a minimum. The cosine curve maintains a high learning rate during the initial training phase, and gradually cools down later to achieve fine convergence, avoiding the need for manual tuning of parameters as in stepwise descent.
IV. Assessment System and Reliability Model Evaluation
V. Reference and Referencing Significance for Researchers
1  
Attention Mechanism Spatialization: Coordinate Attention integrates positional information into the feature map with minimal computational overhead, making it more suitable for retaining two-dimensional structured signal data than using SE modules alone. It can be extended to other time-frequency analysis tasks.
2  
Mixup with label smoothing: Both regularization methods enhance the generalization ability of small models from two perspectives: data space and label distribution. When sample size is limited or class imbalance exists, their combination is highly recommended.
3  
EMA in Validation: The parameters snapshots during training are often not the optimal model. EMA can almost costlessly uncover better weights, a technique simple yet often overlooked.
4  
Lightweight inference deployment: The proposed MBConvCA module and overall network design strictly adhere to the total parameter limit, while introducing multi-scale information through dual pooling. It is suitable for portability to FPGAs or embedded real-time spectrum monitoring devices.  
5  
Training process visualization and recording: The system records loss and accuracy curves, and outputs confusion matrices and various category metrics. This provides a clear quantitative basis for subsequent model diagnosis and optimization, and is recommended as an experimental standardization process.  
VI. Conclusion In the field of radio frequency cognitive, the precision and efficiency of algorithms often need to be delicately balanced. This paper, by deeply analyzing the theoretical foundation and synergistic effects of key technologies such as coordinate attention, MBConv, Mixup, etc., demonstrates a feasible path for constructing high-performance classification models under strict parameter constraints. We hope this design concept can provide new ideas and references for researchers engaged in intelligent signal processing in communication and embedded deep learning model compression, promoting the practical application of lightweight deep learning methods in electromagnetic perception.

## Images

![Designing a Lightweight and Efficient Radar Signal Classification Network: From Coordinate Attention to Hybrid Enhancement Training [with Python code].](01.jpg)
![Designing a Lightweight and Efficient Radar Signal Classification Network: From Coordinate Attention to Hybrid Enhancement Training [with Python code].](02.jpg)
![Designing a Lightweight and Efficient Radar Signal Classification Network: From Coordinate Attention to Hybrid Enhancement Training [with Python code].](03.jpg)

<!-- PAYMENT_BLOCK -->

## Get this dataset

- Payment link (Stripe): [Buy now](https://buy.stripe.com/3cs8yP7sY87d0vu9AB)
- Payment QR code: ![Payment QR code](pay.png)

<!-- /PAYMENT_BLOCK -->
