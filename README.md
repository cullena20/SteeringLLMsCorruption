# Understanding and Mitigating Dataset Corruption in LLM Steering

Codebase for "Understanding and Mitigating Dataset Corruption in LLM Steering". We study the robustness of contrastive steering to control LLM 
behavior to corruption of the dataset used to train the steering direction. We additionally test the effect of steering where we use robust
mean estimators to take the means of activations, finding that the Lee Valiant estimator often mitigates most of the unwanted effects of
corruption on steering performance.

This codebase contains general code to calculate steering vectors, perform LLM steering, and to run evaluations over varying settings.
