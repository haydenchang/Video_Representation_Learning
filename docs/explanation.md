# Explanation

We train a self-supervised model on nuScenes front camera clips by hiding (masking) parts of the video and asking the model to reconstruct them. 
The model sees a clip of T frames, we mask a large fraction of spatiotemporal patches, and the encoder must build a representation that lets a small decoder predict the missing pixels/patches. 
Because there are no labels, the supervision signal comes from the reconstruction loss. 
After pretraining, we freeze the encoder and evaluate whether the learned representations are useful by training a simple linear probe on a proxy task and comparing against a randomly initialized encoder. 
If pretrained beats random, we scale dataset size on RCAC and measure how representation quality and throughput change.