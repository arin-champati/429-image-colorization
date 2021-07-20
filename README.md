# Small-Scale Automatic Image Colorization

Princeton University COS 429 Final Paper.

Contributors: 
Arin Champati, Brendan Houle, Byron Zhang

Description: 
We attempted to colorize grayscale images from the CIFAR-10 dataset without the naive approach of regression. Instead, we classified pixels within a quantized color space, preferring colors that are more rare in the dataset's color distribution.

We accomplished this task with a small dataset and very little computational resources due to our optimization of the quantization method and utilizing transfer learning from ImageNet. 

You can read our analysis on different color spaces, quantization bin sizes, effect of transfer learning, final results, etc. in the paper included in this GitHub repository. If you are curious about our implementation, our code is included in the repository as well, albeit a little messy. 

Thanks for visiting!

![Alt text](images/example_outputs.png?raw=true)
