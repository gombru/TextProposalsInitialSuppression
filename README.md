# TextProposals with early FCN pruning
This is an improved version of the [TextProposals](https://github.com/lluisgomez/TextProposals) algorithm by Lluis Gomez. It uses [TextFCN](https://github.com/gombru/TextFCN) to discard non-text regions, removing false positives and increasing the alforithm efficiency.

## Publications
This work lead to two publications, which you may cite if using this improved TextProposals method or the TextFCN:

[FAST: Facilitated and Accurate Scene Text Proposals through FCN Guided Pruning](http://www.sciencedirect.com/science/article/pii/S0167865517302982)  
Dena Bazazian, Raul Gomez, Anguelos Nicolaou, Lluis Gomez, Dimosthenis Karatzas, Andrew D.Bagdanov.   
Pattern Recognition Letters, 2017.

[Improving Text Proposals for Scene Images with Fully Convolutional Networks](https://arxiv.org/abs/1702.05089)  
Dena Bazazian, Raul Gomez, Anguelos Nicolaou, Lluis Gomez, Dimosthenis Karatzas and Andrew Bagdanov.  
ICPR workshop (DLPR), 2016.

## Usage:
For a detailed usage explanation refeer to [TextProposals](https://github.com/lluisgomez/TextProposals) repo. This code is similar, but includes support to load heatmaps produced by TextFCN and prune regions. 
The FCN model toguether with the code to produce heatmaps and train it is available in the [TextFCN](https://github.com/gombru/TextFCN) repo.

img2hierarchy image_path classifier_path (heatmap_path) (suppression_threshold)

	-image path and classifier_path are mandatory
	-If no suppression_threshold is given, 0.10 will be used
	-If no heatmap_path and suppression_threshold are given, initial suppression won't be used.
