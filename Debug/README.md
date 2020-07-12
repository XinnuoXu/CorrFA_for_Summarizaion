# System attacking


## Step1: Create your examples
To try some DIY examples, you should write your examples down in `input.src` and `input.tgt`. `input.src` is for document trees and `input.tgt` is for summaries(just sentences, not trees). Note that

* each line in `input.src` is the tree MR for a document. The format is strings jointed by '\t'.
* the number of lines in `input.src` is equal to `input.tgt`.
* lines in `input.src` should be the copy of the first row (sorry for the duplication. I will make it decent soon).
* the first row in `input.tgt` should be the ground truth. Please find some examples in `input.src` and `input.tgt` in this repository. 

## Step2: Get Corr scores and debug information

Run `sh run.sh`. it will create a directory called `output/` containing following files:

* `corr.txt` is the corr scores for each examples. It's one row per example with the format `document \t summary \t Corr_F \t Corr_A` for each row.
* `{k}.txt` are details for fact/argument distances in the kth example (<img src="http://latex.codecogs.com/gif.latex?d_{ij}^f" border="0"/> and <img src="http://latex.codecogs.com/gif.latex?d_{ij}^a" border="0"/> in the paper). 
* `{k}.hl` are fact level weights <img src="http://latex.codecogs.com/gif.latex?\mathbf{w}_\ast^f" border="0"/> for the kth example.

## Step3: Visualization

To visualize the fact/argument distances (`{k}.txt`) and fact level weights (`{k}.hl`), you can copy the file you are interested in to `../Display/` and replace it to the file `attn_vis_data.json`. Then run

```
sh run_service.sh
```

You can view the visualization result by visiting http://localhost:8000 is your web browser.

Here is an example for fact distance of `F2-sent` in the summary. Note that, you need to put your mouse on the `F2-sent` to make the highlights show up.
![alt text](https://github.com/XinnuoXu/CorrFA_for_Summarizaion/blob/master/Display/distance.png)

Here is an example for fact level weights for an example. Note that, you need to put your mouse on the `[FACT WEIGHTS]` to make the highlights show up.
![alt text](https://github.com/XinnuoXu/CorrFA_for_Summarizaion/blob/master/Display/weights.png)



