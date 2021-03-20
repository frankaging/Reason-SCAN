# ReaSCAN: Compositional Generalizations with Grounded Abstract Language Reasoning
Humans learn world knowledge from past experiences and apply learnt knowledge
when facing new situations. Recently, SCAN-like datasets have been proposed
to test whether neural architectures learn generalizable knowledge that transfers
to unseen situations. However, we show that previous datasets naturally limit
neural models to learn simple pattern matching rather than grounded language
reasoning in achieving compositional generalizations. To resolve these limitations,
we propose a novel task ReaSCAN that requires neural models to reason about
surroundings over syntactically difficult languages. Different from previous tasks,
ReaSCAN also enforces models to reason over complex relations between objects
presented in the game world, which imitates real-world situations and evaluates
neural models against human-like reasonings and generalization abilities. We show
that existing neural architectures fail to generalize knowledge for unseen tasks in
the dataset, and perform worse comparing to previous SCAN-based tasks. We
hope ReaSCAN guides the community to invent neural architectures that acquire
generalizable reasoning abilities.

## Dataset Generation Workflow
<img src="https://i.ibb.co/BVkL2Z9/g-SCAN-workflow.png" width="500">
  - A diagram of the ReaSCAN generation workflow.

## ReaSCAN DSLs
<img src="https://i.ibb.co/ckzN3zN/Screen-Shot-2021-03-19-at-7-46-36-PM.png" width="500">
  - Domain Specific Languages (DSLs) for ReaSCAN command generations.

## Comparisons with other SCAN-like Datasets
<img src="https://i.ibb.co/Lzr3gBY/command-compare.png" width="500">
  - Generated examples from gSCAN dataset and ReaSCAN dataset. Relational clauses in ReaSCAn
dataset are bolded.

## Demo Notebook
We provide a quick demo notebook for existing functionalities of this dataset, 
they can be founded in `demo.ipynb` bu running
```python
cd code/dataset
jupyter notebook
```
Make sure you install all the dependencies before you run this.
