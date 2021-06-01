# ReaSCAN: Compositional Reasoning in Language Grounding
ReaSCAN is a synthetic navigation task that requires models to reason about surroundings over syntactically difficult languages.

## Contents

* [Citation](#Citation)
* [Dataset files](#dataset-files)
* [Quick start](#quick-start)
* [Data format](#data-format)
* [Models](#models)
* [Other files](#other-files)
* [License](#license)

## Citation

[Zhengxuan Wu](http://zen-wu.social), [Elisa Kreiss](https://www.elisakreiss.com/), [Desmond C. Ong](https://web.stanford.edu/~dco/), and [Christopher Potts](http://web.stanford.edu/~cgpotts/). 2020. [ReaSCAN: Compositional Reasoning in Language Grounding](http://zen-wu.social). Ms., Stanford University.

```stex
  @article{wu-etal-2020-dynasent,
    title={{ReaSCAN}: Compositional Reasoning in Language Grounding},
    author={Wu, Zhengxuan and Kreiss, Elisa and Ong, Desmond C. and Potts, Christopher},
    journal={},
    url={},
    year={2021}}
```

## Dataset files

### Off-the-shelves ReaSCAN

We generated ReaSCAN using our pipeline with fixed random seeds. You can reproduce the version of ReaSCAN we use in the paper by running the pipeline. Additionally, we also update the version we use to a online folder where you can directly download and use as-it-is. Note that, the dataset files are really large. It may take a while to download them.

Our generated data is in [ReaSCAN-v1.0.zip](), which is saved in a shared drive. The dataset consists subsets generated for different patterns (P1: non-clause (similar to gSCAN), P2: single-clause, P3: two-clause, P4: three-clause) and different compositional splits (see [our paper]() for details about each split).

By patterns,
* `ReaSCAN-compositional`: ReaSCAN P1 + P2 + P3, containing train, dev and test sets.
* `ReaSCAN-compositional-p1`: ReaSCAN P1, containing train, dev and test sets.
* `ReaSCAN-compositional-p2`: ReaSCAN P2, containing train, dev and test sets.
* `ReaSCAN-compositional-p3`: ReaSCAN P3, containing train, dev and test sets.
* `ReaSCAN-compositional-p1-test`: ReaSCAN P1, containing test set only.
* `ReaSCAN-compositional-p2-test`: ReaSCAN P2, containing test set only.
* `ReaSCAN-compositional-p3-test`: ReaSCAN P3, containing test set only.

By splits,
* `ReaSCAN-compositional-a1`: ReaSCAN A1 compositional split, containing test set only.
* `ReaSCAN-compositional-a2`: ReaSCAN A2 compositional split, containing test set only.
* `ReaSCAN-compositional-a3`: ReaSCAN A3 compositional split, containing test set only.
* `ReaSCAN-compositional-b1`: ReaSCAN B1 compositional split, containing test set only.
* `ReaSCAN-compositional-b2`: ReaSCAN B2 compositional split, containing test set only.
* `ReaSCAN-compositional-c`: ReaSCAN C compositional split, containing test set only.

Special split,
* `ReaSCAN-compositional-p3-rd`: ReaSCAN P3 with random distractors, containing train, dev and test sets.
* `ReaSCAN-compositional-p4` or `ReaSCAN-compositional-p4-test`: ReaSCAN P4 only, containing test set only.

### Regenerate ReaSCAN







