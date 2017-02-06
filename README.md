# evoplast

2015, 2016 Oswald Berthold

My branch of experimental work related to **evolution** and **plasticity**.

In Nature there is strong interaction between evolutionary and individual adaptation. In artificial evolution this interaction has only been given limited attention but offers huge potential. This project is a collection of my experiments in this direction.

Notable non-exhaustive foundations are
- Gregory Bateson: The Logical Categories of Learning and Communication, in Steps Toward an Ecology of Mind
- Kenneth Stanley & others: "A Hypercube-Based Encoding for Evolving Large-Scale Neural Networks". Artificial Life. 2009
- Paul Tonelli, Jean-Baptiste Mouret: On the Relationships between Generative Encodings, Regularity, and Learning Abilities when Evolving Plastic Artificial Neural Networks, PLOS ONE, November 2013 | Volume 8 | Issue 11 | e79138

## Examples

Run it like

`python ep3.py --numpopulation 20 --numgenerations 50 --mode es_vanilla --measure PI --measure_k 1 --measure_tau 1 --estimator kraskov2 --generator double --plotinterval 10 --op_mutation pareto --plotsave`

Use

`python ep3.py -h`

or code inspection to figure out the possible variations.
