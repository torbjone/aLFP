# Warning: This code is not fully up to date with Python 3 and LFPy 2.X. If you run into problems, please raise an issue or contact the authors.#

# README #

The python code to reproduce all figures from a modeling study of the contribution
of active conductances to the Local Field Potential for single cells.
Figure numbers in paper_figures.py corresponds to figures in this paper:
https://www.ncbi.nlm.nih.gov/pubmed/27079755

### How do I get set up? ###
In all folders containing .mod files the command "nrnivmodl" (Linux and Mac) must be excuted in a terminal. This assumes
that NEURON (www.neuron.yale.edu) is set up on the system and functioning properly. LFPy must also be installed.
This can be done by pip install, "pip install LFPy", but see

lfpy.github.io/information.html#installing-neuron-with-python

for more information on how to make NEURON and Python work together.

No attempts has been made for this to work at other operating systems than Linux, but we are happy to help people try
get started.

### Who do I talk to? ###

Torbjørn V Ness - torbness@gmail.com

Michiel Remme

Gaute T Einevol - gaute.einevoll@nmbu.no
