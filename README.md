# Modelling learning tasks in spiking neural networks
Building on research from cognitive and computational neuroscience, deep
learning is evolving rapidly and have even surpassed humans in some recognition
tasks [1]. Contemporary theories from cognitive neuroscience however, tells us
that learning in the biological brain occurs in spiking neural networks
instead of the layered neural networks used in machine learning [2].

Pfeil et. al. [3], Tavanaei and Maida [4], Walter et al. [5] and Schmitt et al. [6] have shown that spiking neural networks are capable of solving a wide range of learning tasks.
The mentioned networks however, require considerable configuration in both hardware and software [5, 6, 7].

Because of their biological similarities neuromorphic hardware are of great interest to neuroscientists [7].
The programming and configuration needed to instrument neuromorphic experiments are however inaccessible to most neuroscientist.

Based on a Krechevky maze-learning task [8] and the theory of Reorganisation of Elementary Funtions (REF) [9], this thesis sets out to explore the feasibility of instrumenting and evaluating experiments in neuromorphic hardware with a more accessible domain-specific language (DSL), Volr [10].
By building on the REF model, the setup can, if successful, be transferred to other learning tasks, and could lay the foundation for a robust neuromorphic experimental framework for cognitive neuroscientists.

To measure the success of the learning task the neuromorphic experiment will be evaluated against a regular layered neural network (on traditional hardware).

## Hypothesis
This thesis examines the hypothesis that *the model for the Reorganisation of Elementary Functions can be implemented using spiking neural networks*.

The hypothesis drives two outcomes: a spiking neural network representation of the REF model and a Krechevky maze experiment.
The experiment makes it possible to test the learning capacities of the spiking neural network. And because the maze can be modelled in both spiking and non-spiking neural networks, it can form the basis for a comparison between the spiking neural network model and a traditional layered neural network.

## Tasks
The tasks involved in this thesis are as follows:

* Implement a Krechevky maze experiment and regular layered neural network in the data-parallel functional programming language Futhark
* Implement a prototypical REF model that can be emulated both using the spiking neural network simulator, NEST [11] and the neuromorphic hardware platform BrainScaleS [6].
* Construct a Krechevky maze experiment [8] that encodes maze features as temporal spiking stimulus.
* Execute the Krechevky maze experiment using the prototypical neuromorphic REF model on the NEST and BrainScaleS platforms.
* Compare the performance of the neuromorphic experiments to the Futhark experiment, with a focus on learning rate and accuracy.

## Thesis milestones
The thesis will start on the 23rd of April 2018 and will be handed in on the
19th of August. The process is divided into four miletones with associated
deadlines:

| Milestone | Date of completion |
| -------------------------------------- | ------------------- |
| Account for contemporary theories from cognitive neuroscience on spiking neural networks and learning models in spiking neural networks | 20th of May |
| Implementation of neuromorphic and traditional machine learning experiment | 30th of June |
| Executer, analyse and compare experiments | 31st of July |
| Final writing phase | 12th of August |

## Learning objectives

* Survey models for describing learning tasks in neuromorphic hardware
* Implement spiking neural networks on simulated as well at hardware-backed platforms
* Evaluate learning tasks on spiking neural networks
* Evaluate learning models on neuromorphic hardware using the Volr DSL
* Evaluate, analyse and compare learning task implementations on neuromorphic as well as traditional hardware

## Sources
1. J. Schmidhuber: "Deep Learning in Neural Networks: An Overview",
Neural Networks, pp. "85-117, Volume 61, 2015.
2. Peter Dayan and L. F. Abbot: "Theoretical neuroscience - Computational and Mathematical Modeling of Neural Systems", MIT Press 2001.
3.  Thomas Pfeil, Andreas Grübl, Sebastian Jeltsch, Eric Müller, Paul Müller, Mihai A. Petrovici, Michael Schmuker, Daniel Brüderle, Johannes Schemmel and Karlheinz Meier: "Six networks on a universan neuromorphic computing substrate", Frontiers in Neuroscience, 2013.
4. A. Tavanei and Anthony S. Maida: "A Minimal Spiking Neural Network to Rapidly Train and Classify Handwritten Digits in Binary and 10-Digit Tasks", International Journal of Advanced Research in Artificial Intelligence, Vol. 4, No. 7, 2015.
5. Florian Walter, Florian Röhrbein and Alois Knoll: "Neuromorphic implementations of neurobiological learning algorithms for spiking neural networks", Neural Networks,
Volume 72, pp. 152-167, 2015.
6. Sebastian Schmitt, Johann Klaehn, Guillaume Bellec, Andreas Grübl, Maurice Guettler, Andreas Hartel, Stephan Hartmann, Dan Husmann de Oliveira, Kai Husmann, Vitali Karasenko, Mitja Kleider, Christoph Koke, Christian Mauch, Eric Müller, Paul Müller, Johannes Partzsch, Mihai A. Petrovici, Stefan Schiefer, Stefan Scholze, Bernhard Vogginger, Robert A. Legenstein, Wolfgang Maass, Christian Mayr, Johannes Schemmel and Karlheinz Meier: "Neuromorphic Hardware In The Loop: Training a Deep Spiking Network on the BrainScaleS Wafer-Scale System", CoRR, abs/1703.01909 2017.
7. Daniel Brüderle, Mihai A. Petrovici, Bernhard Vogginger, Matthias Ehrlich, Thomas Pfeil, Sebastian Millner, Andreas Grübl, Karsten Wendt, Eric Müller, Marc-Olivier Schwartz, Dan Husmann de Oliveira, Sebastian Jeltsch, Johannes Fieres, Moritz Schilling, Paul Müller, Oliver Breitwieser, Venelin Petkov, Lyle Muller, Andrew P. Davison, Pradeep Krishnamurthy, Jens Kremkow, Mikael Lundqvist, Eilif Muller, Johannes Partzsch, Stefan Scholze, Lukas Zühl, Christian Mayr, Alain Destexhe, Markus Diesmann, Tobias C. Potjans, Anders Lansner, René Schüffny, Johannes Schemmel and Karlheinz Meier: "A Comprehensive Workflow for General-Purpose Neural Modeling with Highly Configurable Neuromorphic Hardware Systems", Biol Cybern. 2011 May;104(4-5):263-96.
8. Krechevsky, I.: "Hypotheses in Rats", Psychological Review, Vol. 39, pp. 516-532, 1932.
9. Jesper Mogensen and Morten Overgaard: Reorganization of the Connectivity between Elementary Functions - A Model Relating Conscious States to Neural Connections, Fronties in Psychology, 8:652, 2017.
10. Jens Egholm Pedersen: "Modelling learning systems - a DSL for cognitive neuroscientist", Copenhagen University, project report, 2018. Available at https://github.com/Jegp/volr-report/blob/master/report.pdf
11. Kunkel Susanne, Schmidt Maximilian, Eppler Jochen M., Plesser Hans E., Masumoto Gen, Igarashi Jun, Ishii Shin, Fukai Tomoki, Morrison Abigail, Diesmann Markus, Helias Moritz:_"Spiking network simulation code for petascale computers", Frontiers in Neuroinformatics, Vol. 8, p. 78, 2014.
