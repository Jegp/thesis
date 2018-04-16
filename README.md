# Modelling learning tasks in spiking neural networks
Building on research from cognitive and computational neuroscience, deep
learning is evolving rapidly and has even surpassed humans in some recognition
tasks [1]. Contemporary theories from cognitive neuroscience however, tell us
that learning in the biological brain occurs in spiking neural networks
instead of the layered neural networks used in machine learning [2].

Pfeil et. al. [3], Tavanaei and Maida [4] and Walter et al. [5] have shown that spiking neural networks are capable of solving a wide range of learning tasks.
The neuromorphic hardware platform BrainScaleS has even been showed to learn classification tasks [6]. Such platforms however, require considerable configuration in both hardware and software [5, 6, 7].

Because of their biological similarities neuromorphic hardware is of great interest to neuroscientists [7].
The programming and configuration needed to setup neuromorphic experiments are however inaccessible to most neuroscientists.

Based on a Krechevky maze-learning task [8] and the theory of Reorganisation of Elementary Funtions (REF) [9], this thesis sets out to explore how a more accessible domain specific language (Volr) [10] can help to build and evaluate experiments in neuromorphic hardware.
By building on the REF model, the setup can be transferred to other learning tasks, if successful, and could lay the foundation for a robust neuromorphic experimental framework for cognitive neuroscientists.

## Hypothesis
This thesis examines the hypothesis that *the model for the Reorganisation of Elementary Functions can be implemented using spiking neural networks*.

The hypothesis drives two outcomes: a spiking neural network representation of the REF model and a Krechevky maze experiment.

## Scope
The hypothesis will be evaluated based on two criteria: the similarity of the spiking REF model to contemporary neurocognitive theories on the mammalian brain as well as its capacity to learn a given problem.

The thesis will focus on how well the the model describes contemporary neurocognitive theories of learning, keeping the limits of the experimental platforms in mind.
Because the REF model has not been mapped to its physiological properties, it is outside the scope of this thesis to provide an exact comparison the biological properties.

To provide a context for the learning capacity of the spiking neural networks, they will be compared to a regular non-spiking neural network.

## Experimental setup
Three models will be built: a spiking neural network simulation via NEST [11], a spiking neural network emulation via BrainScaleS [6] and a layered non-spiking neural network written in Futhark [12].
All three models will be trained to perform the same Krechevky maze task.

Since the neuromorphic hardware is significantly more performant than the software simulation, it is desirable to run it on chip.
It is however, important to retain the simulation as a baseline, because of unexpected analogue effects of the neuromorphic hardware.

The non-spiking neural network will be written in Futhark and executed using OpenCL.

## Tasks
The tasks involved in this thesis are as follows:

* Implement a Krechevky maze experiment and a regular layered neural network in Futhark.
* Implement a prototypical REF model in Volr that can be emulated both using the spiking. neural network simulator NEST [11] and the neuromorphic hardware platform BrainScaleS [6].
* Compare the REF model to contemporary theories on learning and adaptation in cognitive neuroscience.
* Extend the Krechevky maze experiment [8] to encode maze features as temporal spiking stimulus.
* Execute the Krechevky maze experiment using the prototypical neuromorphic REF model on the NEST and BrainScaleS platforms.
* Compare the performance of the neuromorphic experiments to the Futhark experiment, with a focus on learning rate and accuracy.

## Thesis milestones
The thesis will start on the 23rd of April 2018 and will be handed in on the
19th of August. The process is divided into four miletones with associated
deadlines:

| Milestone | Date of completion |
| ---------------------------------------------- | ------------------ |
| Account for contemporary theories from cognitive neuroscience on spiking neural networks and learning models in spiking neural networks | 20th of May |
| Implementation of neuromorphic and traditional machine learning experiment | 30th of June |
| Execute, analyse and compare experiments | 31st of July |
| Finish writing | 12th of August |

## Learning objectives

* Survey models for describing learning tasks in neuromorphic hardware.
* Implement spiking neural networks on simulated as well as hardware-backed platforms.
* Evaluate learning models using spiking neural networks.
* Evaluate learning models using regular machine learning techniques.
* Analyse and compare learning models on neuromorphic as well as traditional hardware.

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
11. Kunkel Susanne, Schmidt Maximilian, Eppler Jochen M., Plesser Hans E., Masumoto Gen, Igarashi Jun, Ishii Shin, Fukai Tomoki, Morrison Abigail, Diesmann Markus, Helias Moritz: "Spiking network simulation code for petascale computers", Frontiers in Neuroinformatics, Vol. 8, p. 78, 2014.
12. Henriken, Troels: "Design and Implementation of the Futhark Programming Language", Ph.D. thesis, Faculty of Science, Copenhagen University 2017.
