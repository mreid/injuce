Injuce - An induction toolkit in Clojure
========================================

This is a small personal project used to better understand Clojure and some classic statistical machine learning algorithms.

Initially, this will just consist of some simple data handling routines and an implementation of stochastic gradient descent.

Set up
------

### Requirements ###

* [Clojure][] - 

* [Incanter][]

[clojure]: http://clojure.org/
[incanter]: http://incanter.org/

## Incanter Script ##

The programs here will be run using the `clj` script that comes with Incanter. This script sets up the classpath for Incanter.

To run a program in this package, call it as follows:

	$ ../incanter/bin/clj FILENAME.clj

where `../incanter` should be replaced with whatever path gets you to your installation of Incanter.

I'll aim to make this easier in the future.

Performance Notes
-----------------
Just parsing the entire training set `train.dat.gz` takes some time:

	$ zless ../../sgd/svm/train.dat.gz | clj sgd.clj 
	"Elapsed time: 189872.08 msecs"
	781265

That is a total of ~3 mins for a rate of 0.24 msecs/example. However, the C++ version of `svmsgd` also takes some time to read in `train.dat.gz` (in the order of a several minutes).

It seems the most expensive part of the Clojure SGD is the operations on the sparse vectors represented as hash maps. Use of `jvisualvm` shows at least 20% of processing time is spent in calls to `clojure.lang.Var.get`. 

The current code cannot complete a single training run through all the data, throwing a `OutOfMemory` (out of Java heap) exception after processing about 11k examples. To get to this point takes hours of processing (compared to Bottou's `svmsgd` which takes seconds to process the entire training set).

To do
-----
[_] Write a new parser that reads the `train.bin.gz` format.
[_] Make use of the COLT libraries that Incanter is built upon for sparse vecs.