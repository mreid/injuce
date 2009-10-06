;; ---- Sparse Vector Operations ----
(ns test)
(def *max-features* 47153)

(ns colt (:import 
   (cern.colt.matrix.tfloat.impl DenseFloatMatrix1D SparseFloatMatrix1D)
   (cern.jet.math.tfloat FloatFunctions)
   (edu.emory.mathcs.utils ConcurrencyUtils))
   (:use clojure.contrib.profile))

(defn create
   "Returns a sparse vector created from a list of int/float pairs [k v]"
   [m]
   (prof :colt-create
   (let [v (SparseFloatMatrix1D. test/*max-features*)] 
      (do 
         (doseq [[i f] m] (.set v i f))
         v))))

(defn cardinality
   "Returns the number of non-zero elements in the given sparse vector"
   [v] (.cardinality v))

(defn add [x y] (prof :colt-add (.assign x y FloatFunctions/plus)))

(defn inner [x y] (prof :colt-inner (.zDotProduct x y)))

(defn norm
   "Returns the l_2 norm of the (sparse) vector v"
   [v] (Math/sqrt (inner v v)))

(defn scale [a v] (.assign v (FloatFunctions/mult a)))

(defn project
   "Scales x so it is inside the ball of radius r"
   [x r] 
   (let [n (norm x)]
      (if (> n r) 
         (scale (/ r n) x))))

;; ---- Map Operations ----
(ns map
   (:use clojure.contrib.profile))

(defn cardinality
   "Returns the number of non-zero elements in the given sparse vector"
   [v] (count v))

(defn add
   "Returns the sparse sum of two sparse vectors x and y" 
   [x y] (prof :map-add (merge-with + x y)))

(defn inner 
   [x y]
   (prof :map-inner (reduce + (map #(* (get x % 0) (get y % 0)) (keys y)))))

(defn norm
   "Returns the l_2 norm of the (sparse) vector v"
   [v] (Math/sqrt (inner v v)))

;; ==== Speed Test ====
(ns test
   (:import (edu.emory.mathcs.utils ConcurrencyUtils))
   (:use clojure.contrib.profile))

(def *sparsity* 0.05)

(defn rand-indices
   "Returns a lazy infinite sequence of integers in [0,range)."
   [range]
   (repeatedly #(int (Math/floor (rand range)))))

(defn rand-values
   "Returns a lazy infinite sequence of random floats in the range [min,max)"
   [min,max]
   (repeatedly
      #(+ min (rand (- max min)))))

(defn rand-vector
   "Returns a map representing a vector with specified dimension and sparsity.
    The values in the non-zero entries are floats in [0,1)"
   [dimension sparsity]
   (let [size (Math/floor (* dimension sparsity))]
      (zipmap 
         (take size (rand-indices dimension))
         (take size (rand-values 0 1)))))

;(set! *warn-on-reflection* true)
(ConcurrencyUtils/setNumberOfThreads 1) ; Done to stop time wasted in Futures

;; So far it seems map/add and map/inner are faster than the Colt libraries
;; for very sparse vectors (0.01-0.1) but Colt is faster for less sparse
;; vectors (0.2-)
(profile (dotimes [i 1000]
   (let [x  (rand-vector 45000 0.2)
         y  (rand-vector 45000 0.2)
         xc (colt/create x)
         yc (colt/create y)]
      (colt/inner xc yc)
      (colt/add xc yc)
      (map/add x y)
      (map/inner x y)      
   )))
   
;; Compute all inner products
; ... with map-based vectors
; ... with colt-based vectors

;; Compuate all pairwise additions
; ... with map-based vectors
; ... with colt-based vectors
