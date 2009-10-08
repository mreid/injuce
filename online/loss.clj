;; Loss functions and their derivatives
;;
;; AUTHOR: Mark Reid <mark@reid.name>
;; CREATED: 2009-10-08

(ns loss
   (:use vec))

(defn deriv
   "Returns the derivative of the loss l evaluated at y v"
   [l y v] (l y v true))

(defn hinge
   "Returns the hinge loss for the prediction v"
   ([y v]   (max 0 (- 1 (* y v))))
   ([y v _] (if (> (hinge y v) 0) (- y) 0)))
   
(defn zero-one
   "Returns 1 if and only if v is of opposite sign to y"
   ([y v]   (if (< (* y v) 0) 1 0))
   ([y v _] 0))

(defn exp
   ([y v]   (Math/pow Math/E (- (* y v))))
   ([y v _] (- (* y (exp y v)))))
