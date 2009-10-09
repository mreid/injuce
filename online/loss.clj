;; Loss functions and their derivatives
;;
;; AUTHOR: Mark Reid <mark@reid.name>
;; CREATED: 2009-10-08

(ns loss
   (:use vec (clojure.contrib profile)))

(def *max-deriv* 100000)

(defn deriv
   "Returns the (clamped) derivative of the loss l evaluated at y v"
   [l y v] 
   (let [d (l y v true)]
      (if (> (Math/abs d) *max-deriv*) *max-deriv* d)))

(defn hinge
   "Returns the hinge loss for the prediction v"
   ([y v]   (prof :hinge (max 0 (- 1 (* y v)))))
   ([y v _] (prof :dhinge (if (> (hinge y v) 0) (- y) 0))))
   
(defn zero-one
   "Returns 1 if and only if v is of opposite sign to y"
   ([y v]   (if (< (* y v) 0) 1 0))
   ([y v _] 0))

(defn exp
   ([y v]   (prof :exp (Math/pow Math/E (- (* y v)))))
   ([y v _] (prof :dexp (- (* y (exp y v))))))

(defn logistic
   ([y v]   (prof :logistic  (Math/log (+ 1 (exp y v)))))
   ([y v _] (prof :dlogistic (/ (exp y v true) (+ 1 (exp y v))) )))
