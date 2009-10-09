;; Tests for sparse vectors 
;;
;; AUTHOR: Mark Reid <mark@reid.name>
;; CREATED: 2009-10-09
(ns tests.vec
   (:use (clojure.contrib test-is) vec))

; Helper functions
(def *epsilon* 0.000001)  ; Tolerance for equality

(defn approx0 
    "Returns true iff the value x is within epsilon of 0"
    [x] (< (Math/abs x) *epsilon*))

(defn =eps
   "Returns true iff x-y is within epsilon of 0"
   [x y] (approx0 (- x y)))

; ---- Tests ----
(def x (create [ [1 1.0]  [2 2.0] [3 3.0] ]))
(def y (create [ [1 -1.0]         [3 1.0] [4 10.0]]))
(def z (create [          [2 2.0] [3 4.0] [4 10.0]]))

(deftest test-norm
   (is (=eps 14.0  (* (norm x) (norm x))) "Norm squared of x is 14.0")
   (is (=eps 102.0 (* (norm y) (norm y))) "Norm squared of y is 102.0")
   (is (=eps 120.0 (* (norm z) (norm z))) "Norm squared of z is 120.0"))

(deftest test-add
   (is (= z (add x y)) "Sum of x and y is z"))
   
(deftest test-inner
   (are (=eps (* (norm _1) (norm _1)) (inner _1 _1)) x y z)
   (is (=eps 2.0   (inner x y))   "Inner product of x and y is 2.0")
   (is (=eps 16.0  (inner x z))   "Inner product of x and y is 16.0")
   (is (=eps 104.0 (inner y z))   "Inner product of y and z is 104.0"))
