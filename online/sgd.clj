;; Stochatistic Gradient Descent in Clojure
;;
;; AUTHOR: Mark Reid <mark@reid.name>
;; CREATED: 2009-07-08

; The Pegasos algorithm (Shalev-Shwartz, Singer & Srebro, ICML 2007) solves
; the above problem for hinge loss with l_2 regularisation (i.e., SVM) as 
; follows:
;
; PEGASOS
;  Given: 
;     S     - training examples
;     \lambda  - regularisation factor
;     T     - number of iterations
;     k     - size of subset of S to use for gradient computation
;
;   Initialise:
;     Choose w_1 s.t. ||w_1|| ≤ \lambda^{-1/2}
;
;  Do the following:
;     For t = 1, 2, ..., T do
;        \eta_t <- 1/(\lambda t)
;        A_t  <- Some subset of k training examples from S
;        B_t  <- { x \in A_t : y <w_t, x> < 1 } (i.e., misclassified examples)
;        w'_t <- (1-\eta_t \lambda) w_t + \eta_t / k \sum_{(x,y)\in B_t} y x
;        w_{t+1} <- \min{ 1, \lambda^{-1/2}/||w'_t|| } w'_t
;     Output w_{T+1}

(ns sgd
   (:use vec learner (clojure.contrib profile))
   (:require loss))

;; Offset for time step based on code in Bottou's svmsgd.
(def offset
   #^"Computes a starting offset from given a lambda a la Bottou's svmsgd"
   (memoize
      (fn [lambda] (/ 1 (* (Math/sqrt (/ 1 (Math/sqrt lambda))) lambda)))))

(def radius
   #^"Computes the radius of projection from a given lambda"
   (memoize
      (fn [lambda] (/ 1 (Math/sqrt lambda)))))

(defn rate
   "Returns the learning rate for the given lambda parameter and time t"
   [t lambda]
   (/ 1 (* lambda (+ t (offset lambda)))))

(defn predict
   "Returns the real-valued prediction for the given model on the given input"
   [model input] (inner model input))

(defn correct
   "Returns a corrected version of the weight vector w"
   [w v x y t loss-fn lambda proj-rate]
   (prof :correct
      (let [eta    (rate t lambda)
            dloss  (loss/deriv loss-fn y v)
            w-new  (add
                     (scale (- 1 (* eta lambda)) w)
                     (scale (* (- eta) dloss) x))  ]
         (if (zero? (mod t proj-rate))
            (project
               w-new
               (radius lambda))
            w-new))))

(defn step
   "Returns an updated model by taking the last model, the next training 
    and applying the Pegasos update step"
   [model t example loss-fn lambda proj-rate]
   (let [x (:x example)
         y (:y example)
         v (predict model x)]
      (if (> (loss-fn y v) 0)
         (correct model v x y t loss-fn lambda proj-rate)
         model)))

(defn make-step
   "Returns an update function with given learning rate"
   [lambda proj-rate]
   (fn [w [t example]] (step w t example loss/hinge lambda proj-rate)))

(defn make-initial
   "Creates an initial model for an SGD run"
   [] {})

(defn make-learner
   "Creates an SVM SGD learner with the given regularisation parameter lambda"
   [lambda projection-rate]
   (struct-map learner
      :name       (print-str 
                     "Pegasos ( lambda = " lambda 
                        ", projections = " projection-rate ")")
      :initial    (create [])
      :predict    predict
      :step       (make-step lambda projection-rate)
   ))
