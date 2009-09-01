;; Stochatistic Gradient Descent in Clojure
;;
;; AUTHOR: Mark Reid <mark@reid.name>
;; CREATED: 2009-07-08

; Basic online convex optimisation (OCO) game is as follows:
;  1. Repeat
;  1.1      Player makes an play p_t
;  1.2      World gives player response r_t
;  1.3      Player incurs some cost l(p_t,r_t)
;
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
;        w'_t <- (1-\eta_t \lambda) w_t + \eta_t / k \sum_{(x,y)\in A_t} y x
;        w_{t+1} <- \min{ 1, \lambda^{-1/2}/||w'_t|| } w'_t
;     Output w_{T+1}
;
; The particular problem I'm going to solve here is the RCV1-V2 Reuters problem
; described by Léon Bottou here:
;
;  http://leon.bottou.org/projects/sgd
;
; This is a problem with 781265 training instances, 23149 test instances
; and 47152 parameters / features. The data used as input is the output of
; Bottou's `preprocess` command (see above URL for details). The format is
; one training example per line that looks like:
;
;  y f_1:v_1 f_2:v_2 ... f_n:v_n
;
; where y is either 1 or -1, each f_i is an integer representing a feature index
; and each v_i a float.
;
; The suggested value of lambda for this dataset is 0.0001.

; Notes
; -----
; Current version (2009-09-01) uses atoms and global variables to do inplace 
; rather than the older functional style updates. This is because the old 
; version was chewing way to much memory. I suspect that even though Clojure has
; some nice tricks to ensure deltas to existing things like maps do not cost
; too much, they cannot handle the amount of modification an online SGD needs.

; The older version also had some nasty bugs:
;  1. The projection step was never happening
;  2. The scaling of w in the correction step was wrong
;  3. No bias term

; These are now fixed (hopefully).

; The current version (2009-09-01) is still surprisingly memory-hungry - though
; much less than earlier version. I suspect this is because all the examples 
; use new hashes/SparseDoubleMatrix1D instances.

; TODO: Use a single sparse vector for all example and just clear/replace

(ns sgd (:import 
      (java.io FileReader BufferedReader)
      (cern.colt.matrix.tfloat.impl DenseFloatMatrix1D SparseFloatMatrix1D)
      (cern.jet.math.tfloat FloatFunctions)
      (edu.emory.mathcs.utils ConcurrencyUtils)))

;; ---- Sparse Vector Operations ----
(def *max-features* 47153)

(defn create
   "Returns a sparse vector created from a list of int/float pairs [k v]"
   [m] 
   (let [v (SparseFloatMatrix1D. (inc *max-features*))] 
      (do 
         (doseq [[i f] m] (.set v i f))
         (.set v *max-features* 1.0)      ; Bias term
         v)))
         
(defn cardinality
   "Returns the number of non-zero elements in the given sparse vector"
   [v] (.cardinality v))

(defn add [x y] (.assign x y FloatFunctions/plus))

(defn inner [x y] (.zDotProduct x y))

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

;; ---- Parsing ----

(defn parse-feature 
   [string] 
   (let [ [_ key val] (re-matches #"(\d+):(.*)" string)]
      [(Integer/parseInt key) (Float/parseFloat val)]))

(defn parse-features
   [string]
   (create (map parse-feature (re-seq #"[^\s]+" string))))

(defn parse
   "Returns a map {:y label, :x sparse-feature-vector} parsed from given line"
   [line]
   (let [ [_ label features] (re-matches #"^(-?\d+)(.*)$" line) ]
      {:y (Float/parseFloat label), :x (parse-features features)}))

;; ---- Training ----
;; ---- Global variables for training ----
(def w      (DenseFloatMatrix1D. (inc *max-features*)))  ; +1 for bias term
(def step   (atom 1))
(def errors (atom 0))
(def lambda (atom 0.0001))

(defn hinge-loss
   "Returns the hinge loss of the weight vector w on the given example"
   [example] (max 0 (- 1 (* (:y example) (inner w (:x example))))))
   
(defn correct
   "Returns a corrected version of the weight vector w"
   [example]
   (let [x   (:x example)
         y   (:y example)
         eta (/ 1 (* @lambda @step))
         r   (/ 1 (Math/sqrt @lambda))]
      (do
         (scale (/ (- @step 1) @step) w)
         (scale (* eta y) x)
         (add w x)
         (project w r)
      )))

(defn report
   "Prints stats about the given model to STDERR at the specified interval"
   [interval]
   (if (zero? (mod @step interval))
      (binding [*out* *err*]
         (println 
            "Step:" @step 
             "\t Features in w =" (cardinality w) 
             "\t Errors =" @errors 
             "\t Accuracy =" (/ (float @errors) @step)))))

(defn update
   "Updates the model by taking an example and applying the Pegasos update step"
   [example]
   (let [loss (hinge-loss example)]
      (do 
         (if (> loss 0) 
            (do 
               (correct example)
               (reset! errors (inc @errors))))
         (reset! step (inc @step)))))

(defn train
   "Returns a model trained from the initial model on the given examples"
   [examples]
   (doseq [example examples]
      (update example)
      (report 1000)))

;; ---- Main method ----

(defn main
   "Trains a model from the examples and prints out its weights"
   []
   (train (map parse (-> *in* BufferedReader. line-seq))))

(set! *warn-on-reflection* true)
(ConcurrencyUtils/setNumberOfThreads 1) ; Done to stop time wasted in Futures
(println (time (main)))

; Time how long it takes just to parse input
;(prn (time (count (map parse (-> *in* BufferedReader. line-seq)))))
