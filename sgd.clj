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

; Parsing all of the examples in the full 781,000+ data set takes 2:40 on my
; 2.66GHz 2Gb RAM iMac

; The current version (2009-09-01) is still surprisingly memory-hungry - though
; much less than earlier version. I suspect this is because all the examples 
; use new hashes/SparseDoubleMatrix1D instances.

; 2009-09-06: Changed (map parse ...) into a while loop that read from *in*
;             On train2000 data set, new version uses a constant 25Mb while 
;             older version steadily grew to 30Mb. 

; 2009-09-08: Current version now behaves very similarly to Bottou's SVMSGD.
;             Did this to track down problem causing lack of convergence.
;             Turns out the problem was that I was counting an error when
;             the loss was positive instead of when the margin was positive
;             which was causing a drastic overcount of the error.

; TODO: See if the entire data set can be stored in memory as sparse vectors.
; TODO: Now that the error assessment is fixed, try converting back to more
;       functional style and go for speed.

(ns sgd (:import 
      (java.io FileReader BufferedReader Reader)))

;; ---- Sparse Vector Operations ----
(def *max-features* 47153)

(defn cardinality
   "Returns the number of non-zero elements in the given sparse vector"
   [v] (count v))

(defn add
   "Returns the sparse sum of two sparse vectors x and y" 
   [x y] (merge-with + x y))

(defn inner
   "Returns the inner product of the vectors x and y"
   [x y]
   (reduce + (map #(* (get x % 0) (get y % 0)) (keys y))))

(defn norm
   "Returns the l_2 norm of the (sparse) vector v"
   [v] (Math/sqrt (inner v v)))

(defn pointwise
   "Returns the vector resulting from applying f to the values in v"
   [f v] 
   (reduce
      ; Anonymous function to add [key f(val)] to result map
      (fn [result entry] (assoc result (key entry) (f (val entry))))
      {} v))

(defn scale 
   "Returns the vector of v that is scaled by the float a"
   [a v] (pointwise #(* a %) v))

(defn project
   "Scales x so it is inside the ball of radius r"
   [x r] 
   (let [n (norm x)]
      (if (> n r) 
         (scale (/ r n) 
         x))))

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
(defn correct
   "Returns a corrected version of the weight vector w"
   [w example t lambda]
   (let [x        (:x example)
         y        (:y example)
         eta (/ 1 (* lambda t))
         s   (- 1 (* eta lambda))
         r   (/ 1 (Math/sqrt lambda))
         w1  (scale (- 1 (/ 1 t)) w)
         ]
      (project (add w1 (scale (* eta y) x)) r)))

;(defn correct
;   "Returns a corrected version of the weight vector w"
;   [example]
;   (let [x   (:x example)
;         y   (:y example)
;         eta (/ 1 (* @lambda (+ @step offset)))
;         s   (- 1 (* eta @lambda))
;         r   (/ 1 (Math/sqrt @lambda))]
;      (do
;         (scale s w)
;         (scale (* eta y) x)
;         (add w x)
;         (reset! bias (+ @bias (* eta y 0.01)))
;;         (project w r)
;      )))


(defn report
   "Prints stats about the given model to STDERR at the specified interval"
   [model interval]
   (if (zero? (mod (:step model) interval))
      (let [t      (:step model)
           size   (cardinality (:w model))
           errors (:errors model) ]
         (binding [*out* *err*]
            (println "Step:" t 
             "\t Features in w =" size 
             "\t Errors =" errors 
             "\t Accuracy =" (/ (float errors) t))))))
 
(defn update
   "Returns an updated model by taking the last model, the next training 
    and applying the Pegasos update step"
   [model example]
   (let [lambda (:lambda model)
        t      (:step   model)
        w      (:w      model)
        errors (:errors model)
        error  (> (hinge-loss w example) 0)]
      (do 
         (report model 100)
         { :w      (if error (correct w example t lambda) w), 
           :lambda lambda, 
           :step   (inc t), 
           :errors (if error (inc errors) errors)} )))
 
(defn train
   "Returns a model trained from the initial model on the given examples"
   [initial examples]
   (reduce update initial examples))
 
;; ---- Main method ----
 
(defn main
	"Trains a model from the examples and prints out its weights"
	[]
	(let [start 	{:lambda 0.0001, :step 1, :w (create {}), :errors 0} 
		  examples 	(map parse (-> *in* BufferedReader. line-seq)) 
		  model     (train start examples)]
		(println (:w model))))
;; ---- Global variables for training ----
;(def w      (DenseFloatMatrix1D. *max-features*))
;(def bias   (atom 0.0))
;(def errors (atom 0))
;(def lambda (atom 0.0001))
;(def step   (atom 0))

;; Offset for time step based on code in Bottou's svmsgd.
;(def eta0   (Math/sqrt (/ 1 (Math/sqrt @lambda))))
;(def offset (/ 1 (* eta0 @lambda)))

(defn margin
   "Returns the margin y.(<w,x> + bias)"
   [example] (* (:y example) (inner w (:x example))))

(defn hinge-loss
   "Returns the hinge loss for the margin value z"
   [z] (max 0 (- 1 z)))
   
(defn report
   "Prints stats about the given model to STDERR at the specified interval"
   [interval]
   (if (zero? (mod @step interval))
      (binding [*out* *err*]
         (println 
            "Step:" @step 
             "\t Features in w =" (cardinality w)
             "\t |w| =" (norm w)      
             "\t Errors =" @errors 
             "\t Cumm. Acc. =" (- 1 (/ (float @errors) (inc @step)))))))

(defn update
   "Updates the model by taking an example and applying the Pegasos update step"
   [example]
   (let [z (margin example)]
      (do 
         (report 1000)
         (if (> (hinge-loss z) 0)
            (do 
               (if (<= z 0) (reset! errors (inc @errors)))
               (correct example)))
         (reset! step (inc @step)))))

(defn train
   "Parses STDIN, converting each line into an example and updating the model."
   [examples]
   (doall (map #(do (update %) %) examples)))

(defn err [example] (if (<= (margin example) 0) 1 0))

(defn testmodel
   "Applies the model w to the given examples and calculates the error"
   [examples]
   (reduce + (map #(err %) examples)))

;; ---- Main ----
;; Execute the program
(set! *warn-on-reflection* true)
(ConcurrencyUtils/setNumberOfThreads 1) ; Done to stop time wasted in Futures
(def test-set (atom []))

(if (== 2 (count *command-line-args*))
   (do
      (let [test-size (Integer/parseInt (nth *command-line-args* 1))]
         (println "Training (using first" test-size "examples for testing)...")
         (time 
            (while (.ready #^Reader *in*)
               (let [example (parse (read-line))]
                  (if (>= test-size (count @test-set))
                     (swap! test-set conj example) ; Save a test example or
                     (update example)))))          ; ... update the model
         (println "...completed!")
         (println "Training error rate =" (* (/ @errors @step) 100.0) "%")
         (println "Testing on first" test-size "examples...")
         (let [test-errors (testmodel @test-set)]
            (println "Test Error Rate: " (* (/ test-errors test-size) 100.0) "%")))))
