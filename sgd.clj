;; Stochatistic Gradient Descent in Clojure
;;
;; AUTHOR: Mark Reid <mark@reid.name>
;; CREATED: 2009-07-08

; Basic online convex optimisation (OCO) game is as follows:
;	1.	Repeat
;	1.1		Player makes an play p_t
;	1.2		World gives player response r_t
;	1.3		Player incurs some cost l(p_t,r_t)
;
; The Pegasos algorithm (Shalev-Shwartz, Singer & Srebro, ICML 2007) solves
; the above problem for hinge loss with l_2 regularisation (i.e., SVM) as 
; follows:
;
; PEGASOS
;	Given: 
;		S 		- training examples
;		\lambda	- regularisation factor
;		T		- number of iterations
;		k		- size of subset of S to use for gradient computation
;
;   Initialise:
;		Choose w_1 s.t. ||w_1|| ≤ \lambda^{-1/2}
;
;	Do the following:
;		For t = 1, 2, ..., T do
;			\eta_t <- 1/(\lambda t)
;			A_t  <- Some subset of k training examples from S
;			B_t  <- { x \in A_t : y <w_t, x> < 1 } (i.e., misclassified examples)
;			w'_t <- (1-\eta_t \lambda) w_t + \eta_t / k \sum_{(x,y)\in A_t} y x
;			w_{t+1} <- \min{ 1, \lambda^{-1/2}/||w'_t|| } w'_t
;		Output w_{T+1}
;
; The particular problem I'm going to solve here is the RCV1-V2 Reuters problem
; described by Léon Bottou here:
;
;	http://leon.bottou.org/projects/sgd
;
; This is a problem with 781265 training instances, 23149 test instances
; and 47152 parameters / features. The data used as input is the output of
; Bottou's `preprocess` command (see above URL for details). The format is
; one training example per line that looks like:
;
;	y f_1:v_1 f_2:v_2 ... f_n:v_n
;
; where y is either 1 or -1, each f_i is an integer representing a feature index
; and each v_i a float.
;
; The suggested value of lambda for this dataset is 0.0001.

(ns sgd)
(import '(java.io FileReader BufferedReader))

; Convert a file to a lazy sequence of lines.
; (-> "sgd.clj" FileReader. BufferedReader. line-seq)

(defn add
	"Returns the sparse sum of two sparse vectors x y"
	[x y] (merge-with + x y))

(defn inner
	"Computes the inner product of the sparse vectors (hashes) x and y"
	[x y]
	(reduce + (map #(* (get x % 0) (get y % 0)) (keys y))))

(defn margin
	"Returns the margin of the parameter vector w on instance x with label y"
	[w x y] (* y (inner w x)))

(defn norm
	"Returns the l_2 norm of the (sparse) vector v"
	[v] (Math/sqrt (inner v v)))

(defn scale
	"Returns the scalar product of the sparse vector v by the scalar a"
	[a v] (zipmap (keys v) (map * (vals v) (repeat a))))

(defn project
	"Returns the projection of a parameter vector w onto the ball of radius r"
	[w r]
	(let [w-norm (norm w)]
		(scale (if (> w-norm r) (/ r w-norm) 1) w)))

(defn update
	"Returns an updated model by taking the last model, the next training 
	 and applying the Pegasos update step"
	[model example]
	(let [lambda (:lambda model)
		  t      (:step   model)
		  wt     (:w      model)
		  errors (:errors model)
		  y      (:y      example)
		  x      (:x      example)
		  error  (< (margin wt x y) 1)
		
		  eta    (/ 1 (* lambda t))
		  wt1    (scale (- 1 (* eta lambda)) wt) 
		  neww   (project
					(if error (add wt1 (scale y x)) wt1)
					(/ 1 (Math/sqrt lambda)))]
		(do (if (== (mod t 100) 0) (prn t errors (norm wt) (margin wt x y)))
		{ :w      neww, 
		  :lambda lambda, 
		  :step (inc t), 
		  :errors (if error (inc errors) errors)} )))

(defn train
	"Returns a model trained from the initial model on the given examples"
	[initial examples]
	(reduce update initial examples))

(defn parse-feature 
	[string] 
	(let [ [_ key val] (re-matches #"(\d+):(.*)" string)]
		[(Integer/parseInt key) (Float/parseFloat val)]))
	
(defn parse-features
	[string]
	(into {} (map parse-feature (re-seq #"[^\s]+" string))))

(defn parse
	"Returns a map {:y label, :x sparse-feature-vector} parsed from given line"
	[line]
	(let [ [_ label features] (re-matches #"^(-?\d+)(.*)$" line) ]
		{:y (Float/parseFloat label), :x (parse-features features)}))

(defn main
	"Call to run the example"
	[]
	(let [start 	{:lambda 0.0001, :step 1, :w {}, :errors 0} 
		  examples 	(map parse (-> *in* BufferedReader. line-seq))
		  model		(train start examples) ]
		[(count (:w model)), (:errors model)] ))

;(with-open
;	[file (BufferedReader. (FileReader.  "test.data"))]
;	(prn (train  {:lambda 0.1, :step 1, :w {}} (map parse (line-seq file)))))

; Time how long it takes to parse input
;(prn (time (count (map parse (-> *in* BufferedReader. line-seq)))))

(set! *warn-on-reflection* true)
(prn (time (main)))
