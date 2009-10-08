;; Structures and functions for general online learners
;;
;; AUTHOR: Mark Reid <mark@reid.name>
;; CREATED: 2009-10-08

; Basic online convex optimisation (OCO) game is as follows:
;  1. Repeat
;  1.1      Player makes an play p_t
;  1.2      World gives player response r_t
;  1.3      Player incurs some cost l(p_t,r_t)
;
(ns learner
   (:require loss))

; All online algorithms must provide the following:
;  1. A function (step model [time example]) which returns a new model 
;  2. A function (predict model input) which returns a prediction for the input 
;  3. An initial model
(defstruct learner 
   :name    ; A string identifier for the learner
   :step    ; Performs an update. Type: model [time example] -> model 
   :predict ; Performs a prediction. Type: model input -> float
   :initial ; An initial model
)

; ---- Evaluation ----
(defn error
   [model predict example]
   (loss/zero-one (predict model (:x example)) (:y example)))

(defn summarise
   "Returns summary statistics for the given map of numbers"
   [xs]
   (let [n     (count xs)
         sum   (reduce + 0.0 xs)]
      {:mean   (/ sum n)
       :count  n
       :total  sum}))

(defn evaluate
   "Returns test statistics for a model and prediction fn on the test examples"
   [model predict examples]
   (summarise (map #(error model predict %) examples)))

; ---- Reporting ----
(defn make-reporter
   "Create a function that reports on model performance on test every n steps"
   [n predict-fn test-exs]
   (fn [model time]
         (if (zero? (mod time n))
            (do
               (println "Step" time)
               (println (evaluate model predict-fn test-exs))
            ))))

; ---- Training ----
(defn index
   "Returns a lazy sequences of pairs [1 x1] [2 x2] ... given [x1 x2]"
   [xs] (map (fn [a b] [a b]) (iterate inc 1) xs))

(defn train
   "Returns a model trained by learner on the given examples"
   [learner examples]
   (let [test-exs  (take 1000 examples)
         train-exs (drop 1000 examples)
         reporter  (make-reporter 1000 (:predict learner) test-exs)]
      (reduce 
         (fn [model [time example]]
            (do
               (reporter model time)
               ((:step learner) model [time example])
            ))
         (:initial learner) 
         (index train-exs)
      )))
