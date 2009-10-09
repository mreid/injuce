;; Online algorithm runner
;;
;; AUTHOR: Mark Reid <mark@reid.name>
;; CREATED: 2009-10-08

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
; * Parsing all of the examples in the full 781,000+ data set takes 2:40 on my
;   2.66GHz 2Gb RAM iMac
;
; * Memory uses rises quickly to around 160Mb of real memory then stablises.
;
; To Do
; -----
; * Optimise gradient calculations in SGD via rearranging calls to add & scale.
;
; * Improve the parsing speed so that lines are parsed char-by-char into maps.
;
; * Make the model vector in SGD dense and update vector methods to handle
;   sparse/dense updates. 

(ns run
   (:use (clojure.contrib profile)))

(def *num-test*    100)
(def *num-train*   800000)
(def *report-freq* 10000)

(def *lambda* 0.0001)
(def *projection-freq* 100)

(ns clojure.contrib.profile)
(def *enable-profiling* false)

(ns run
   (:require data learner sgd))

(time
;   (profile
      (learner/train 
         (sgd/make-learner *lambda* *projection-freq*) 
         (take (+ *num-train* *num-test*) (data/stdin))
         *num-test*    
         *report-freq*));)
