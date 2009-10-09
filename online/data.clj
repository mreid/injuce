;; Parsing and data management
;;
;; AUTHOR: Mark Reid
;; CREATED: 2009-10-08

(ns data
   (:import 
      (java.io FileInputStream InputStreamReader BufferedReader)
      (java.util.zip GZIPInputStream))
   (:use (clojure.contrib profile))
   (:require vec))

;; ---- Parsing ----
;; TODO: Make this faster by removing regular expressions and building map
;;       as characters are scanned
(defn parse-feature 
   [string] 
   (let [ [_ key val] (re-matches #"(\d+):(.*)" string)]
      [(Integer/parseInt key) (Float/parseFloat val)]))

(defn parse-features
   "Parses a string of the form 'int:float' into a sequence of pairs [int float]"
   [string]
   (vec/create (map parse-feature (re-seq #"[^\s]+" string))))

(defn parse
   "Returns a map {:y label, :x sparse-feature-vector} parsed from given line"
   [line number]
   (prof :parse
      (let [ [_ label features] (re-matches #"^(-?\d+)(.*)$" line) ]
         { :t number 
           :y (Float/parseFloat label) 
           :x (parse-features features) })))

(defn from-reader
   "Returns a lazy sequence of examples parsed from the given Reader"
   [reader]
   (map parse (-> reader BufferedReader. line-seq) (iterate inc 1)))

;; --- Small RCV Data Set - 2000 examples ----
(def rcv-small-file "../data/train2000.dat.gz")   
(defn rcv-small
   []
   (from-reader 
      (-> rcv-small-file FileInputStream. GZIPInputStream. InputStreamReader.)))

;; ---- Reads examples from STDIN ----
(defn stdin
   [] (from-reader *in*))
