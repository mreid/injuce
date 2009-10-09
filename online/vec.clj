;; Sparse vectors and operations
;;
;; AUTHOR: Mark Reid <mark@reid.name>
;; CREATED: 2009-10-08
(ns vec
   (:use (clojure.contrib profile)))

(defstruct svec 
   :entries       ; Map of entries from int -> flot
   :norm2         ; The squared norm of the vector
   :scale         ; The scaling factor for the vector
)

(defn- sq
   "Returns the square of x"
   [x] (* x x))

(defn- svec-inc
   "Updates a map of entries and its squared norm with a new int->float."
   [[entries norm2] [k v]]
   (if (zero? v)
      [entries norm2]
      (if (contains? entries k)
         (let [old (get entries k)]
            [ (conj entries [k v]) (+ (- norm2 (sq old)) (sq v)) ])
         [ (conj entries [k v]) (+ norm2 (sq v)) ] )))

(defn create
   "Returns a map constructed from a sequence of pairs [key value]"
   [pairs] 
   (prof :create 
      (let [[entries norm2] (reduce svec-inc [{} 0] pairs)]
         (struct svec entries norm2 1))))

;(defn create
;   "Returns a new vec built from the sequence of triples v"
;   [v] (struct svec {} 0 1))

(defn cardinality
   "Returns the number of non-zero elements in the given sparse vector"
   [v] (count (:entries v)))

(defn- add-inc
   [xscale yscale]
   (fn [[xmap xnorm2] [yk yv]]
      (if (zero? yv)
         [xmap xnorm2]
         (let [syv   (* (/ yv yscale) xscale)]
            (if (contains? xmap yk)
               (let [xv    (get xmap yk)
                     nxv   (+ xv syv)
                     norm2 (+ (- xnorm2 (sq xv)) (sq (+ xv syv)))]
                  (if (zero? nxv)
                     [ (dissoc xmap yk) norm2 ]
                     [ (conj xmap [yk nxv]) norm2 ] ))
               [ (conj xmap [yk syv]) (+ xnorm2 (sq syv)) ])))))

(defn- add-update
   [x y]
   (reduce 
      (add-inc (:scale x) (:scale y))
      [ (:entries x) (:norm2 x) ]
      (:entries y)))

(defn add
   "Returns the sparse sum of two sparse vectors x and y.
    (More efficient if sparsest vector is y)" 
   [x y] 
   (prof :add 
      (let [ [entries norm2] (add-update x y) ]
         (struct svec entries norm2 (:scale x)))))

(defn inner
   "Returns the inner product of the vectors x and y.
    (More efficient if sparsest vector is y)"
   [x y]
   (prof :inner
      (let [xmap  (:entries x)
            ymap  (:entries y)
            scale (* (:scale x) (:scale y)) ]
         (* scale
            (reduce + 
               (map 
                  (fn [[k v]] (* (get xmap k 0) v))
                  ymap))))))

(defn norm
   "Returns the l_2 norm of the (sparse) vector v"
   [v] 
   (prof :norm
      (* (Math/abs (:scale v)) (Math/sqrt (:norm2 v)))))

(defn scale 
   "Returns the vector of v that is scaled by the float a"
   [a v] 
   (prof :scale 
      (if (zero? a)
         (struct svec {} 0 1)
         (struct svec (:entries v) (:norm2 v) (* a (:scale v))))))

(defn sub
   "Returns the vector = x - y"
   [x y] (add x (scale -1 y)))

(defn project
   "Scales x so it is inside the ball of radius r"
   [x r] 
   (prof :project
      (let [n (norm x)]
         (if (> n r) 
            (scale (/ r n) x)
            x))))