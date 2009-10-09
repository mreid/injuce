;; Sparse vectors and operations
;;
;; AUTHOR: Mark Reid <mark@reid.name>
;; CREATED: 2009-10-08
(ns vec
   (:use (clojure.contrib profile)))

(def *min-scale* 0.001)
(def *max-scale* 1000)

(defstruct svec 
   :entries       ; Map of entries from int -> flot
   :norm2         ; The squared norm of the vector
   :scale)        ; The scaling factor for the vector

; Zero vector
(def zero (struct svec {} 0 1))

(defn- sq
   "Returns the square of x"
   [x] (* x x))

(defn cardinality
   "Returns the number of non-zero elements in the given sparse vector"
   [v] (count (:entries v)))

(defn get-val
   "Returns the value of x at index k"
   [x k] (* (get (:entries x) k 0) (:scale x)))

(defn set-val
   "Returns a copy of x with the value for index k set to v"
   ( [x k v]
      (let [xmap    (:entries x)
            xn2     (:norm2   x)
            xscale  (:scale   x)
            xv      (get xmap k 0)
            scaledv (/ v xscale)
            norm2  (+ (- xn2 (sq xv)) (sq scaledv))]
         (if (zero? v)
            (struct svec xmap norm2 xscale)
            (struct svec (assoc xmap k scaledv) norm2 xscale))))

   ( [x [k v]] (set-val x k v) ))

(defn entries
   "Returns all the [key value] pairs for x"
   [x] (map (fn [k] [k (get-val x k)]) (keys (:entries x))))

(defn create
   "Returns a map constructed from a sequence of pairs [key value]"
   [pairs] 
   (prof :create 
      (reduce set-val zero pairs)))

(defn- add-once
   "Returns the vector obtained by adding the value v to entry k in x"
   [x [k v]]
   (set-val x k (+ (get-val x k) v)))

(defn add
   "Returns the sparse sum of two sparse vectors x and y.
    (More efficient if sparsest vector is y)" 
   [x y] 
   (prof :add 
      (reduce add-once x (entries y))))

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

(defn- point-mult
   [s m]
   (prof :rescale
      (reduce (fn [new-m [k v]] (conj new-m [k (* s v)])) [] m)))

(defn- rescale
   "Applies the scale factor to the entries of v and resets its scale to 1"
   [v]
   (let [s (:scale v)]
      (if (or (< (Math/abs s) *min-scale*) (> (Math/abs s) *max-scale*))
         (create (point-mult s (:entries v)))
         v)))

(defn scale 
   "Returns the vector of v that is scaled by the float a"
   [a v] 
   (prof :scale 
      (if (zero? a)
         (struct svec {} 0 1)
         (rescale (struct svec (:entries v) (:norm2 v) (* a (:scale v)))))))

(defn project
   "Scales x so it is inside the ball of radius r"
   [x r] 
   (prof :project
      (let [n (norm x)]
         (if (> n r) 
            (scale (/ r n) x)
            x))))