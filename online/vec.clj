;; Sparse vectors and operations
;;
;; AUTHOR: Mark Reid <mark@reid.name>
;; CREATED: 2009-10-08
(ns vec)

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
   "Returns the vector resulting from applying f to the values in x"
   [f x] 
   (reduce
      ; Anonymous function to add [key f(val)] to result map
      (fn [result [k v]] (assoc result k (f v)))
      {} x))

(defn scale 
   "Returns the vector of v that is scaled by the float a"
   [a v] 
   (if (zero? a)
      {}
      (pointwise #(* a %) v)))

(defn sub
   "Returns the vector = x - y"
   [x y] (add x (scale -1 y)))

(defn project
   "Scales x so it is inside the ball of radius r"
   [x r] 
   (let [n (norm x)]
      (if (> n r) 
         (scale (/ r n) x)
         x)))