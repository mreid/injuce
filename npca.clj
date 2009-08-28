;; Fast NPCA 
;; =========
;; An implementation of Fast Non-parametric Principle Component Analysis based
;; on the description of Algorithm 3 in [1].
;;
;; Problem Description
;; -------------------
;; An m x n matrix Y describes m users' numerical rankings on n items.
;; The aim of NPCA is to find a low-rank approximation X for Y as the product of
;; two factors: 
;; 
;;          Y ≈ U.V' = X
;; 
;; where U is m x k and V is n x k for k << min(m,n).
;;
;; The rows of the matrix U are the k-dimensional vectors u_i, for i=1...m and
;; similarly v_i for V.
;;
;; Probabilistic PCA assumes each entry in Y is some noisy transformation of
;; unknown, latent variables. That is:
;; 
;;          Y[i,j]   = u_i' v_j + e[i,j]  where  e[i,j] ~ N(0,\lambda)
;; 
;; and it is assumed that the prior over the k-dimensional vectors U[i,.]
;; are N(0,I).
;;
;; Solution
;; --------
;; NPCA solves the above problem by implementing an EM-like algorithm:
;;
;; E-step: [compute sufficient stats. for p(x_i|y, K, \lambda)]
;;
;;       E(x_i) = K[:,Ob_i](K_{Ob_i} + \lambda I)^{-1} y_{Ob_i}
;;     Cov(x_i) = K - K[:,Ob_i](K_{Ob_i} + \lambda I)^{-1} K[Ob_i,:]
;;
;; M-step: [update parameters based on results of E-step]
;;
;;          K <- 1/M \sum_{i=1}^M [ Cov(x_i) + E(x_i).E(x_i)' ]
;;    \lambda <- 1/|Ob| \sum_{(i,j)\in Ob} [ C[i,j] + ( Y[i,j] - E(X[i,j]) )^2 ]
;;
;;   where: 
;;       Ob = set of index pairs (i,j) for non-missing entries in Y
;;       Ob_i = non-zero (Observed) indicies for row i of Y
;;       K[:,Ob_i] = submatrix of K of rows with column indices in Ob_i
;;       K[Ob_i,:] = submatrix of K of columns with row indices in Ob_i
;;
;; References
;; ----------
;; [1]   Fast Nonparametric Matrix Factorization for Large-Scale Collaborative
;;       Filtering. K. Yu, S. Zhu, J. Lafferty and Y. Gong, SIGIR 2009.
;;       <http://www.cs.cmu.edu/~lafferty/pdfs/sigir469-yu.pdf>

(set! *warn-on-reflection* true)

;; ---- General Matrices ----
(ns matrix
   (:import
      (cern.colt.matrix.tdouble.algo DenseDoubleAlgebra)
      (cern.colt.matrix.tdouble DoubleMatrix1D DoubleMatrix2D)
      (cern.jet.math.tdouble DoubleFunctions)))

(def *dense-ops*  DenseDoubleAlgebra/DEFAULT)

;; ---- Vector Interrogators ----
(defn size
   "Returns the number of elements in the given vector"
   [#^DoubleMatrix1D M] (.size M))

;; ---- Matrix Interrogators ----
(defn rows
   "Returns the number of rows in the given matrix"
   [#^DoubleMatrix2D M] (.rows M))

(defn cols
   "Returns the number of columns in the given matrix"
   [#^DoubleMatrix2D M] (.columns M))

;; ---- Matrix Operations ----
;(defn subvector
;   "Returns the subvector consisting of rows of v with indicies in s"
;   [v s] (.viewSelection v s ))
;
;(defn submatrix
;   "Returns the submatrix consisting of rows and cols of m with indicies in 
;    the given collection s (repeated if necessary)"
;   ([m s]      (submatrix m s s))
;   ([m s1 s2]  (.viewSelection m (int-array s1) (int-array s2))))

;(defn invert
;   "Returns the (pseudo-) inverse of the given matrix"
;   [matrix] (.inverse *dense-ops* matrix))
;
;(defn add!
;   "Overrides and returns A <- A + B"
;   [A B] (.assign A B DoubleFunctions/plus) )
;
;(defn sub!
;   "Overrides and returns A <- A - B"
;   [A B] (.assign A B DoubleFunctions/minus) )

;(defn mult!
;   "Overrides and returns C <- alpha A x B + beta C. A is a matrix, and B and C 
;    can be a matrix or vector."
;   ([A B C alpha beta]  (.zMult A B C alpha beta false false))
;   ([A B C alpha]       (mult! A B C alpha 0))
;   ([A B C]             (mult! A B C 1 0)))

;; ---- Sparse Matrix Algebra ----
(ns sparse
   (:import 
      (cern.colt.matrix AbstractMatrix1D AbstractMatrix2D)
      (cern.colt.matrix.tdouble.algo DenseDoubleAlgebra SparseDoubleAlgebra)
      (cern.colt.matrix.tdouble DoubleMatrix1D DoubleFactory1D DoubleMatrix2D DoubleFactory2D)
      (cern.colt.matrix.tdouble.impl SparseDoubleMatrix1D SparseDoubleMatrix2D)
      (cern.jet.math.tdouble DoubleFunctions)))
      
;; ---- Constants ----
(def *factory1d*  DoubleFactory1D/sparse)
(def *factory2d*  DoubleFactory2D/sparse)
(def *sparse-ops* SparseDoubleAlgebra/DEFAULT)

;; ---- Matrix Constructors ----
(defn #^DoubleMatrix1D new-vector
   "Returns a new sparse vector of dimension n"
   [#^Integer n] (SparseDoubleMatrix1D. n))

(defn #^DoubleMatrix2D new-matrix
   "Returns a new sparse matrix of dimension m x n"
   ([m n]   (SparseDoubleMatrix2D. m n))
   ([n]     (new-matrix n n)))   

(defn new-identity
   "Returns a new n x n identity matrix"
   [n] (.identity *factory2d* n))

;; ---- Matrix Operations ----
;(defn sub1
;   "Returns a new sparse vector equal to a - b"
;   [a b] (matrix/sub! (.copy a) b))

(defmulti mult 
   "Returns a new sparse vector or matrix with is the product A.x"
   (fn [A x] (class x)))
   
(defmethod mult DoubleMatrix1D [#^DoubleMatrix2D A #^DoubleMatrix1D x]
   (.zMult A x (new-vector (matrix/size x))))

(defmethod mult DoubleMatrix2D [#^DoubleMatrix2D A #^DoubleMatrix2D B]
   (.zMult A B (new-matrix (matrix/rows B) (matrix/cols B))))

;; ==== Test Objects =====
(ns user)

(def A (sparse/new-matrix 3 3))
(def B (sparse/new-matrix 3 3))
(def C (sparse/new-matrix 3 3))

(def x (sparse/new-vector 3))
(def y (sparse/new-vector 3))

;(ns npca 
;   (:import 
;      (java.io FileReader BufferedReader)))

;; ---- NPCA Algorithm ----
;(def *Y* nil)  ;; This is the numUsers x numItems data matrix to be filled
;
;(defn select
;   "Returns the submatrix "
;   [])
;
;(defn npca
;   "Runs the NPCA algorithm on the matrix Y for maxIter iterations"
;   [Y maxIter]
;   (let [m  (matrix/rows Y)
;         K  (sparse/matrix m)]
;      (dotimes [iter maxIter]
;      
;         (dotimes [i m]
;            (let [G  (invert (select i K))]   
;               )
;         )
;      
;;         (update-k)
;;         (update-mu)
;      ) 
;;      [K mu]
;      ))
;
;(defn update-k 
;   "Overrides and returns K <- K + 1/m KBK"
;   [K B m]
;   (let [n  (rows K)
;         I  (sparse-identity n)        ;; Overridden below
;         S  (mult! B K I (/ 1.0 m))]   ;; S = I <- I + 1/m BK
;      (mult! K S K)))                  ;; K <- KS
