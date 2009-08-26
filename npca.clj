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
(ns npca 
   (:import 
      (java.io FileReader BufferedReader)
      (cern.colt.matrix.tdouble.algo DenseDoubleAlgebra SparseDoubleAlgebra)
      (cern.colt.matrix.tdouble DoubleMatrix2D DoubleFactory2D)
      (cern.colt.matrix.tfloat.impl SparseFloatMatrix1D)
      (cern.jet.math.tfloat FloatFunctions)
      (edu.emory.mathcs.utils ConcurrencyUtils)))

;; ---- Constants ----
(def *dense* DoubleFactory2D/dense)
(def *dense-ops* DenseDoubleAlgebra/DEFAULT)

(def *sparse* DoubleFactory2D/dense)
(def *sparse-ops* SparseDoubleAlgebra/DEFAULT)

;; ---- Vector Operations ----
(defn subvector
   "Returns the subvector consisting of rows of v with indicies in s"
   [v s]
   ())

;; ---- Matrix Operations ----

(defn submatrix
   "Returns the submatrix consisting of rows and cols of m with indicies in s"
   [m s]
   (let [s-ints (int-array s)]
      (.viewSelection m s-ints s-ints)))

(defn invert
   "Returns the (pseudo-) inverse of the given matrix"
   [matrix] (.inverse *dense-ops* matrix))
