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
      (cern.colt.matrix.tdouble DoubleMatrix1D DoubleFactory1D DoubleMatrix2D DoubleFactory2D)
      (cern.jet.math.tdouble DoubleFunctions)
      (edu.emory.mathcs.utils ConcurrencyUtils)))

;; ---- Constants ----
(def *dense-1d* DoubleFactory1D/dense)
(def *dense-2d* DoubleFactory2D/dense)
(def *dense-ops* DenseDoubleAlgebra/DEFAULT)

(def *sparse-1d* DoubleFactory1D/sparse)
(def *sparse-2d* DoubleFactory2D/sparse)
(def *sparse-ops* SparseDoubleAlgebra/DEFAULT)

;; ---- Matrix Operations ----
(defn subvector
   "Returns the subvector consisting of rows of v with indicies in s"
   [v s] (.viewSelection v s ))

(defn submatrix
   "Returns the submatrix consisting of rows and cols of m with indicies in 
    the given collection s (repeated if necessary)"
   ([m s]      (submatrix m s s))
   ([m s1 s2]  (.viewSelection m (int-array s1) (int-array s2))))

(defn invert
   "Returns the (pseudo-) inverse of the given matrix"
   [matrix] (.inverse *dense-ops* matrix))

(defn add!
   "Overrides and returns A <- A + B"
   [A B] (.assign A B DoubleFunctions/plus) )

(defn mult!
   "Overrides and returns C <- alpha A x B + beta C."
   [A B C alpha beta] (.zMult A B C alpha beta false false))

;; ---- NPCA Algorithm ----
(defn update-k 
   "Overrides and returns K <- K + 1/m KBK"
   [K B m]
   (let [BK (mult! )])
   )
