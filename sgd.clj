;; Stochatistic Gradient Descent in Clojure
;;
;; AUTHOR: Mark Reid <mark@reid.name>
;; CREATED: 2009-07-08

; Basic online convex optimisation (OCO) game is as follows:
;	1.	Repeat
;	1.1		Player makes an play p_t
;	1.2		World gives player response r_t
;	1.3		Player incurs some cost l(p_t,r_t)

(ns sgd)

; Convert a file to a lazy sequence of lines.
; (-> "sgd.clj" FileReader. BufferedReader. line-seq)

; World will wrap this lazy sequence and use it for its responses.
(defn response
	"Returns the next example from a sequence, ignoring the input"
	[]
	)

(defn oco-step
	"Performs a single step of the OCO game and returns the loss incured"
	[]
	(let [pt (play player)
		  rt (respond world pt)]
		(loss pt rt)))

(defn oco
	"Runs an OCO game"
	[]
	(iterate 
		()))

(defn player-create 
	[start loss]
	(agent {:prediction start, :loss loss}))

(defn player-prediction [player] (:prediction @player))

(defn game-step
	"In a game step the player makes a prediction and is then given an
	 observation by the world (possibly adversarially)"
	[player world]
	(let [observation ])
	(update player ))
	
(defn game
	"A game is a process that takes a learning rule and an environment and 
	 sequentially shows the rule an instance, waits for its prediction then
	 shows it the instance label."
	[rule environment]
	nil)
	
