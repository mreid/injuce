;; Stochatistic Gradient Descent in Clojure
;;
;; AUTHOR: Mark Reid <mark@reid.name>
;; CREATED: 2009-07-08

(ns sgd)

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
	
