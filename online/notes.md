Profiling Notes
===============

Thankfully, someone has already done the hard work of micro-benchmarking various implementation decisions:

    http://gnuvince.wordpress.com/2009/05/11/clojure-performance-tips/

Experiment 1
------------

Running the following at the top level:

    (profile
      (learner/train 
         (sgd/make-learner 0.001 100) 
         (take 2000 (data/stdin)))))
   
### Using Naive implementation of gradient update in correct

	   Name      mean       min       max     count       sum
	    add   2338345     20000  15110000      1114  2604917000
	correct  12456579   1833000  154048000       557  6938315000
	 dhinge      7890      6000     35000       557   4395000
	  hinge      3482      1000     53000      1557   5422000
	  inner     97594      9000  12283000      2005  195676000
	  parse    906810     27000  97937000      2000  1813620000
	project   5229600   3045000   7356000         5  26148000
	  scale   2535123     15000  144338000      1674  4243797000

### Algebraic manipulation of gradient update to minimise scaling

	   Name      mean       min       max     count       sum
	    add    160784     26000   6024000       557  89557000
	correct   4481694    796000  141773000       557  2496304000
	 dhinge      8389      6000     62000       557   4673000
	  hinge      2924      1000     53000      1557   4553000
	  inner    155485     11000  43635000      2005  311749000
	  parse    822927     28000  93411000      2000  1645854000
	project   7660000   3414000  16781000         5  38300000
	  scale   2086476     15000  141573000      1117  2330594000

### Change to `inner` so it destructures the map into `[k v]` 

	   Name      mean       min       max     count       sum
	    add    129289     29000   5447000       557  72014000
	correct   4408249    674000  147886000       557  2455395000
	 dhinge      8463      6000    268000       557   4714000
	  hinge      3049      1000    256000      1557   4748000
	  inner    103973      7000  13144000      2005  208466000
	   norm   5306800   2422000  13169000         5  26534000
	  parse    893034     33000  93633000      2000  1786069000
	project   6854800   3982000  13177000         5  34274000
	  scale   2071798     15000  147741000      1117  2314199000

### Prior to new `vec` optimisations (on Work iMac)


    Step 1000
    {:mean 0.106, :count 1000, :total 106.0}
       Name      mean       min       max     count       sum
        add    136642     36000   3047000       557  76110000
    correct   4332317    664000  156838000       557  2413101000
     create    617012     18000  28319000      2000  1234024000
     dhinge      7789      6000     94000       557   4339000
      hinge      3132      1000     47000      1557   4877000
      inner    112597      8000  10918000      2005  225757000
       norm   4778200   2552000  10951000         5  23891000
      parse    682077     32000  36173000      2000  1364154000
    project   7469600   3572000  11448000         5  37348000
      scale   2023675     19000  156687000      1117  2260446000
    "Elapsed time: 4176.285 msecs"



Other changes
-------------
* Changed `assoc` call in `vec/pointwise` to `conj` and noticed no improvement.