;;; ================================================================
;;; MARKOV TASK MODEL
;;; ================================================================
;;; (c) 2022, Developed by Cher Yang, University of Washington
;;;           chery@uw.edu
;;; ================================================================
;;; This is an ACT-R model of the Markov Task.
;;; ================================================================
;;;
;;;  -*- mode: LISP; Syntax: COMMON-LISP;  Base: 10 -*-
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;
;;; Author      :Cher Yang
;;;
;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;
;;; Filename    :markov-model3.lisp
;;; Version     :v3.0
;;;
;;; Description : model-base + motivation
;;;
;;; Bugs        :
;;;
;;;
;;; To do       :
;;;
;;;
;;; ----- History -----
;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;
;;; General Docs:
;;;
;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;
;;; Public API:
;;;
;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;
;;; Design Choices:
;;;
;;; Task description
;;; - At state 0: a "+" appears on the screen and prepare WM. 
;;; - At stage 1: a pair of Markov Stimli; encode stimuli in WM
;;; - Planing: retrieve a memory R = 10; STATE1-LEFT-STIMULUS = ?; 
;;;
;;; CHUNK0-0
;;; STATUS  PROCESS
;;; REWARD  10
;;; STATE1-LEFT-STIMULUS  A1
;;; STATE1-RIGHT-STIMULUS  A2
;;; STATE2-LEFT-STIMULUS  B1
;;; STATE2-RIGHT-STIMULUS  B2
;;; STATE1-SELECTED-STIMULUS  RIGHT
;;; STATE2-SELECTED-STIMULUS  RIGHT
;;;
;;;   Four productions compete: choose-left(), choose-right(),
;;;   dont-choose-left(), dont-choose-right().
;;;   Agent makes response L or R 
;;; - At stage 2: a pair of Markov Stimli; encode stimuli in WM
;;;   Four productions compete: choose-left(), choose-right(),
;;;   dont-choose-left(), dont-choose-right().
;;;   Agent makes response L or R 
;;; - In the end of each trial, a reward is delivered, encodes reward amount
;;; - stop the experiment "done"
;;; 
;;; Chunk Type descriptions:
;;; - markov-stimulus: Contain the state information (0,1,2,3), left and right
;;;                    stimulus properties
;;;
;;; - wm: Contains stimuli properties from two states, selected response, and 
;;;       reward amount
;;;
;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; 
;;; Productions: 
;;; ===== fixation =====
;;; p prepare-wm ()
;;; p find-screen ()  
;;; p process-fixation()
;;; ===== state1 =====
;;; p encode-state1-stimulus ()  
;;; p plan-state1-retrieve ()  
;;; |--- MOT > 0: p plan-state1-retrieve-success ()  
;;; |--- MOT <= 0: p plan-state1-retrieve-skip()
;;; |------ p plan-state1-random-left()  ;;; compete
;;; |--------- p choose-state1-left()
;;; |------ p plan-state1-random-right() ;;; compete
;;; |--------- p choose-state1-right()
;;; ===== state2 =====
;;; p encode-state2-stimulus ()  
;;; p plan-state2-retrieve ()  
;;; |--- MOT > 0: p plan-state2-retrieve-success ()  
;;; |--- MOT <= 0: p plan-state2-retrieve-skip()
;;; |------ p plan-state2-random-left()  ;;; compete
;;; |--------- p choose-state2-left()
;;; |------ p plan-state2-random-right() ;;; compete
;;; |--------- p choose-state2-right() 
;;; ===== feedback =====
;;; p encode-state3() 
;;; p refresh-memory() 
;;; p refresh-memory-success()
;;; p done()
;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;; --------- CHUNK TYPE ---------
(chunk-type (markov-stimulus (:include visual-object))
      kind 
      stage
      color
      left-stimulus
      right-stimulus)

(chunk-type (markov-stimulus-location (:include visual-location))
      kind
      stage
      color
      left-stimulus
      right-stimulus
      location
      reward)

(chunk-type markov-reward
      kind
      stage
      reward)

(chunk-type wm    
      status
      state1-left-stimulus
      state1-right-stimulus 
      state2-left-stimulus
      state2-right-stimulus 
      state1-selected-stimulus
      state2-selected-stimulus
      reward)

;;; DEFINE PLAN GOAL
(chunk-type phase
      step
      stage
      plan-state1-selected-stimulus
      plan-state2-selected-stimulus
      motivation                    ;;; mental counts
      updated-motivation            ;;; mental counts
      time-onset                    ;;; mental clock
      time-duration                 ;;; mental clock
      previous-reward)              ;;; previous reward given

;;; --------- DM ---------
(add-dm 
  (start-trial isa phase step attend-fixation time-onset 0.0 previous-reward 0.0)
 )


;;; ------------------------------------------------------------------
;;; INITIALIZATION
;;; ------------------------------------------------------------------

(p prepare-wm
   "Init task, prepare WM"
   ?imaginal>
     buffer empty
     state free

   ?manual>
     preparation free
     processor free
     execution free

   ?goal>
     ;buffer   empty
     state    free

   =goal>
     isa      phase
     step     attend-fixation
     time-onset =TIME
     previous-reward =R
==>
   
   +imaginal>
     isa wm
     status process
   
   *goal>
     motivation =R
   
   !eval! (trigger-reward 0) ; CLEAR REWARD  
   
   !output! (in prepare-wm() previous-rewar value is =R time-onset is =TIME)
)

(p find-screen
   "Look at the screen (if you were not already looking at it)"
    =visual-location>
    ?visual>
     state      free
   
    ?goal>
     state      free
==>
    +visual>
      cmd      move-attention
      screen-pos =visual-location 
)


(p process-fixation
   "Attend to the fixation cross STATE0"
    ?visual>
      state    free
    =visual>
      text     T
      value    "+"
    ?imaginal>
      state    free
    =goal>
     isa        phase
     step       attend-fixation 
     motivation =MOT 
==>
   !bind!       =TIME (mp-time)
    =goal>
      step       attend-stimulus
      stage      0 
      updated-motivation  =MOT  ; init MOT (keep track of discounted motivation)
      time-onset =TIME
   
    !output! (in process-fixation time-onset =TIME updated-motivation =MOT)
)

;;; ----------------------------------------------------------------
;;; ATTEND MARKOV STIMULUS
;;; ----------------------------------------------------------------
;;; encode markov stimulus in two states
;;; state0： "+"
;;; state1: A1 vs A2
;;; state2: B1 vs. B2 or C1 vs C2
;;; state3: Rewards +10
;;; ----------------------------------------------------------------


;;; ----------------------------------------------------------------
;;; ATTEND MARKOV STIMULUS: STATE1
;;; ----------------------------------------------------------------

(p encode-state1
   "Encodes the STATE1 stimulus in WM"
   =visual>
     kind MARKOV-STIMULUS
     stage 1
     stage =STAGE
     left-stimulus  =L
     right-stimulus =R
     

   =imaginal>
     status process 
     state1-left-stimulus nil
     state1-right-stimulus nil
     state1-selected-stimulus nil

   ?retrieval>
     state free
     buffer empty

   =goal>
     isa        phase
     step       attend-stimulus
==>
   =goal> 
     step       plan-retrieve ;respond
     stage      =STAGE
   
   =visual>
   
   =imaginal> 
     state1-left-stimulus   =L
     state1-right-stimulus  =R
   
   !output! (in encode-state2 =L =R)
)

(p plan-state1-retrieve
   "Plan STATE1 by retrieving a reward memory"
   ?retrieval>
     state free
     buffer empty
   
   ?goal> 
     state free
   
   ?imaginal>
     state free
   
   =imaginal> 
     state1-left-stimulus   =L1
     state1-right-stimulus  =R1
   
   =goal> 
     step       plan-retrieve
     stage      1
     plan-state1-selected-stimulus nil
     updated-motivation   =MOT
     time-onset           =TIME

==>
   ;;; TODO: need to change based on reward policy
   +retrieval>
     ;:recently-retrieved nil
     status process
     > reward 0
     state1-left-stimulus   =L1
     state1-right-stimulus  =R1
     - state2-left-stimulus nil
     - state2-right-stimulus nil
     - state1-selected-stimulus nil
     - state2-selected-stimulus nil
   
   !bind!       =CURRTIME (mp-time)
   !bind!       =DURATION (- =CURRTIME =TIME)
   !bind! =DIFF (- =MOT =DURATION)
   
   =imaginal>
   
   *goal> 
     step  plan-decide
     updated-motivation   =DIFF

  !output! (in retrieve-intended-response() the motivation val is =MOT duration value is =DURATION new motivation val is =DIFF)
)

(p plan-state1-retrieve-success
   "Plan STATE1: success
   Encode planed response in GOAL
   "
   ?retrieval>
    state free
    buffer full
   
   ?goal>
     state free
   
   ?imaginal>
     state free
   
   =goal> 
     step       plan-decide
     plan-state1-selected-stimulus nil
     > updated-motivation 0 ;;; compete with dont-check, higher utility fires
     updated-motivation =MOT
   
   =retrieval>
     > reward 0
     state1-selected-stimulus  =RESPONSE
==>
   ; ENCODE PLANED OUTCOME IN GOAL 
   =goal> 
     step       respond
     plan-state1-selected-stimulus =RESPONSE
   
   !output! (in plan-state1-retrieve-success() S1 =RESPONSE MOT =MOT)
)

;;; ----------------------------------------------------------------
;;; RETRIEVE: FAILED OR SKIP
;;; ----------------------------------------------------------------

(p plan-state1-retrieve-skip
   "Plan STATE1: success, but MOTIVATION < 0
   Encode planed response in GOAL - LEFT
   "
   ?retrieval>
      state busy
   
   ?goal>
       state free
    
   =goal> 
     step       plan-decide
     plan-state1-selected-stimulus nil
     <= updated-motivation 0 ;;; compete with dont-check, higher utility fires
     updated-motivation =MOT

==>
   ; ENCODE PLANED OUTCOME IN GOAL 
   =goal> 
     step       respond-random
   
   -retrieval>
   
   !output! (in plan-state1-retrieve-skip() MOT =MOT)
)


(p plan-state1-retrieve-failure
   "Plan STATE1: failure - randomly select one action
   "
   ?retrieval>
      buffer   failure 
   
   ?goal>
      state free
    
   =goal> 
     step       plan-decide
     plan-state1-selected-stimulus nil
==>
   ; ENCODE PLANED OUTCOME IN GOAL 
   =goal> 
     step       respond
     plan-state1-selected-stimulus left
   
   =goal> 
     step       respond-random
   
   -retrieval>
   
   !output! (in plan-state1-retrieve-failure() )
)


(p plan-state1-random-left
    "Plan STATE1: failure - randomly select one action"
   ?retrieval>
      state free
   
   =goal> 
     step       respond-random
     plan-state1-selected-stimulus nil 
==>
   ; ENCODE PLANED OUTCOME IN GOAL 
   =goal> 
     step       respond
     plan-state1-selected-stimulus left
   
   -retrieval>
   
   !output! (in plan-state1-random-left())
)

(p plan-state1-random-right
    "Plan STATE1: failure - randomly select one action"
   ?retrieval>
      state free
   
   =goal> 
     step       respond-random
     plan-state1-selected-stimulus nil 
==>
   ; ENCODE PLANED OUTCOME IN GOAL 
   =goal> 
     step       respond
     plan-state1-selected-stimulus right
   
   -retrieval>
   
   !output! (in plan-state1-random-right())
)


;;; ----------------------------------------------------------------
;;; ATTEND MARKOV STIMULUS: STATE2
;;; ----------------------------------------------------------------


(p encode-state2
   "Encodes the STATE2 stimulus in WM"
   =visual>
     kind MARKOV-STIMULUS
     stage 2
     stage =STAGE
     left-stimulus  =L
     right-stimulus =R
     
   =imaginal>
     status process 
     - state1-selected-stimulus nil
     state2-left-stimulus nil
     state2-right-stimulus nil
     state2-selected-stimulus nil

   ?retrieval>
     state free
     buffer empty

   =goal>
     isa        phase
     step       attend-stimulus 
==>
   =goal> 
     step       plan-retrieve ;respond
     stage      =STAGE
   
   =visual>
   
   =imaginal>
     state2-left-stimulus   =L
     state2-right-stimulus  =R
   
   !output! (in encode-state2 =L =R)
   ;!eval! (trigger-reward 0) ; CLEAR REWARD 
)


(p plan-state2-retrieve
   "Plan STATE2 by retrieving a reward memory
   
   "
   ?retrieval>
     state free
     buffer empty
   
   ?goal> 
     state free
   
   ?imaginal>
     state free
   
   =imaginal>
    state1-left-stimulus  =S11
    state1-right-stimulus =S12
    state2-left-stimulus  =S21
    state2-right-stimulus =S22
    state1-selected-stimulus =R1
    state2-selected-stimulus nil
    
   
   =goal> 
     step       plan-retrieve
     stage      2
     - plan-state1-selected-stimulus nil
     plan-state2-selected-stimulus nil
     updated-motivation   =MOT
     time-onset           =TIME

==>
   ;;; need to change based on reward policy
   +retrieval>
     ;:recently-retrieved nil
     status process
     > reward 0 
     state1-left-stimulus  =S11
     state1-right-stimulus =S12
     state2-left-stimulus  =S21
     state2-right-stimulus =S22
     state1-selected-stimulus =R1 
   
   !bind!       =CURRTIME (mp-time)
   !bind!       =DURATION (- =CURRTIME =TIME)
   !bind! =DIFF (- =MOT =DURATION)
   
   =imaginal>
   
   *goal> 
     step  plan-decide
     updated-motivation   =DIFF   
   
  !output! (in plan-state2-retrieve =S11 =S12 =S21 =S22 =R1 the motivation val is =MOT duration value is =DURATION new motivation val is =DIFF)
     
)

(p plan-state2-retrieve-success
   "Plan STATE1: success
   Encode planed response in GOAL
   "
   ?retrieval>
     state free
     buffer full
   
   ?goal>
     state free
   
   ?imaginal>
     state free
   
   =goal> 
     step       plan-decide
     - plan-state1-selected-stimulus nil
     plan-state2-selected-stimulus nil
     > updated-motivation 0  ;;; compete with dont-check, higher utility fires
     updated-motivation =MOT
   
   =retrieval>
     > reward 0
     state1-selected-stimulus =RESPONSE1
     state2-selected-stimulus =RESPONSE2
==>
   
   ; ENCODE PLANED OUTCOME IN GOAL 
   =goal> 
     step       respond 
     plan-state2-selected-stimulus =RESPONSE2
   
   ;!output! (in plan-state2-retrieve-success() S1 =RESPONSE1 S2 =RESPONSE2 MOT =MOT) 
)


;;; ----------------------------------------------------------------
;;; RETRIEVE: FAILED OR SKIP
;;; ----------------------------------------------------------------

(p plan-state2-retrieve-skip
   "Plan STATE2: success, but MOTIVATION < 0
   Encode planed response in GOAL - LEFT
   "
   ?retrieval>
      state busy
   
   ?goal>
       state free

   =goal> 
     step       plan-decide
     - plan-state1-selected-stimulus nil
     plan-state2-selected-stimulus nil
     <= updated-motivation 0 ;;; compete with dont-check, higher utility fires
     updated-motivation =MOT

==>
   ; ENCODE PLANED OUTCOME IN GOAL 
   =goal> 
     step       respond-random
   
   -retrieval>
   
   !output! (in plan-state2-retrieve-skip MOT =MOT)
)


(p plan-state2-retrieve-failure
   "Plan STATE2: failure - randomly select one action
   "
   ?retrieval>
      buffer   failure 
   
   ?goal>
      state free
    
   =goal> 
     step       plan-decide
     - plan-state1-selected-stimulus nil
     plan-state2-selected-stimulus nil
==>
   ; ENCODE PLANED OUTCOME IN GOAL  
   =goal> 
     step       respond-random
   
   -retrieval>
   
   !output! (in plan-state2-retrieve-failure )
)


(p plan-state2-random-left
    "Plan STATE2: failure - randomly select one action"
   ?retrieval>
      state free
   
   =goal> 
     step       respond-random
     - plan-state1-selected-stimulus nil   
     plan-state2-selected-stimulus nil 
==>
   ; ENCODE PLANED OUTCOME IN GOAL 
   =goal> 
     step       respond
     plan-state2-selected-stimulus left
   
   -retrieval>
   
   !output! (in plan-state2-random-left )
)

(p plan-state2-random-right
    "Plan STATE2: failure - randomly select one action"
   ?retrieval>
      state free
   
   =goal> 
     step       respond-random
     - plan-state1-selected-stimulus nil   
     plan-state2-selected-stimulus nil 
==>
   ; ENCODE PLANED OUTCOME IN GOAL 
   =goal> 
     step       respond
     plan-state2-selected-stimulus right
   
   -retrieval>
   
   !output! (in plan-state2-random-right )
)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;; REMOVE
;; (p plan-state2-retrieve-failure-left
;;    "Plan STATE2: failure
;;    Encode default response :LEFT
;;    "
;;    ?retrieval>
;;      buffer failure
   
;;    ?goal>
;;      state free
   
;;    ?imaginal>
;;      state free
    
;;    =goal> 
;;      step       plan-decide
;;      - plan-state1-selected-stimulus nil
;;      plan-state2-selected-stimulus nil
;; ==>
;;    ; ENCODE PLANED OUTCOME IN GOAL 
;;    =goal> 
;;      step       respond
;;      plan-state2-selected-stimulus left
   
;;    !output! (in plan-state2-random-left() )
;; )

;; (p plan-state2-retrieve-failure-right
;;    "Plan STATE2: failure
;;    Encode default response :RIGHT
;;    "
;;    ?retrieval>
;;      buffer failure
   
;;    ?goal>
;;      state free
   
;;    ?imaginal>
;;      state free
    
;;    =goal> 
;;      step       plan-decide
;;      - plan-state1-selected-stimulus nil
;;      plan-state2-selected-stimulus nil
;; ==>
;;    ; ENCODE PLANED OUTCOME IN GOAL 
;;    =goal> 
;;      step       respond
;;      plan-state2-selected-stimulus right
   
;;    !output! (in plan-state2-random-right() )
;; )

;;; ----------------------------------------------------------------
;;; RESPONSE SELECTION
;;; ----------------------------------------------------------------
;;; State1: choose-state1-left() and choose-state1-right() compete
;;; State2: choose-state2-left() and choose-state2-right() compete
;;; ----------------------------------------------------------------



;;; ----------------------------------------------------------------
;;; RESPONSE SELECTION: STATE1
;;; ----------------------------------------------------------------


(p choose-state1-left
   "At STATE1: Choose LEFT stimulus"
   ?manual>
     preparation free
     processor free
     execution free
   
   =visual>
     kind markov-stimulus
   
   
   ; USE PLAN TO GUIDE CHOOSE-LEFT     
   =goal>
     isa        phase
     plan-state1-selected-stimulus left 
     step       respond

   =imaginal>
     status process
     - state1-left-stimulus nil
     - state1-right-stimulus nil
     state1-selected-stimulus nil
    
   ?retrieval>
     state free
     buffer empty

==>
   +manual>
     isa punch
     hand left
     finger index
   =visual>
   =goal>
     step       attend-stimulus
   =imaginal>
     state1-selected-stimulus left
   
   !output! (in state1-left())
)


(p choose-state1-right
   "At STATE1: Choose RIGHT stimulus"
   ?manual>
     preparation free
     processor free
     execution free
   
   =visual>
     kind markov-stimulus
        
   ; USE PLAN TO GUIDE CHOOSE-RIGHT     
   =goal>
     isa        phase
     plan-state1-selected-stimulus right 
     step       respond

   =imaginal>
     status process
     - state1-left-stimulus nil
     - state1-right-stimulus nil
     state1-selected-stimulus nil
    
   ?retrieval>
     state free
     buffer empty

==>
   +manual>
     isa punch
     hand right
     finger middle
   =visual>
   =goal>
     step       attend-stimulus
   =imaginal>
     state1-selected-stimulus right
   
   !output! (in state1-right())
)



;;; ----------------------------------------------------------------
;;; RESPONSE SELECTION: STATE2
;;; ----------------------------------------------------------------


(p choose-state2-left
   "At STATE2: Choose LEFT stimulus"
   ?manual>
     preparation free
     processor free
     execution free
   
   =visual>
     kind markov-stimulus
        
   ; USE PLAN TO GUIDE CHOOSE-RIGHT     
   =goal>
     isa        phase
     plan-state2-selected-stimulus left 
     step       respond

   =imaginal>
     status process
     - state2-left-stimulus nil
     - state2-right-stimulus nil
     state2-selected-stimulus nil
    
   ?retrieval>
     state free
     buffer empty

==>
   +manual>
     isa punch
     hand left
     finger index
   =visual>
   =goal>
     step       attend-stimulus
   
   =imaginal>
     state2-selected-stimulus left
   !output! (in state2-left())
  
)

(p choose-state2-right
   "At STATE2: Choose RIGHT stimulus"
   ?manual>
     preparation free
     processor free
     execution free
   
   =visual>
     kind markov-stimulus
        
   ; USE PLAN TO GUIDE CHOOSE-RIGHT     
   =goal>
     isa        phase
     plan-state2-selected-stimulus right 
     step       respond

   =imaginal>
     status process
     - state2-left-stimulus nil
     - state2-right-stimulus nil
     state2-selected-stimulus nil
    
   ?retrieval>
     state free
     buffer empty

==>
   +manual>
     isa punch
     hand right
     finger middle
   =visual>
   =goal>
     step       attend-stimulus
   =imaginal>
     state2-selected-stimulus right
   !output! (in state2-left())
)




;;; ----------------------------------------------------------------
;;; ATTEND MARKOV STIMULUS: STATE3
;;; ----------------------------------------------------------------

(p encode-state3
   "Encodes the STATE3 (REWARD) in WM" 
    =visual>
     kind MARKOV-REWARD
     stage 3
     stage =STAGE
     reward =REWARD
   
    =goal>
     isa        phase
     step       attend-stimulus

   =imaginal>
     status process 
     - state1-selected-stimulus nil
     - state2-selected-stimulus nil
     state1-selected-stimulus =S1
     state2-selected-stimulus =S2
     reward nil

==>
   =goal>
     isa      phase
     step     retrieve
     stage    =STAGE
     previous-reward =REWARD ;;; THIS allows ACT-R set MOTIVATION based on previous trial's reward
   
   =imaginal>
     reward    =REWARD 
   
   !eval! (trigger-reward =REWARD)
   !output! (deliver reward =REWARD WM =S1 =S2)
)

(p refresh-memory 
   "Refresh memory of the trial"
   ?retrieval>
    state free 
   
   ?imaginal>
     state free
     buffer full
   
   
   =goal>
     isa      phase
     step     retrieve
   
   =imaginal>
      isa wm
      status process
      state1-left-stimulus =S11
      state1-right-stimulus =S12
      state2-left-stimulus =S21
      state2-right-stimulus =S22
      state1-selected-stimulus =R1
      state2-selected-stimulus =R2
      reward =R

==>
   =goal>
   
   -imaginal> 
   
   -visual>
   
   +retrieval> 
     status process
     state1-left-stimulus =S11
     state1-right-stimulus =S12
     state2-left-stimulus =S21
     state2-right-stimulus =S22
     state1-selected-stimulus =R1
     state2-selected-stimulus =R2
     reward =R
    
   !output! (refresh-memory  =R1 =R2 =R)

)

(p refresh-memory-success 
    "If successfully refreshed memory, clear retrieval buffer and reset goal buffer"
   ?retrieval>
    state free 
    buffer full
   
   ?imaginal>
     state free
     buffer empty
   
   ?visual> 
     state free 
     buffer empty
   
   ?goal>
    state free
    buffer full
   
   =goal>
     isa      phase
     step     retrieve 

==>
   ; RESET GOAL
   =goal> 
     step       attend-fixation 
     plan-state1-selected-stimulus nil
     plan-state2-selected-stimulus nil
   
   -retrieval>
   
   !output! (refresh-memory-success)
)

(p refresh-memory-failure 
    "If refreshed memory failed, clear retrieval buffer and reset goal buffer"
   ?retrieval>
     buffer failure
     state free
   
   =goal>
     isa      phase
     step     retrieve 

==>
   ; RESET GOAL
   =goal> 
     step       attend-fixation 
     plan-state1-selected-stimulus nil
     plan-state2-selected-stimulus nil
   
   -retrieval>
   
   !output! (refresh-memory-fail)
)


;;; ----------------------------------------------------------------
;;; DONE
;;; ----------------------------------------------------------------

(p done
   "Detects when the experiment is done"
   =visual>
     text           t
     value          "done"
     color          black

   ?visual>
     state          free

   ?manual>
     preparation    free
     processor      free
     execution      free

   ?goal>
     state          free
   
   =goal>
     isa        phase
     step       attend-fixation
==>
   !stop!

)

(goal-focus start-trial)