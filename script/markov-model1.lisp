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
;;; Filename    :markov-model1.lisp
;;; Version     :v1.2
;;;
;;; Description : model-free 
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
;;;   Four productions compete: choose-left(), choose-right(),
;;;   dont-choose-left(), dont-choose-right().
;;;   Agent makes response L or R 
;;; - At stage 2: a pair of Markov Stimli; encode stimuli in WM
;;;   Four productions compete: choose-left(), choose-right(),
;;;   dont-choose-left(), dont-choose-right().
;;;   Agent makes response L or R 
;;; - In the end of each trial, a reward is delivered, encodes reward amount
;;; - stop the experiment "done"
;;; - NOTE: to make model only look back one step, !eval! (trigger-reward 0) 
;;;         is added when state2 is encoded (!!deprecated!!)
;;; 
;;; Chunk Type descriptions:
;;; - markov-stimulus: Contain the state information (STATE, STAGE, LEFT-STIMULUS, RIGHT-STIMULUS)
;; MARKOV-STIMULUS0-0
;;    STATE  A
;;    KIND  MARKOV-STIMULUS
;;    SCREEN-POS  MARKOV-STIMULUS-LOCATION0-0
;;    COLOR  GREEN
;;    STAGE  1
;;    LEFT-STIMULUS  A1
;;    RIGHT-STIMULUS  A2
;;;
;;; - WM: contains CURR-STATE, NEXT-STATE, RESPONSE, and REWARD
;; M-A1
;;    STATUS  PROCESS
;;    LEFT-STIMULUS  A1
;;    RIGHT-STIMULUS  A2
;;    REWARD  NONE
;;    CURR-STATE  A
;;    NEXT-STATE  B
;;    RESPONSE  LEFT
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
;;; p attend-state1()
;;; |--- p choose-state1-left()
;;; |--- p choose-state1-right()
;;; p encode-state1 ()   
;;; ===== state2 =====
;;; p attend-state2 ()  
;;; |--- p choose-state2-left()
;;; |--- p choose-state2-right()
;;; ===== feedback ===== 
;;; p encode-state2 ()  
;;; p done()
;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;


;;; --------- CHUNK TYPE ---------
(chunk-type (markov-stimulus (:include visual-object))
      kind 
      stage
      state
      color
      left-stimulus
      right-stimulus)

(chunk-type (markov-stimulus-location (:include visual-location))
      kind
      stage
      state
      color
      left-stimulus
      right-stimulus
      location
      reward)

(chunk-type markov-reward
      kind
      stage
      reward)

;; (chunk-type wm    
;;       status
;;       state1-left-stimulus
;;       state1-right-stimulus 
;;       state2-left-stimulus
;;       state2-right-stimulus 
;;       state1-selected-stimulus
;;       state2-selected-stimulus
;;       reward)

(chunk-type wm    
      status
      curr-state
      left-stimulus
      right-stimulus
      next-state
      response
      reward)

(chunk-type phase
      step
      stage
      motivation
      plan-state1-selected-stimulus
      plan-state2-selected-stimulus)



;;; --------- DM ---------
(add-dm 
  (start-trial isa phase step attend-stimulus)
)


;;; ------------------------------------------------------------------
;;; INITIALIZATION
;;; ------------------------------------------------------------------

(p prepare-wm
   "Init task, prepare WM"
   ?visual>
    buffer empty
    state free
   
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
     step     attend-stimulus 
     ;motivation 10
==>
   +imaginal>
     isa wm
     status process
   !eval! (trigger-reward 0) ; CLEAR REWARD  
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
     step       attend-stimulus  
==>
    =goal>
      step       attend-stimulus
      stage      0
      
    ;!output! (in process-fixation())
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

(p attend-state1
   "Encodes the STATE1 stimulus in WM"
   =visual>
     kind MARKOV-STIMULUS
     stage 1
     stage =STAGE
     state =STATE
     left-stimulus  =L
     right-stimulus =R
     

   =imaginal>
     status process 
     curr-state nil
     left-stimulus nil
     right-stimulus nil

   ?retrieval>
     state free
     buffer empty

   =goal>
     isa        phase
     step       attend-stimulus
==>
   =goal> 
     step       respond
     stage      =STAGE
   
   =visual>
   
   =imaginal> 
     curr-state =STATE
     left-stimulus  =L
     right-stimulus =R
   
   !output! (in attend-state1 =STATE =L =R)
    
)

(p encode-state1
   "Encodes the STATE2 stimulus in WM"
   =visual>
     kind MARKOV-STIMULUS
     stage 2
     stage =STAGE
     state =STATE
     left-stimulus  =L
     right-stimulus =R
     
   =imaginal>
     status process 
     - curr-state nil
     - response nil
     next-state nil
     reward nil

   ?retrieval>
     state free
     buffer empty
   
   ?manual> 
     state free

   =goal>
     isa        phase
     step       encode-stimulus 
==>
   =goal> 
     step       refresh-memory 
     stage      =STAGE
   
   =visual>
   
   =imaginal>
     next-state =STATE
     reward none
   
   ;-imaginal>
   
   !output! (in encode-state1 =L =R)
)


(p attend-state2
  =visual>
     kind MARKOV-STIMULUS
     stage 2
     stage =STAGE
     state =STATE
     left-stimulus  =L
     right-stimulus =R
   
   =goal> 
     step       attend-stimulus
     stage      =STAGE
   
   ?imaginal>
     state free
     buffer empty
   
==> 
   +imaginal>
     isa wm
     status process
     curr-state =STATE
     left-stimulus  =L
     right-stimulus =R
     response nil
     next-state nil
     reward nil
   
   =goal>
     step       respond
   
   =visual>
   !output! (in encode-state2 =L =R)
)

(p encode-state2
   "Encodes the STATE3 (REWARD) in WM" 
    =visual>
     kind MARKOV-REWARD
     stage 3
     stage =STAGE
     reward =REWARD
   
    =goal>
     isa        phase
     step       encode-stimulus

   =imaginal>
     - curr-state nil
     - left-stimulus nil
     - right-stimulus nil
     - response nil
     next-state nil
     reward nil
     

==>
   =goal> 
     step       refresh-memory  
     stage      =STAGE
   
   =imaginal>
     reward    =REWARD
     next-state none
   
   -visual>
   
   ;-imaginal>
   
   !eval! (trigger-reward =REWARD)
   
   !output! (in  encode-state2 reward =REWARD)
)


(p refresh-memory
  "refresh memorty "
   ?imaginal>
     state free
     buffer full
   
   ?retrieval>
     state free
     buffer empty
   
   =goal>
     step  refresh-memory
   
   =imaginal>
       status  PROCESS
       left-stimulus  =LEFT
       right-stimulus  =RIGHT
       reward  =R
       curr-state  =CURR
       next-state  =NEXT
       response  =RESP
   
==>
   
   =goal>
   
   =imaginal>
   
   +retrieval>
       isa wm
       status  PROCESS
       left-stimulus  =LEFT
       right-stimulus  =RIGHT
       reward  =R
       curr-state  =CURR
       next-state  =NEXT
       response  =RESP
)

(p refresh-success
 "success refresh"
  ?imaginal>
     state free
     buffer full
   
   ?retrieval>
     state free
     buffer full
   
   =goal>
     step  refresh-memory
   
==> 
   
   =goal>
    step attend-stimulus 
   
   -imaginal>
   
   -retrieval>
 )

(p refresh-failure
 "failure refresh"
  ?imaginal>
     state free 
   
   ?retrieval>
     buffer failure
   
   =goal>
     step  refresh-memory
   
==> 
   
   =goal>
    step attend-stimulus 
   
   -imaginal>
   
   -retrieval>
 )

;;; ----------------------------------------------------------------
;;; RESPONSE SELECTION
;;; ----------------------------------------------------------------
;;; State1: choose-state1-left() and choose-state1-right() compete
;;; State2: choose-state2-left() and choose-state2-right() compete
;;; ----------------------------------------------------------------

(p choose-state1-left
   "At STATE1: Choose LEFT stimulus"
   ?manual>
     preparation free
     processor free
     execution free
   
   =visual>
     kind markov-stimulus
     stage 1
        
   =goal>
     isa        phase
     step       respond

   =imaginal>
     status process 
     - curr-state nil
     next-state nil
     response nil
    
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
     step       encode-stimulus 
   
   =imaginal>
      response left
   
   !output! (in state1-left)
  
)

(p choose-state1-right
   "At STATE1: Choose RIGHT stimulus"
   ?manual>
     preparation free
     processor free
     execution free
   
   =visual>
     kind markov-stimulus
     stage 1
        
   =goal>
     isa        phase
     step       respond

   =imaginal>
     status process 
     - curr-state nil
     next-state nil
     response nil
    
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
     step       encode-stimulus 
   
   =imaginal>
      response right
   
   !output! (in state1-right)
  
)


(p choose-state2-left
   "At STATE2: Choose LEFT stimulus"
   ?manual>
     preparation free
     processor free
     execution free
   
   =visual>
     kind markov-stimulus
     stage 2
        
   =goal>
     isa        phase
     step       respond

   =imaginal>
     status process 
     - curr-state nil
     - left-stimulus nil
     - right-stimulus nil
     response nil
     next-state nil
     reward nil
    
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
     step       encode-stimulus 
   
   =imaginal>
      response left
   
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
     stage 2
        
   =goal>
     isa        phase
     step       respond

   =imaginal>
     status process 
     - curr-state nil
     - left-stimulus nil
     - right-stimulus nil
     response nil
     next-state nil
     reward nil
    
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
     step       encode-stimulus 
   
   =imaginal>
      response right
   
   !output! (in state2-left)
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
     step       attend-stimulus 
==>
   !stop!

)

(goal-focus start-trial)
