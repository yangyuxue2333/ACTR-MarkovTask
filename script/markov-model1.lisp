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
;;; Version     :v1.8
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
;;; ===== FIXATION ===== 
;;; P PREPARE-WM () 
;;; P FIND-SCREEN () 
;;; P PROCESS-FIXATION() ;
;; ===== STATE1 =====
;;; P ATTEND-STATE1() 
;;; |--- P CHOOSE-STATE1-LEFT() 
;;; |--- P CHOOSE-STATE1-RIGHT() 
;;; P ENCODE-STATE1 ()  
;;; ===== STATE2 ===== 
;;; P ATTEND-STATE2 () 
;;; |--- P CHOOSE-STATE2-LEFT() 
;;; |--- P CHOOSE-STATE2-RIGHT() 
;;; ===== REWAR ===== 
;;; P ENCODE-STATE2 () 
;;; P REFRESH-MEMORY ()
;;; P DONE() ;;;;

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
      plan-state1
      plan-state1-response
      plan-state2
      plan-state2-response
      motivation                    ;;; mental counts
      updated-motivation            ;;; mental counts
      time-onset                    ;;; mental clock
      time-duration                 ;;; mental clock
      current-reward                ;;; reward received in current trial
      previous-reward               ;;; reward received in previous trial
      ;;; blending
      state-b-blended-value         ;;; 
      state-c-blended-value         ;;;
      diff-blended-value            ;;; state-b-blended-value - state-c-blended-value
      best-blended-state            ;;; "B" or "C"
)



;;; --------- DM ---------

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

   +imaginal>
     isa wm
     status process
     curr-state =STATE
     left-stimulus  =L
     right-stimulus =R
     response nil
     next-state nil
     reward nil
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
     step       attend-stimulus  
     stage      =STAGE
   
   =visual>
   
   @imaginal>
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
     current-reward =REWARD
   
   =imaginal>
     reward    =REWARD 
   
   -visual>
   
   !eval! (trigger-reward =REWARD)
)

(p refresh-memory
  "refresh memory: harvesting imaginal buffer"
   ?imaginal>
     state free
     buffer full
   
   ?retrieval>
     state free
     buffer empty
   
   =goal>
     step  refresh-memory
     plan-state1-response  =RESP1
   
   =imaginal>
       status  PROCESS
       reward  =R
       curr-state  =CURR
       response  =RESP
   
==>
   !output! (encode state2 curr-state  =CURR response  =RESP reward  =R)
   =goal>
    step  refresh-success  ; one-time refresh

   -imaginal>

   ;;; encode state1 memory
   +imaginal>
       isa wm
       status  PROCESS
       left-stimulus  A1
       right-stimulus  A2
       reward  =R
       curr-state  A
       next-state  =CURR
       response  =RESP1
    !output! (encode state1 curr-state A next-state  =CURR response  =RESP1 reward =R)
)


(p refresh-success
 "success refresh"
  ?imaginal>
     state free
     buffer full
   
   ?retrieval>
     state free

   =goal>
     step  refresh-success
==>
   =goal>
    step attend-stimulus

   -imaginal>
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
     plan-state1-response left
   
   =imaginal>
      response left
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
     plan-state1-response right
   
   =imaginal>
      response right
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
     plan-state2-response left
   
   =imaginal>
      response left
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
     plan-state2-response right
   
   =imaginal>
      response right
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



; ######### SETUP MODEL markov-model1 #########
;   >> TASK PARAMETERS: {'MARKOV_PROBABILITY': 0.7, 'REWARD_PROBABILITY': {}, 'REWARD': {'B1': (1, 0), 'B2': (1, 0), 'C1': (1, 0), 'C2': (1, 0)}, 'RANDOM_WALK': 'LOAD', 'M': 1} <<
;   >> ACT-R PARAMETERS: {'v': 't', 'seed': '[100, 0]', 'ans': 0.2, 'lf': 0.1, 'bll': 0.5, 'egs': 0.2, 'alpha': 0.2, 'bln': 't', 'act': 't', 'blt': 't', 'dmt': 'nil', 'rt': -10} <<

;      0.050   PROCEDURAL             PRODUCTION-FIRED PREPARE-WM
;      1.050   PROCEDURAL             PRODUCTION-FIRED FIND-SCREEN
;      1.185   PROCEDURAL             PRODUCTION-FIRED PROCESS-FIXATION
;      2.135   PROCEDURAL             PRODUCTION-FIRED FIND-SCREEN
;      2.270   PROCEDURAL             PRODUCTION-FIRED ATTEND-STATE1
;      2.520   PROCEDURAL             PRODUCTION-FIRED CHOOSE-STATE1-RIGHT
;      2.780   PROCEDURAL             PRODUCTION-FIRED FIND-SCREEN
;      2.915   PROCEDURAL             PRODUCTION-FIRED ENCODE-STATE1
;      2.965   PROCEDURAL             PRODUCTION-FIRED ATTEND-STATE2
;      3.215   PROCEDURAL             PRODUCTION-FIRED CHOOSE-STATE2-RIGHT
;      3.325   PROCEDURAL             PRODUCTION-FIRED FIND-SCREEN
;      3.460   PROCEDURAL             PRODUCTION-FIRED ENCODE-STATE2
; <[MARKOV_STATE]   [R, 0.65]'A2'   [R, 0.50]'C2'   R:[0]   [C][C]
;      3.510   PROCEDURAL             PRODUCTION-FIRED REFRESH-MEMORY
; ENCODE STATE2 CURR-STATE C RESPONSE RIGHT REWARD 0
; ENCODE STATE1 CURR-STATE A NEXT-STATE C RESPONSE RIGHT REWARD 0
;      3.760   PROCEDURAL             PRODUCTION-FIRED REFRESH-SUCCESS
;      3.810   PROCEDURAL             PRODUCTION-FIRED PREPARE-WM
;     80.050   PROCEDURAL             PRODUCTION-FIRED FIND-SCREEN
;     80.185   PROCEDURAL             PRODUCTION-FIRED DONE