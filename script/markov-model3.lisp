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
;;; Version     :v3.3
;;;
;;; Description : model-motivation 
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
;;; - At stage 1: display a pair of Markov Stimli; attend stimuli pair, 
;;;   - Planing: backward planing method
;;;     - PLAN-BACKWARD-AT-STAGE1-STATE2
;;;     - PLAN-BACKWARD-AT-STAGE1-STATE2
;;;     - PLAN-BACKWARD-AT-STAGE1-COMPLETE
;;;   choose action based on planning, encode stimulus, finally refresh memory
;;; 
;;; - At stage 2: a pair of Markov Stimli; attend stimuli pair
;;;   Four productions compete: choose-left(), choose-right(),
;;;   dont-choose-left(), dont-choose-right().
;;;   Agent makes response L or R 
;;;   - Planing: backward planing method
;;;     - PLAN-BACKWARD-AT-STAGE2-STATE2 
;;;     - PLAN-BACKWARD-AT-STAGE2-COMPLETE
;;;   choose action based on planning, encode stimulus, finally refresh memory
;;;
;;; - In the end of each trial, a reward is delivered, encodes reward amount
;;; - stop the experiment "done"
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
;;; P PROCESS-FIXATION()
;;; ===== STATE1 =====
;;; P ATTEND-STATE1()
;;; P PLAN-STATE1 ()  
;;; |--- MOT > 0: P PLAN-STATE1-START ()  
;;; |--------- P PLAN-BACKWARD-AT-STAGE1-STATE2()
;;; |--------- P PLAN-BACKWARD-AT-STAGE1-STATE1()
;;; |--------- P PLAN-BACKWARD-AT-STAGE1-COMPLETE()
;;; |--- MOT <= 0: P PLAN-STATE1-RETRIEVE-SKIP()
;;; |------ P PLAN-STATE1-RANDOM-LEFT()  ;;; COMPETE
;;; |--------- P CHOOSE-STATE1-LEFT()
;;; |------ P PLAN-STATE1-RANDOM-RIGHT() ;;; COMPETE
;;; |--------- P CHOOSE-STATE1-RIGHT()
;;; P REFRESH-MEMORY() 
;;; P REFRESH-MEMORY-SUCCESS()
;;; ===== STATE2 =====
;;; P ENCODE-STATE2-STIMULUS ()  
;;; P PLAN-STATE2-RETRIEVE ()  
;;; |--- MOT > 0: P PLAN-STATE2-START ()  
;;; |--------- P PLAN-BACKWARD-AT-STAGE2()
;;; |--------- P PLAN-BACKWARD-AT-STAGE2-COMPLETE()
;;; |--- MOT <= 0: P PLAN-STATE2-RETRIEVE-SKIP()
;;; |------ P PLAN-STATE2-RANDOM-LEFT()  ;;; COMPETE
;;; |--------- P CHOOSE-STATE2-LEFT()
;;; |------ P PLAN-STATE2-RANDOM-RIGHT() ;;; COMPETE
;;; |--------- P CHOOSE-STATE2-RIGHT() 
;;; ===== FEEDBACK =====
;;; P ENCODE-STATE3() 
;;; P REFRESH-MEMORY() 
;;; P REFRESH-MEMORY-SUCCESS()
;;; P DONE() 
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
      current-reward                ;;; current reward given
      previous-reward)              ;;; previous reward given



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
     state    free

   =goal>
     isa      phase
     step     attend-stimulus 
     motivation =MOT
     time-onset =TIME
     previous-reward =R
==>
   +imaginal>
     isa wm
     status process
   
   =goal>
     ; ucomment this line allow motivation updated based on previous reward value
     ; otherwise motivation is fixed for each trial
     ;motivation =R
      
   !eval! (trigger-reward 0) ; CLEAR REWARD  
   
   !output! (prepare-wm MOTIVATION =MOT TIME-ONSET =TIME PRE-REWARD =R)
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
     motivation =MOT 
   
==>
    
   !bind!       =CURRTIME (mp-time)
   
    =goal> 
      stage      0
      ; INIT MOT (keep track of discounted motivation)
      updated-motivation  =MOT  
      time-onset =CURRTIME
      
      
    !output! (in process-fixation time-onset =CURRTIME updated-motivation =MOT)
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
     step       plan-start ; plan 
     stage      =STAGE
   
   =visual>
   
   =imaginal> 
     curr-state =STATE
     left-stimulus  =L
     right-stimulus =R
   
   !output! (in attend-state1 =STATE =L =R)
    
)

;;; ----------------------------------------------------------------
;;; PLAN (BACKWARD)
;;; ----------------------------------------------------------------
;;; Plan state2, 
;; +retrieval>
;;      isa wm
;;      outcome 2
;;      next-state nil ;; to ensure this is Stage 2
;;; Plan state1,
;;; ...
;;    =retrieval>
;;       isa wm
;;       outcome 2
;;       next-state nil  ;; to ensure this is stage 2!
;;       state =TARGET
;; ==>
;;    ...
;;    +retrieval>
;;       isa wm
;;       state A
;;       next-state =TARGET
;;; ----------------------------------------------------------------

;;; ----------------------------------------------------------------
;;; PLAN: START
;;; ----------------------------------------------------------------
(p plan-start
   "Update Motivtaion values"
   ?imaginal>
       state free
   
   ?goal>
       state free
    
   =goal> 
     isa phase
     step plan-start 
     plan-state1 nil
     plan-state2 nil 
     ;; motivation
     motivation   =MOT
     time-onset   =TIME
   
==> 
   !bind!       =CURRTIME (mp-time)
   !bind!       =DURATION (- =CURRTIME =TIME)
   !bind! =DIFF (- =MOT =DURATION)
   
   =goal>
       step plan
       updated-motivation   =DIFF
   !output! (plan-at-stage1-start MOTIVATION =MOT CURR TIME =CURRTIME UPDATED-MOTIVATION =DIFF)
)


;;; ----------------------------------------------------------------
;;; RETRIEVE: FAILED OR SKIP
;;; ----------------------------------------------------------------

(p plan-skip
   "if M<0, skip"
   ?retrieval>
      state free
   
   ?goal>
       state free
    
   =goal> 
     step       plan 
     <= updated-motivation 0  
     updated-motivation =MOT

==>
   ; ENCODE PLANED OUTCOME IN GOAL 
   =goal> 
     step       respond-random  
     plan-state1 nil
     plan-state2 nil 
   
   -retrieval>
   
   !output! (in plan-state1-retrieve-skip  updated-motivation =MOT)
)


(p choose-random-action-left
   "if skip planning, randomly choose actions"
   ?retrieval>
      state free
      buffer empty
   
   ?goal>
       state free
   
   ?imaginal>
     state free
     buffer full
    
   =goal> 
     step       respond-random
     plan-state1 nil
     plan-state2 nil 
   
   =imaginal>
     isa wm
     response nil
   
==> 
   
   =goal> 
       step respond
   
   =imaginal>
       response left
)

(p choose-random-action-right
   "if skip planning, randomly choose actions"
   ?retrieval>
      state free
      buffer empty
   
   ?goal>
       state free
   
   ?imaginal>
     state free
     buffer full
    
   =goal> 
     step       respond-random
     plan-state1 nil
     plan-state2 nil 
   
   =imaginal>
     isa wm
     response nil
   
==> 
   
   =goal> 
       step respond
   
   =imaginal>
       response right
)


;;; ----------------------------------------------------------------
;;; PLAN STATE1
;;; ----------------------------------------------------------------
(p plan-backward-at-stage1-state2
   "Plan backward at stage1: state2"
   ?retrieval>
        state free
        buffer empty
   
   ?goal>
       state free
   
   ?imaginal>
       state free
   
   =visual>
     kind MARKOV-STIMULUS
     stage 1

   =imaginal>
       - curr-state nil
       respond nil
       next-state nil
   
   =goal>
       isa phase
       step plan
       plan-state1 nil
       plan-state2 nil
       > updated-motivation 0
       motivation =MOT
       time-onset =TIME
==> 
   +retrieval>
        isa wm
        status process
        > reward 0
        next-state none
        :recently-retrieved nil      
   
   !bind!       =CURRTIME (mp-time)
   !bind!       =DURATION (- =CURRTIME =TIME)
   !bind! =DIFF (- =MOT =DURATION)
   
   =goal> 
       updated-motivation   =DIFF
   
   =imaginal> 
   
   =visual>

   !output! (plan-backward-state2 motivation =MOT updated-motivation =DIFF)
)

(p plan-backward-at-stage1-state1
   "Plan backward at stage1: state1"
   ?retrieval>
        state free
        buffer full
   
   ?goal>
       state free
   
   =visual>
     kind MARKOV-STIMULUS
     stage 1
   
   =imaginal>
       - curr-state nil
       respond nil
       next-state nil
   
   =goal>
       isa phase
       step plan
       plan-state1 nil
       plan-state2 nil
       > updated-motivation 0
       motivation =MOT
       time-onset =TIME
   
   =retrieval>
        isa wm
        status process
        curr-state =CURR
==> 
   -retrieval>
   
   +retrieval>
        isa wm
        status process
        curr-state A
        reward none
        next-state =CURR
        :recently-retrieved nil       
   
   !bind!       =CURRTIME (mp-time)
   !bind!       =DURATION (- =CURRTIME =TIME)
   !bind! =DIFF (- =MOT =DURATION)
   
   =goal>
       plan-state2 =CURR
       updated-motivation   =DIFF 
   
   =imaginal>
   
   =visual>

   !output! (plan-backward-state1 STATE2 IS =CURR motivation =MOT updated-motivation =DIFF)
)


(p plan-backward-at-stage1-complete
   "Plan until state1"
   ?retrieval>
        state free
        buffer full
   
   ?goal>
       state free
   
   =visual>
     kind MARKOV-STIMULUS
     stage 1
   
   =imaginal>
       - curr-state nil
       response nil
       next-state nil
   
   =goal>
       isa phase
       step plan
       plan-state1 nil
       - plan-state2 nil
       > updated-motivation 0
       motivation =MOT
       time-onset =TIME
   
   =retrieval>
        isa wm
        status process
        curr-state A
        reward none
        response =RESP

==>
   !bind!       =CURRTIME (mp-time)
   !bind!       =DURATION (- =CURRTIME =TIME)
   !bind! =DIFF (- =MOT =DURATION)
   
   =goal> 
       step respond
       plan-state1 nil
       plan-state2 nil
       updated-motivation   =DIFF 
   
   =imaginal>
       response =RESP
   
   -retrieval>
   
   =visual>
   
   !output! (plan-backward-complete STATE1 IS A RESP IS =RESP MOT =MOT updated-motivation =DIFF)
)


;;; ----------------------------------------------------------------
;;; ENCODE STATE1
;;; ----------------------------------------------------------------

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


;;; ----------------------------------------------------------------
;;; ATTEND STATE2
;;; ----------------------------------------------------------------

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
     step       plan-start; plan ;respond
     plan-state2 nil
   
   =visual>
   
   !output! (in attend-state2 =L =R)
)

;;; ----------------------------------------------------------------
;;; PLAN STATE2
;;; ----------------------------------------------------------------

(p plan-backward-at-stage2
   "Plan backward at stage2"
   ?retrieval>
        state free
        buffer empty
   
   ?goal>
       state free
   
   =visual>
     kind MARKOV-STIMULUS
     stage 2

   =imaginal>
       status process
       - curr-state nil
       curr-state =CURR
       response nil
       next-state nil
       reward nil
   
   =goal>
       isa phase
       step plan
       plan-state2 nil
       > updated-motivation 0
       motivation =MOT
       time-onset =TIME
       
==> 
   +retrieval>
        isa wm
        status process
        curr-state =CURR
        > reward 0
        next-state none
        :recently-retrieved nil
   
   !bind!       =CURRTIME (mp-time)
   !bind!       =DURATION (- =CURRTIME =TIME)
   !bind! =DIFF (- =MOT =DURATION)
    
   =goal>
       plan-state2 =CURR
       updated-motivation   =DIFF 
   
   =imaginal>
   
   =visual>

   !output! (plan-backward-at-stage2 curr-time =CURRTIME pdated-motivation =DIFF)
)

 (p plan-backward-at-stage2-complete
   "Plan complete"
   ?retrieval>
        state free
        buffer full
   
   ?goal>
       state free
   
   =visual>
     kind MARKOV-STIMULUS
     stage 2
   
   =imaginal>
       - curr-state nil
       response nil
       next-state nil
   
   =goal>
       isa phase
       step plan
       - plan-state2 nil
       > updated-motivation 0
       motivation =MOT
       time-onset =TIME
   
   =retrieval>
        isa wm
        status process 
        response =RESP

==>
   !bind!       =CURRTIME (mp-time)
   !bind!       =DURATION (- =CURRTIME =TIME)
   !bind! =DIFF (- =MOT =DURATION)
    
   =goal> 
       step respond
       plan-state1 nil
       plan-state2 nil
       updated-motivation   =DIFF 
   
   =imaginal>
       response =RESP
   
   -retrieval>
   
   =visual>
   
   !output! (plan-backward-complete-state2 =RESP updated-motivation =DIFF)
)


;;; ----------------------------------------------------------------
;;; ENCODE STATE2
;;; ----------------------------------------------------------------

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
     ; allows ACT-R set MOTIVATION based on previous trial's reward
     previous-reward =REWARD 
     current-reward  =REWARD
   
   =imaginal>
     reward    =REWARD
     next-state none
   
   -visual>
   
   ;-imaginal>
   
   !eval! (trigger-reward =REWARD)
   
   !output! (in  encode-state2 reward =REWARD)
)

;;; ----------------------------------------------------------------
;;; REFRESH MEMORY
;;; ---------------------------------------------------------------- 

(p refresh-memory
  "refresh memorty   !!!major updates: refreshing until M value is negative "
   ?imaginal>
     state free
     buffer full
   
   ?retrieval>
     state free
     buffer empty
   
   =goal>
     step  refresh-memory
     > updated-motivation 0 
       motivation =MOT
       time-onset =TIME
       updated-motivation =U
   
   =imaginal>
       status  PROCESS
       left-stimulus  =LEFT
       right-stimulus  =RIGHT
       reward  =R
       curr-state  =CURR
       next-state  =NEXT
       response  =RESP
   
==>
   
   !bind!       =CURRTIME (mp-time)
   !bind!       =DURATION (- =CURRTIME =TIME)
   !bind! =DIFF (- =MOT =DURATION)

   =goal>
    step  refresh-memory
    updated-motivation   =DIFF 
   
   =imaginal>
   
   +imaginal>
       isa wm
       status  PROCESS
       left-stimulus  =LEFT
       right-stimulus  =RIGHT
       reward  =R
       curr-state  =CURR
       next-state  =NEXT
       response  =RESP
       ; :recently-retrieved reset
)

(p refresh-success
 "success refresh"
  ?imaginal>
     state free
     buffer full
   
   =goal>
     step  refresh-memory
     > updated-motivation 0
   
==> 
   
   =goal>
    step attend-stimulus 
   
   -imaginal>
   
 )

(p refresh-failure
 "failure refresh"
  ?imaginal>
     state free
     buffer full
   
   =goal>
     step  refresh-memory
     < updated-motivation 0   
   
==> 
   
   =goal>
    step attend-stimulus 
   
   -imaginal>
 )

;;; ----------------------------------------------------------------
;;; RESPONSE SELECTION
;;; ----------------------------------------------------------------
;;; If planed: choose planed action choose-state1-plan-left()/right()
;;; If not planed: randomly choose actions
;;; State1: choose-state1-left() and choose-state1-right() compete
;;; State2: choose-state2-left() and choose-state2-right() compete
;;; ----------------------------------------------------------------

(p choose-state1-left
   "At STATE1: Choose planed stimulus"
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
     response left
     response =RESP
    
   ?retrieval>
     state free
     buffer empty

==>
   +manual>
     isa punch
     hand left
     finger index
   
   =imaginal>
   
   =visual>
   
   =goal>
     step       encode-stimulus 

   !output! (in choose-planed-state1 =RESP)  
)


(p choose-state1-right
   "At STATE1: Choose planed stimulus"
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
     response right
     response =RESP
    
   ?retrieval>
     state free
     buffer empty

==>
   +manual>
     isa punch
     hand right
     finger middle
   
   =imaginal>
   
   =visual>
   
   =goal>
     step       encode-stimulus 

   !output! (in choose-planed-state1 =RESP)  
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
     - response nil
     response left
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
   
   !output! (in choose-state2-plan-right())
  
)

(p choose-state2-right
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
     - response nil
     response right
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
   
   !output! (in choose-state2-plan-right())
  
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

;; (goal-focus start-trial)
