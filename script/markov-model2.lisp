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
;;; Filename    :markov-model2.lisp
;;; Version     :v2.4
;;;
;;; Description : model-base 
;;;
;;; Bugs        :
;;;
;;;
;;; To do       :
;;;
;;;
;;; v key updates: use encoding, rather than retrieval in memory refresh
;;;    refresh all trial memories
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
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; 
;;; Productions: 
;;;; ===== FIXATION =====
;;; P PREPARE-WM ()
;;; P FIND-SCREEN ()  
;;; P PROCESS-FIXATION()
;;; ===== STATE1 =====
;;; P ATTEND-STATE1()
;;; P PLAN-STATE1 ()  
;;; |--------- P PLAN-BACKWARD-AT-STAGE1-STATE2()
;;; |--------- P PLAN-BACKWARD-AT-STAGE1-STATE1()
;;; |--------- P PLAN-BACKWARD-AT-STAGE1-COMPLETE()
;;; |--------- P CHOOSE-STATE1-LEFT()
;;; |--------- P CHOOSE-STATE1-RIGHT()
;;; P ENCODE-STATE1()
;;; P REFRESH-MEMORY() 
;;; P REFRESH-MEMORY-SUCCESS()
;;; ===== STATE2 =====
;;; P ENCODE-STATE2-STIMULUS ()  
;;; P PLAN-STATE2 ()  
;;; |--------- P PLAN-BACKWARD-AT-STAGE2()
;;; |--------- P PLAN-BACKWARD-AT-STAGE2-COMPLETE()
;;; |--------- P CHOOSE-STATE2-LEFT()
;;; |--------- P CHOOSE-STATE2-RIGHT() 
;;; ===== REWARD =====
;;; P ENCODE-STATE2() 
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
      current-reward                ;;; reward received in current trial
      previous-reward)              ;;; reward received in previous trial



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
     time-onset =TIME
     previous-reward =R
==>
   +imaginal>
     isa wm
     status process
   
   *goal>
     motivation =R

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
     motivation =MOT 
==>   
   !bind!       =TIME (mp-time)
   
    =goal> 
      stage      0
      ; INIT MOT (keep track of discounted motivation)
      updated-motivation  =MOT  
      time-onset =TIME
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
     step       plan 
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
(p plan-backward-at-stage1-state2
   "Plan backward at stage1: state2"
   ?retrieval>
        state free
        buffer empty
   
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
==> 
   +retrieval>
        isa wm
        status process
        > reward 0
        next-state none
        :recently-retrieved nil
   
   =imaginal>
   =goal>
   =visual>
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
   
   =imaginal>
   
   =visual>
   
   =goal>
       plan-state2 =CURR

   !output! (plan1-1 retrieved S2 =CURR)
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
   
   =retrieval>
        isa wm
        status process
        curr-state A
        next-state =NEXT
        reward none
        response =RESP

==>
   
   =goal> 
       step respond
       plan-state1 nil
       plan-state2 nil
   
   =imaginal>
       response =RESP
   
   -retrieval>
   
   =visual>
   
   !output! (plan1-2 response =RESP retrieved S2 =NEXT)
)

;;; ----------------------------------------------------------------
;;; ENCODE STATE1
;;; ----------------------------------------------------------------
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
     step       plan 
     plan-state2 nil
   
   =visual>
)

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
==> 
   +retrieval>
        isa wm
        status process
        curr-state =CURR
        > reward 0
        next-state none

   =imaginal>
   
   =goal>
       plan-state2 =CURR
   
   =visual>
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
       curr-state =CURR
       response nil
       next-state nil
   
   =goal>
       isa phase
       step plan
       - plan-state2 nil
   
   =retrieval>
        isa wm
        status process 
        response =RESP
==>
   
   =goal> 
       step respond
       plan-state1 nil
       plan-state2 nil
   
   =imaginal>
       response =RESP
   
   -retrieval>
   
   =visual>

   !output! (PLAN2 REAL curr-state =CURR response =RESP)
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
     next-state none
   
   -visual>
      
   !eval! (trigger-reward =REWARD)
)

;;; ----------------------------------------------------------------
;;; REFRESH MEMORY
;;; ---------------------------------------------------------------- 
;;; Retrieve WM 
;;; ----------------------------------------------------------------

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
    step  refresh-success  ; one-time refresh
   
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
;;; If planed: choose planed action choose-state1-plan-left()/right()
;;; If not planed: randomly choose actions
;;; State1: choose-state1-left() and choose-state1-right() 
;;; State2: choose-state2-left() and choose-state2-right() 
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