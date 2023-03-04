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
;;; Version     :v2.8
;;;
;;; Description : model-base 
;;;
;;; Bugs        :
;;;
;;;
;;; To do       :
;;;
;;;
;;; v key updates: use blending
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
      previous-reward               ;;; reward received in previous trial
      ;;; blending
      state-b-blended-value         ;;; 
      state-c-blended-value         ;;;
      diff-blended-value            ;;; state-b-blended-value - state-c-blended-value
      best-blended-state            ;;; "B" or "C"
)

(chunk-type response
    (response-type t)
    action)

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
        status prepare
        curr-state nil
        left-stimulus nil
        right-stimulus nil
        next-state nil
        response nil
        reward nil

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

    =imaginal>
   
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

    =imaginal>
      status process
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

   =imaginal>
    curr-state =STATE
)

;;; ----------------------------------------------------------------
;;; PLAN (BACKWARD-BLENDING)
;;; ----------------------------------------------------------------
;;; planing: stage1-state2
;;;;;; BLEND-STATE-B
;;;;;; BLEND-STATE-C
;;;;;; if: BLEND-VALUE-B > BLEND-VALUE-C: CHOOSE-B
;;;;;; else: CHOOSE-C
;;; planing: stage1-state1
;;;;;; RETRIEVE-RESPONSE: retrieve the response that mostly likely
;;;;;; leading to best-state
;;; ----------------------------------------------------------------

(p plan-backward-at-stage1-start
   "Plan backward at stage1: state2"
   ?blending>
        state free
        buffer empty
        error nil
   
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
    =goal>
        step plan-blend
    +blending>
        isa wm
        curr-state B
        :ignore-slots (STATUS RESPONSE NEXT-STATE LEFT-STIMULUS RIGHT-STIMULUS)

   =imaginal>
   =goal>
   =visual>
)

(p plan-backward-at-stage1-blend-b
     =visual>
         kind MARKOV-STIMULUS
         stage 1
     
     ?blending>
       state free

     ?imaginal>
        state free 
        buffer full

     ?goal>
        state free 
    =goal>
        isa phase
        step plan-blend
        state-b-blended-value nil
        state-c-blended-value nil

    =blending>
       isa wm
       reward =val 
       curr-state =s


  ==>
   !output! (state =s blended reward is =val)

   ; Overwrite the blended chunk to erase it and keep it
   ; from being added to dm.  Not necessary, but keeps the
   ; examples simpler.

   @blending>

   +blending>
     isa wm
     curr-state C  
     :ignore-slots (STATUS RESPONSE NEXT-STATE LEFT-STIMULUS RIGHT-STIMULUS)

   =goal>
      state-b-blended-value =val

   =visual>
)

(p plan-backward-at-stage1-blend-c
     =visual>
         kind MARKOV-STIMULUS
         stage 1
         
     ?blending>
       state free

     ?imaginal>
        state free 
        buffer full

     ?goal>
        state free

     =blending>
       isa wm
       reward =c
       curr-state =s
        

     =goal>
        isa phase
        step plan-blend
        - state-b-blended-value nil
        state-c-blended-value nil
        state-b-blended-value =b
    
     ==>
     
     !output! (state =s blended reward is =c)

     ; Overwrite the blended chunk to erase it and keep it
     ; from being added to dm.  Not necessary, but keeps the
     ; examples simpler.

     @blending>

     =visual>
        
     
    !bind! =diff(- =b =c)

    =goal>
        step plan-evaluate 
        state-c-blended-value =c
        diff-blended-value =diff
  )

(p plan-backward-at-stage1-choose-b 
    =visual>
         kind MARKOV-STIMULUS
         stage 1

    ?imaginal>
        state free 
        buffer full

    ?goal>
        state free
        buffer full
    =goal>
        isa phase
        step plan-evaluate
        >= diff-blended-value 0
 
==> 
   
   =goal> 
        step plan-retrieve  
        plan-state2 B
        best-blended-state  B
   
   =visual>
       
  )


(p plan-backward-at-stage1-choose-c 
    =visual>
         kind MARKOV-STIMULUS
         stage 1

    ?imaginal>
        state free 
        buffer full

    ?goal>
        state free
        buffer full

    =goal>
        isa phase
        step plan-evaluate
        <= diff-blended-value 0
 
==> 
   
   =goal> 
        step plan-retrieve 
        plan-state2 C
        best-blended-state C

   =visual>        
  )

(p plan-backward-at-stage1-retrieve-response
    =visual>
         kind MARKOV-STIMULUS
         stage 1

  ?retrieval>
        state free 
        buffer empty

  ?imaginal>
        state free 
        buffer full

  ?goal>
      state free
      buffer full 
  =goal>
      isa phase
      step plan-retrieve
      - best-blended-state  nil
      best-blended-state  =best-blended-state 
 
==> 
  ; retrieve the response that leads to next-state = best-blended state
  +retrieval> 
      isa wm
      next-state  =best-blended-state 
      > reward 0
  
  =visual>
  =goal> 
        step plan-complete
)

(p plan-backward-at-stage1-complete
   "Plan until state1"
   =visual>
         kind MARKOV-STIMULUS
         stage 1
   ?retrieval>
        state free 
        buffer full
    ?imaginal>
        state free 
        buffer full
    ?blending>
       state free

    ?goal>
        state free 
    =retrieval>
        response =RESP

    =imaginal>
       - curr-state nil
       response nil
       next-state nil
   
   =goal>
       isa phase
       step plan-complete
       plan-state1 nil
       - plan-state2 nil
       - best-blended-state  nil
       best-blended-state  =best-blended-state 
==>
   =goal> 
       step respond
       ; reset goal blended values
       plan-state1 nil
       plan-state2 nil
       state-b-blended-value nil
       state-c-blended-value nil
       plan-state1-response =RESP
   
   =imaginal>
       response =RESP
   
   -retrieval>
   -blending>
   
   =visual>
   
   ; !output! (plan1-2 response =RESP retrieved S2 =NEXT)
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
     ; do not refresh state1
     step       attend-stimulus ;refresh-memory 
     stage      =STAGE
   
   =visual>
   -imaginal>
   
   ; =imaginal>
   ;   next-state =STATE
   ;   reward none 
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
     step       plan-retrieve 
     plan-state2 nil
   
   =visual>
)

; (p plan-backward-at-stage2
;    "Plan backward at stage2"
;    ?blending>
;         state free
   
;    ?goal>
;        state free
   
;    =visual>
;      kind MARKOV-STIMULUS
;      stage 2

;    =imaginal>
;        status process
;        - curr-state nil
;        curr-state =CURR
;        response nil
;        next-state nil
;        reward nil
   
;    =goal>
;        isa phase
;        step plan
;        plan-state2 nil
; ==> 
;    +blending>
;         isa wm
;         status process
;         curr-state =CURR
;         > reward 0
;         next-state none

;    =imaginal>
   
;    =goal>
;        plan-state2 =CURR
   
;    =visual>
; )
(p plan-backward-at-stage2-retrieve-response
    "Plan backward at stage2"
    =visual>
     kind MARKOV-STIMULUS
     stage 2 

  ?retrieval>
        state free 
        buffer empty

  ?imaginal>
        state free 
        buffer full

  ?goal>
      state free
      buffer full 

  =imaginal>
       status process
       - curr-state nil
       curr-state =CURR
       response nil
       next-state nil
       reward nil

  =goal>
      isa phase
      step plan-retrieve
      plan-state2 nil
      
==> 
  ; retrieve the response that leads to next-state = best-blended state
  =goal> 
        step plan-complete
        plan-state2 =CURR
  +retrieval> 
      isa wm
      curr-state  =CURR
      > reward 0

  =imaginal>
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
       step plan-complete
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
     ;next-state none
   
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
    !output! (encode state2 curr-state A next-state  =CURR response  =RESP1 reward =R)
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