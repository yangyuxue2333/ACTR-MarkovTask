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
;;; Version     :v3.8
;;;
;;; Description : model-motivation 
;;;
;;; Bugs        :
;;;
;;;
;;; To do       : 
;;;
;;;
;;; v key updates: updated based on MB v2.8, use blending
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
      previous-reward               ;;; previous reward given

      state-b-blended-value         ;;; blended value of state-b
      state-c-blended-value         ;;; blended value of state-c
      diff-blended-value            ;;; state-b-blended-value - state-c-blended-value
      best-blended-state            ;;; selected best state based on blended value
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
     ; motivation is fixed for each trial
     ; motivation =R
      
   !eval! (trigger-reward 0) ; CLEAR REWARD  
   
   !output! (INIT MOTIVATION =MOT TIME-ONSET =TIME PRE-REWARD =R)
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
      
      
    !output! (T =CURRTIME updated-motivation =MOT)
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
   !output! (MOTIVATION =MOT CURR TIME =CURRTIME UPDATED-MOTIVATION =DIFF)
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
   
   !output! (UPDATED-MOTIVATION =MOT)
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

(p plan-backward-at-stage1-start
   "Plan backward at stage1: state2"
   ?blending>
        state free
        buffer empty
        error nil
   
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

    !bind!       =CURRTIME (mp-time)
    !bind!       =DURATION (- =CURRTIME =TIME)
    !bind! =DIFF (- =MOT =DURATION) 

    =goal>
        step plan-blend
        updated-motivation   =DIFF

    +blending>
        isa wm
        curr-state B
        :ignore-slots (STATUS RESPONSE NEXT-STATE LEFT-STIMULUS RIGHT-STIMULUS)

   =imaginal> 
   =visual>

   !output! (MOTIVATION =MOT CURR TIME =CURRTIME UPDATED-MOTIVATION =DIFF)
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
       > updated-motivation 0
       motivation =MOT
       time-onset =TIME

==>
   !bind!       =CURRTIME (mp-time)
   !bind!       =DURATION (- =CURRTIME =TIME)
   !bind! =DIFF (- =MOT =DURATION)
   =goal> 
       step respond
       ; reset goal blended values
       plan-state1 nil
       plan-state2 nil
       state-b-blended-value nil
       state-c-blended-value nil
       plan-state1-response =RESP
       updated-motivation   =DIFF 
   
   =imaginal>
       response =RESP
   
   -retrieval>
   -blending>
   
   =visual> 
   !output! (MOTIVATION =MOT CURR TIME =CURRTIME UPDATED-MOTIVATION =DIFF)
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
     response =RESP1

   ?retrieval>
     state free
     buffer empty
   
   ?manual> 
     state free

   =goal>
     isa        phase
     step       encode-stimulus 
     plan-state1-response =RESP2
==> 
   
   =goal>
    step attend-stimulus 
    stage      =STAGE 

 !output! (imaginal =RESP1 plan-state1-response =RESP2)
   
   =visual>
   @imaginal>
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
)

;;; ----------------------------------------------------------------
;;; PLAN STATE2
;;; ----------------------------------------------------------------

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
      step plan
      plan-state2 nil
      > updated-motivation 0
      motivation =MOT
      time-onset =TIME
      
==>
   !bind!       =CURRTIME (mp-time)
   !bind!       =DURATION (- =CURRTIME =TIME)
   !bind! =DIFF (- =MOT =DURATION)
  ; retrieve the response that leads to next-state = best-blended state
  =goal> 
        plan-state2 =CURR
        updated-motivation   =DIFF
  +retrieval>
      isa wm
      status process
      curr-state  =CURR
      > reward 0

  =imaginal>
  =visual>

  !output! (curr-state =CURR updated-motivation   =DIFF)
  
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

   !output! (curr-state =CURR response =RESP)
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
   
   -visual> 
   
   !eval! (trigger-reward =REWARD)
   
   !output! (reward =REWARD)
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
     plan-state1-response  =RESP1
     > updated-motivation 0 
       motivation =MOT
       time-onset =TIME
       updated-motivation =U
   
   =imaginal>
       status  PROCESS 
       reward  =R
       curr-state  =CURR 
       response  =RESP
   
==>
   !output! (encode state2 curr-state  =CURR response  =RESP reward  =R)

   !bind!       =CURRTIME (mp-time)
   !bind!       =DURATION (- =CURRTIME =TIME)
   !bind! =DIFF (- =MOT =DURATION)

   =goal>
    step  refresh-success
    updated-motivation   =DIFF
   

   -imaginal>
   +imaginal>
       isa wm
       status  PROCESS
       left-stimulus  A1
       right-stimulus  A2
       reward  =R
       curr-state  A
       next-state  =CURR
       response  =RESP1
)

(p refresh-success
 "success refresh"
  ?imaginal>
     state free
     buffer full
   
   =goal>
     step  refresh-success 

   =imaginal>
       status  PROCESS 
       reward  =R
       next-state  =NEXT
       response  =RESP1
   
==> 

   !output! (encode state1 curr-state A next-state  =NEXT response  =RESP1 reward =R)

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
     <= updated-motivation 0   
   
==> 
   
   =goal>
    step attend-stimulus 
   
   @imaginal> ; overwrite, not harvest
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
     step      encode-stimulus 

   
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
     step      encode-stimulus 
   
   =imaginal>
      response right 
  
)

(p complete-motor-response2

    ?manual>
     preparation free
     processor free
     execution free
   
   =visual>
     kind markov-stimulus
     stage 2
        
   =goal>
     isa        phase
     step       complete-motor-response2

   =imaginal>
     status process 
     - curr-state nil
     - left-stimulus nil
     - right-stimulus nil
     - response nil
     next-state nil
     reward nil
==>
    =visual>
   
    =goal>
     step       encode-stimulus 

    =imaginal> 

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



; ######### SETUP MODEL markov-model3-8 #########
;     >> TASK PARAMETERS: {'MARKOV_PROBABILITY': 0.7, 'REWARD_PROBABILITY': {}, 'REWARD': {'B1': (1, -1), 'B2': (1, -1), 'C1': (1, -1), 'C2': (1, -1)}, 'RANDOM_WALK': 'LOAD', 'M': 5} <<
;     >> ACT-R PARAMETERS: {'v': 't', 'seed': '[100, 0]', 'ans': 0.2, 'lf': 0.1, 'bll': 0.2, 'egs': 0.2, 'alpha': 0.2} <<

;      0.050   PROCEDURAL             PRODUCTION-FIRED PREPARE-WM
; INIT MOTIVATION 5 TIME-ONSET 0.0 PRE-REWARD 0.0
;      1.050   PROCEDURAL             PRODUCTION-FIRED FIND-SCREEN
;      1.185   PROCEDURAL             PRODUCTION-FIRED PROCESS-FIXATION
; T 1.185 UPDATED-MOTIVATION 5
;      2.135   PROCEDURAL             PRODUCTION-FIRED FIND-SCREEN
;      2.270   PROCEDURAL             PRODUCTION-FIRED ATTEND-STATE1
;      2.520   PROCEDURAL             PRODUCTION-FIRED PLAN-START
; MOTIVATION 5 CURR TIME 2.52 UPDATED-MOTIVATION 3.665
;      2.570   PROCEDURAL             PRODUCTION-FIRED PLAN-BACKWARD-AT-STAGE1-START
; MOTIVATION 5 CURR TIME 2.57 UPDATED-MOTIVATION 3.615
;      2.643   PROCEDURAL             PRODUCTION-FIRED PLAN-BACKWARD-AT-STAGE1-BLEND-B
; STATE B BLENDED REWARD IS 0.35900998
;      2.716   PROCEDURAL             PRODUCTION-FIRED PLAN-BACKWARD-AT-STAGE1-BLEND-C
; STATE C BLENDED REWARD IS -0.3435566
;      2.766   PROCEDURAL             PRODUCTION-FIRED PLAN-BACKWARD-AT-STAGE1-CHOOSE-B
;      2.816   PROCEDURAL             PRODUCTION-FIRED PLAN-BACKWARD-AT-STAGE1-RETRIEVE-RESPONSE
;      2.884   PROCEDURAL             PRODUCTION-FIRED PLAN-BACKWARD-AT-STAGE1-COMPLETE
; MOTIVATION 5 CURR TIME 2.884 UPDATED-MOTIVATION 3.3009999
;      2.934   PROCEDURAL             PRODUCTION-FIRED CHOOSE-STATE1-RIGHT
;      3.194   PROCEDURAL             PRODUCTION-FIRED FIND-SCREEN
;      3.329   PROCEDURAL             PRODUCTION-FIRED ENCODE-STATE1
; IMAGINAL RIGHT PLAN-STATE1-RESPONSE RIGHT
;      3.379   PROCEDURAL             PRODUCTION-FIRED ATTEND-STATE2
;      3.629   PROCEDURAL             PRODUCTION-FIRED PLAN-START
; MOTIVATION 5 CURR TIME 3.629 UPDATED-MOTIVATION 2.556
;      3.679   PROCEDURAL             PRODUCTION-FIRED PLAN-BACKWARD-AT-STAGE2-RETRIEVE-RESPONSE
; CURR-STATE C UPDATED-MOTIVATION 2.506
;      3.810   PROCEDURAL             PRODUCTION-FIRED PLAN-BACKWARD-AT-STAGE2-COMPLETE
; CURR-STATE C RESPONSE RIGHT
;      3.860   PROCEDURAL             PRODUCTION-FIRED CHOOSE-STATE2-RIGHT
;      3.970   PROCEDURAL             PRODUCTION-FIRED FIND-SCREEN
;      4.105   PROCEDURAL             PRODUCTION-FIRED ENCODE-STATE2
; REWARD -1
; <[MARKOV_STATE]     [R, 1.06]'A2'   [R, 0.73]'C2'   R:[-1]  [C][C]
;      4.155   PROCEDURAL             PRODUCTION-FIRED REFRESH-MEMORY
; ENCODE STATE2 CURR-STATE C RESPONSE RIGHT REWARD -1
;      4.405   PROCEDURAL             PRODUCTION-FIRED REFRESH-SUCCESS
; ENCODE STATE1 CURR-STATE A NEXT-STATE C RESPONSE RIGHT REWARD -1
;      4.455   PROCEDURAL             PRODUCTION-FIRED PREPARE-WM
; INIT MOTIVATION 5 TIME-ONSET 1.185 PRE-REWARD -1
;    101.050   PROCEDURAL             PRODUCTION-FIRED FIND-SCREEN
;    101.185   PROCEDURAL             PRODUCTION-FIRED PROCESS-FIXATION
; T 101.185 UPDATED-MOTIVATION 5
;    102.135   PROCEDURAL             PRODUCTION-FIRED FIND-SCREEN
;    102.270   PROCEDURAL             PRODUCTION-FIRED ATTEND-STATE1
;    102.520   PROCEDURAL             PRODUCTION-FIRED PLAN-START
; MOTIVATION 5 CURR TIME 102.52 UPDATED-MOTIVATION 3.665001
;    102.570   PROCEDURAL             PRODUCTION-FIRED PLAN-BACKWARD-AT-STAGE1-START
; MOTIVATION 5 CURR TIME 102.57 UPDATED-MOTIVATION 3.6149979
;    102.668   PROCEDURAL             PRODUCTION-FIRED PLAN-BACKWARD-AT-STAGE1-BLEND-B
; STATE B BLENDED REWARD IS 0.83848315
;    102.744   PROCEDURAL             PRODUCTION-FIRED PLAN-BACKWARD-AT-STAGE1-BLEND-C
; STATE C BLENDED REWARD IS -0.761646
;    102.794   PROCEDURAL             PRODUCTION-FIRED PLAN-BACKWARD-AT-STAGE1-CHOOSE-B
;    102.844   PROCEDURAL             PRODUCTION-FIRED PLAN-BACKWARD-AT-STAGE1-RETRIEVE-RESPONSE
;    102.947   PROCEDURAL             PRODUCTION-FIRED PLAN-BACKWARD-AT-STAGE1-COMPLETE
; MOTIVATION 5 CURR TIME 102.947 UPDATED-MOTIVATION 3.237999
;    102.997   PROCEDURAL             PRODUCTION-FIRED CHOOSE-STATE1-LEFT
;    103.207   PROCEDURAL             PRODUCTION-FIRED FIND-SCREEN
;    103.342   PROCEDURAL             PRODUCTION-FIRED ENCODE-STATE1
; IMAGINAL LEFT PLAN-STATE1-RESPONSE LEFT
;    103.392   PROCEDURAL             PRODUCTION-FIRED ATTEND-STATE2
;    103.642   PROCEDURAL             PRODUCTION-FIRED PLAN-START
; MOTIVATION 5 CURR TIME 103.642 UPDATED-MOTIVATION 2.5429993
;    103.692   PROCEDURAL             PRODUCTION-FIRED PLAN-BACKWARD-AT-STAGE2-RETRIEVE-RESPONSE
; CURR-STATE B UPDATED-MOTIVATION 2.4929962
;    103.902   PROCEDURAL             PRODUCTION-FIRED PLAN-BACKWARD-AT-STAGE2-COMPLETE
; CURR-STATE B RESPONSE RIGHT
;    103.952   PROCEDURAL             PRODUCTION-FIRED CHOOSE-STATE2-RIGHT
;    104.162   PROCEDURAL             PRODUCTION-FIRED FIND-SCREEN
;    104.297   PROCEDURAL             PRODUCTION-FIRED ENCODE-STATE2
; REWARD -1
; <[MARKOV_STATE]     [L, 1.07]'A1'   [R, 0.91]'B2'   R:[-1]  [C][C]
;    104.347   PROCEDURAL             PRODUCTION-FIRED REFRESH-MEMORY
; ENCODE STATE2 CURR-STATE B RESPONSE RIGHT REWARD -1
;    104.597   PROCEDURAL             PRODUCTION-FIRED REFRESH-SUCCESS
; ENCODE STATE1 CURR-STATE A NEXT-STATE B RESPONSE LEFT REWARD -1
;    104.647   PROCEDURAL             PRODUCTION-FIRED PREPARE-WM
; INIT MOTIVATION 5 TIME-ONSET 101.185 PRE-REWARD -1
;    180.050   PROCEDURAL             PRODUCTION-FIRED FIND-SCREEN
;    180.185   PROCEDURAL             PRODUCTION-FIRED DONE