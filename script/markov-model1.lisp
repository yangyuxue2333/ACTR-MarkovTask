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
;;; Version     :v1.0
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
;;; |--- p choose-state1-left()
;;; |--- p choose-state1-right()
;;; ===== state2 =====
;;; p encode-state2-stimulus ()  
;;; |--- p choose-state2-left()
;;; |--- p choose-state2-right()
;;; ===== feedback =====
;;; p encode-reward() 
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

(chunk-type phase
      step
      stage
      motivation)

;;; --------- DM ---------
(add-dm 
  (start-trial isa phase step attend-fixation)
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
     step       attend-fixation 
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
     step       respond
     stage      =STAGE
   
   =visual>
   
   =imaginal> 
     state1-left-stimulus   =L
     state1-right-stimulus  =R
   
   !output! (in encode-state2 =L =R)
)

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
     step       respond
     stage      =STAGE
   
   =visual>
   
   =imaginal>
     state2-left-stimulus   =L
     state2-right-stimulus  =R
   
   !output! (in encode-state2 =L =R)
)

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
     reward nil

==>
   =goal> 
     step       attend-fixation 
     stage      =STAGE
   
   =imaginal>
     reward    =REWARD
   
   -visual>
   
   -imaginal>
   
   !eval! (trigger-reward =REWARD)
   
   !output! (deliver reward =REWARD)
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
        
   =goal>
     isa        phase
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
        
   =goal>
     isa        phase
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


(p choose-state2-left
   "At STATE2: Choose LEFT stimulus"
   ?manual>
     preparation free
     processor free
     execution free
   
   =visual>
     kind markov-stimulus
        
   =goal>
     isa        phase
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
        
   =goal>
     isa        phase
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
