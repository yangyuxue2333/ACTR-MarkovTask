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
;;; Description :This declarative model simulates markov task based on xx (2006)'s
;;;              paradigm. This model takes motivation parameter as mental clock:
;;;              if the time used longer than model's motivation parameter (in sec unit),
;;;              it would give up checking; the the time used withine the limit, the
;;;              model would continue retrieving.
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
;;; The model looks at the fixation cross on the screen and prepare WM. Once
;;; the markov-stimulus occurs, it would encode the stimulus in WM.
;;; 4 productions compete: choose-left(), choose-right(),
;;; dont-choose-left(), dont-choose-right().
;;; In the end of each trial, a reward is delivered
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; 
;;; Productions: 
;;; ===== fixation =====
;;; p prepare-wm ()
;;; p find-screen ()  
;;; ===== stimulus =====
;;; p encode-stimulus ()  
;;; |--- p choose-left()
;;; |--- p choose-right()
;;; |--- p dont-choose-left()
;;; |--- p dont-choose-right ()
;;; ===== response ===== 
;;; p respond()
;;; ===== feedback =====
;;; p encode-feedback() 
;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(clear-all)
(define-model markov-model
    (sgp :seed (100 0)
         :er t
         :esc t
         :ncnar t ;normalize chunk names after run
         :model-warnings nil
         :ans 0.1
         :auto-attend t
         ;:le 0.63
         ;:lf 0.1
         ;:bll 0.1
         ;:mas 4.0
         :rt -100  ;retrieval threshold
         :ul t
         :egs 0.1
         ;:alpha 0.4
         ;:imaginal-activation 3.0
         ;:motor-feature-prep-time 0.01
         ;:dat 0.05  ; default action time for all productions
         :show-focus t 
         :needs-mouse t
         :model-warnings nil
         :v nil
         :trace-detail low
         ;:ult t
         ;:act t
         :trace-filter production-firing-only
         ;:pct t
         ;:blt t
         ;:reward-hook "detect-reward-hook"
         ;:cycle-hook "detect-production-hook"

    )
)