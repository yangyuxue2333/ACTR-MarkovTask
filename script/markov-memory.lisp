(add-dm 
 (M1-1 isa wm
      status process
      state1-left-stimulus A1
      state1-right-stimulus A2
      state2-left-stimulus B1
      state2-right-stimulus B2
      state1-selected-stimulus LEFT
      state2-selected-stimulus LEFT
      reward 2)
 (M1-2 isa wm
      status process
      state1-left-stimulus A1
      state1-right-stimulus A2
      state2-left-stimulus B1
      state2-right-stimulus B2
      state1-selected-stimulus LEFT
      state2-selected-stimulus RIGHT
      reward 2)
 (M1-3 isa wm
      status process
      state1-left-stimulus A1
      state1-right-stimulus A2
      state2-left-stimulus C1
      state2-right-stimulus C2
      state1-selected-stimulus RIGHT
      state2-selected-stimulus LEFT
      reward 2)
  (M1-4 isa wm
      status process
      state1-left-stimulus A1
      state1-right-stimulus A2
      state2-left-stimulus C1
      state2-right-stimulus C2
      state1-selected-stimulus RIGHT
      state2-selected-stimulus RIGHT
      reward 2)
 
  ;; RARE PATH
  (M2-1 isa wm
      status process
      state1-left-stimulus A1
      state1-right-stimulus A2
      state2-left-stimulus C1
      state2-right-stimulus C2
      state1-selected-stimulus LEFT
      state2-selected-stimulus LEFT
      reward 2)
   (M2-2 isa wm
      status process
      state1-left-stimulus A1
      state1-right-stimulus A2
      state2-left-stimulus C1
      state2-right-stimulus C2
      state1-selected-stimulus LEFT
      state2-selected-stimulus RIGHT
      reward 2)
    (M2-3 isa wm
      status process
      state1-left-stimulus A1
      state1-right-stimulus A2
      state2-left-stimulus B1
      state2-right-stimulus B2
      state1-selected-stimulus RIGHT
      state2-selected-stimulus LEFT
      reward 2) 
     (M2-4 isa wm
      status process
      state1-left-stimulus A1
      state1-right-stimulus A2
      state2-left-stimulus B1
      state2-right-stimulus B2
      state1-selected-stimulus RIGHT
      state2-selected-stimulus RIGHT
      reward 2)
 
 
 ;;; ZERO REWARD
 ;;; COMMON PATH
 (M3-1 isa wm
      status process
      state1-left-stimulus A1
      state1-right-stimulus A2
      state2-left-stimulus B1
      state2-right-stimulus B2
      state1-selected-stimulus LEFT
      state2-selected-stimulus LEFT
      reward 0)
 (M3-2 isa wm
      status process
      state1-left-stimulus A1
      state1-right-stimulus A2
      state2-left-stimulus B1
      state2-right-stimulus B2
      state1-selected-stimulus LEFT
      state2-selected-stimulus RIGHT
      reward 0)
 (M3-3 isa wm
      status process
      state1-left-stimulus A1
      state1-right-stimulus A2
      state2-left-stimulus C1
      state2-right-stimulus C2
      state1-selected-stimulus RIGHT
      state2-selected-stimulus LEFT
      reward 0)
  (M3-4 isa wm
      status process
      state1-left-stimulus A1
      state1-right-stimulus A2
      state2-left-stimulus C1
      state2-right-stimulus C2
      state1-selected-stimulus RIGHT
      state2-selected-stimulus RIGHT
      reward 0)
 
  ;;; RARE PATH
  (M4-1 isa wm
      status process
      state1-left-stimulus A1
      state1-right-stimulus A2
      state2-left-stimulus C1
      state2-right-stimulus C2
      state1-selected-stimulus LEFT
      state2-selected-stimulus LEFT
      reward 0)
   (M4-2 isa wm
      status process
      state1-left-stimulus A1
      state1-right-stimulus A2
      state2-left-stimulus C1
      state2-right-stimulus C2
      state1-selected-stimulus LEFT
      state2-selected-stimulus RIGHT
      reward 0)
    (M4-3 isa wm
      status process
      state1-left-stimulus A1
      state1-right-stimulus A2
      state2-left-stimulus B1
      state2-right-stimulus B2
      state1-selected-stimulus RIGHT
      state2-selected-stimulus LEFT
      reward 0) 
    (M4-4 isa wm
      status process
      state1-left-stimulus A1
      state1-right-stimulus A2
      state2-left-stimulus B1
      state2-right-stimulus B2
      state1-selected-stimulus RIGHT
      state2-selected-stimulus RIGHT
      reward 0)
 )