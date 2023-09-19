#--------------------------------------------------------
# The CMakeLists.txt for:                     pRLMonitor
# Author(s):                              Mike McCarrick
#--------------------------------------------------------

SET(SRC
  RLMonitor.cpp
  RLMonitor_Info.cpp
  main.cpp
)

ADD_EXECUTABLE(pRLMonitor ${SRC})

TARGET_LINK_LIBRARIES(pRLMonitor
   ${MOOS_LIBRARIES}
   mbutil
   m
   pthread)

