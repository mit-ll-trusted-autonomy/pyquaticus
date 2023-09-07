#=======================================================================
# FILE:  moos-ivp-extend/CMakeLists.txt
# DATE:  2012/07/24
# INFO:  Top-level CMakeLists.txt file for the moos-ivp-extend project
# NAME:  Maintained by Mike Benjamin - Original setup by Christian Convey
#        Chris Gagner, and tips borrowed from Dave Billin
#=======================================================================

CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT( IVP_EXTEND )

set (CMAKE_CXX_STANDARD 11)

#=======================================================================
# Set the output directories for the binary and library files
#=======================================================================

GET_FILENAME_COMPONENT(IVP_EXTEND_BIN_DIR "${CMAKE_SOURCE_DIR}/bin"  ABSOLUTE )
GET_FILENAME_COMPONENT(IVP_EXTEND_LIB_DIR "${CMAKE_SOURCE_DIR}/lib"  ABSOLUTE )

SET( LIBRARY_OUTPUT_PATH      "${IVP_EXTEND_LIB_DIR}" CACHE PATH "" )
SET( ARCHIVE_OUTPUT_DIRECTORY "${IVP_EXTEND_LIB_DIR}" CACHE PATH "" )
SET( LIBRARY_OUTPUT_DIRECTORY "${IVP_EXTEND_LIB_DIR}" CACHE PATH "" )

SET( EXECUTABLE_OUTPUT_PATH    "${IVP_EXTEND_BIN_DIR}" CACHE PATH "" )
SET( RUNTIME_OUTPUT_DIRECTORY "${IVP_EXTEND_BIN_DIR}"  CACHE PATH "" )

#=======================================================================
# Find MOOS
#=======================================================================
find_package(MOOS 10.0)
INCLUDE_DIRECTORIES(${MOOS_INCLUDE_DIRS})

#=======================================================================
# Find the "moos-ivp" base directory
#=======================================================================

# Search for the moos-ivp folder
find_path( MOOSIVP_SOURCE_TREE_BASE
           NAMES build-ivp.sh build-moos.sh configure-ivp.sh
           PATHS "../moos-ivp" "../../moos-ivp" "../../moos-ivp/trunk/" "../moos-ivp/trunk/"
           DOC "Base directory of the MOOS-IvP source tree"
           NO_DEFAULT_PATH
)

if (NOT MOOSIVP_SOURCE_TREE_BASE)
    message("Please set MOOSIVP_SOURCE_TREE_BASE to  ")
    message("the location of the \"moos-ivp\" folder ")
    return()
endif()

#======================================================================
# Specify where to find IvP's headers and libraries...
#======================================================================

FILE(GLOB IVP_INCLUDE_DIRS ${MOOSIVP_SOURCE_TREE_BASE}/ivp/src/lib_* )
INCLUDE_DIRECTORIES(${IVP_INCLUDE_DIRS})

FILE(GLOB IVP_LIBRARY_DIRS ${MOOSIVP_SOURCE_TREE_BASE}/lib )
LINK_DIRECTORIES(${IVP_LIBRARY_DIRS})

#======================================================================
# Specify Compiler Flags
#======================================================================
IF( ${WIN32} )
   #---------------------------------------------
   # Windows Compiler Flags
   #---------------------------------------------
   IF(MSVC)
      # Flags for Microsoft Visual Studio
      SET( WALL_ON OFF CACHE BOOL 
         "tell me about all compiler warnings (-Wall) ")
      IF(WALL_ON)
         SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
      ENDIF(WALL_ON)
   ELSE(MSVC)
      # Other Windows compilers go here
   ENDIF(MSVC)

ELSE( ${WIN32} )
   #---------------------------------------------
   # Linux and Apple Compiler Flags
   #---------------------------------------------
   # Force -fPIC because gcc complains when we don't use it with x86_64 code.
   # Note sure why: -fPIC should only be needed for shared objects, and
   # AFAIK, CMake gets that right when building shared objects. -CJC
   SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -g")
   IF(CMAKE_COMPILER_IS_GNUCXX)
      # Flags for the GNU C++ Compiler
      SET( WALL_ON OFF CACHE BOOL 
         "tell me about all compiler warnings (-Wall) ")
      IF(WALL_ON)
         SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall" -C++11)
      ENDIF( WALL_ON)
   ELSE(CMAKE_COMPILER_IS_GNUCXX)
    
   ENDIF(CMAKE_COMPILER_IS_GNUCXX)

ENDIF( ${WIN32} )

#=======================================================================
# Add Subdirectories
#=======================================================================
ADD_SUBDIRECTORY( src )
