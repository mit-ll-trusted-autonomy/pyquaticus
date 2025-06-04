/*****************************************************************/
/*    NAME: Mike McCarrick                                       */
/*    ORGN: NRL Code 5554                                        */
/*    FILE: BHV_RLAgent.h                                        */
/*    DATE: 20 Feb 2021                                          */
/*                                                               */
/* Based on BHV_SimpleWaypoint.h by M.Benjamin/MIT               */
/*****************************************************************/
 
#ifndef BHV_RLAGENT_HEADER
#define BHV_RLAGENT_HEADER

#include <string>
#include "IvPBehavior.h"
#include "XYPoint.h"

class BHV_RLAgent : public IvPBehavior {
public:
  BHV_RLAgent(IvPDomain);
  ~BHV_RLAgent() {};
  
  bool         setParam(std::string, std::string);
  void         onIdleState();
  IvPFunction* onRunState();

protected:
  IvPFunction* buildFunctionWithZAIC();

protected: // Configuration parameters


protected: // State variables
  double   m_desired_speed;
  double   m_desired_heading;
  double   m_osx;
  double   m_osy;
};

#ifdef WIN32
	// Windows needs to explicitly specify functions to export from a dll
   #define IVP_EXPORT_FUNCTION __declspec(dllexport) 
#else
   #define IVP_EXPORT_FUNCTION
#endif

extern "C" {
  IVP_EXPORT_FUNCTION IvPBehavior * createBehavior(std::string name, IvPDomain domain) 
  {return new BHV_RLAgent(domain);}
}
#endif
