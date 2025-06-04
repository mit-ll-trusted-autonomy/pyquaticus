/************************************************************/
/*    NAME: Mike McCarrick                                  */
/*    ORGN: NRL Code 5554                                   */
/*    FILE: RLMonitor.h                                     */
/*    DATE: 23 Feb 2021                                     */
/************************************************************/

#ifndef RLMonitor_HEADER
#define RLMonitor_HEADER

#include "MOOS/libMOOS/MOOSLib.h"
#include "MOOS/libMOOS/Thirdparty/AppCasting/AppCastingMOOSApp.h"

class RLMonitor : public AppCastingMOOSApp
{
public:
  RLMonitor();
  ~RLMonitor();

protected: // Standard MOOSApp functions to overload  
  bool OnNewMail(MOOSMSG_LIST &NewMail);
  bool Iterate();
  bool OnConnectToServer();
  bool OnStartUp();
  bool buildReport();

protected:
  void RegisterVariables();

private: // Configuration variables

private: // State variables TODO
  double m_rla_speed;
  double m_rla_heading;
  double m_rla_action_count;
  bool   m_blue_flag_grabbed;
};

#endif 
