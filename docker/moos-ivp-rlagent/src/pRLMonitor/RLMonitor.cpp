/************************************************************/
/*    NAME: Mike McCarrick                                  */
/*    ORGN: NRL Code 5554                                   */
/*    FILE: RLMonitor.cpp                                   */
/*    DATE: 23 Feb 2021                                     */
/************************************************************/

#include <iterator>
#include "MBUtils.h"
#include "RLMonitor.h"

using namespace std;

//---------------------------------------------------------
// Constructor

RLMonitor::RLMonitor()
{
  // Initialize state variables
  m_rla_speed = 0.0;
  m_rla_heading = 0.0;
  m_rla_action_count = 0;
  m_blue_flag_grabbed = 0;
}

//---------------------------------------------------------
// Destructor

RLMonitor::~RLMonitor()
{
}

//---------------------------------------------------------
// Procedure: OnNewMail

bool RLMonitor::OnNewMail(MOOSMSG_LIST &NewMail)
{
  AppCastingMOOSApp::OnNewMail(NewMail);

  MOOSMSG_LIST::iterator p;
   
  for(p=NewMail.begin(); p!=NewMail.end(); p++) {
    CMOOSMsg &msg = *p;
    string key = msg.GetKey();
    double dval = msg.GetDouble();
    string sval = msg.GetString();
    
    if (key == "RLA_SPEED")
      m_rla_speed = dval;
    else if (key == "RLA_HEADING")
      m_rla_heading = dval;
    else if (key == "RLA_ACTION_COUNT")
      m_rla_action_count = dval;
    else if (key == "BLUE_FLAG_GRABBED")
      m_blue_flag_grabbed = (sval == "true");
  }
  return(true);
}

//---------------------------------------------------------
// Procedure: OnConnectToServer

bool RLMonitor::OnConnectToServer()
{
   RegisterVariables();
   return(true);
}

//---------------------------------------------------------
// Procedure: Iterate()
//            happens AppTick times per second

bool RLMonitor::Iterate()
{
  AppCastingMOOSApp::PostReport();
  return(true);
}

//---------------------------------------------------------
// Procedure: OnStartUp()
//            happens before connection is open

bool RLMonitor::OnStartUp()
{
  AppCastingMOOSApp::OnStartUp();

  list<string> sParams;
  m_MissionReader.EnableVerbatimQuoting(false);
  if(m_MissionReader.GetConfiguration(GetAppName(), sParams)) {
    list<string>::iterator p;
    for(p=sParams.begin(); p!=sParams.end(); p++) {
      string line  = *p;
      string param = tolower(biteStringX(line, '='));
      string value = line;
      
      if(param == "foo") {
        //handled
      }
      else if(param == "bar") {
        //handled
      }
    }
  }
  
  RegisterVariables();	
  return(true);
}

//---------------------------------------------------------
// Procedure: buildReport()
bool RLMonitor::buildReport()
{
  m_msgs << "speed: " << m_rla_speed << endl;
  m_msgs << "heading: " << m_rla_heading << endl;
  m_msgs << "action count: " << int(m_rla_action_count) << endl;
  m_msgs << "blue flag grabbed:  " << m_blue_flag_grabbed << endl;
  return(true);
}


//---------------------------------------------------------
// Procedure: RegisterVariables

void RLMonitor::RegisterVariables()
{
  AppCastingMOOSApp::RegisterVariables();
  Register("RLA_SPEED", 0);
  Register("RLA_HEADING", 0);
  Register("RLA_ACTION_COUNT", 0);
  Register("BLUE_FLAG_GRABBED", 0);
}

