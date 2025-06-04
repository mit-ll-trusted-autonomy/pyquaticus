/*****************************************************************/
/*    NAME: Mike McCarrick                                       */
/*    ORGN: NRL Code 5554                                        */
/*    FILE: BHV_RLAgent.cpp                                      */
/*    DATE: 20 Feb 2021                                          */
/*                                                               */
/* Based on BHV_SimpleWaypoint.cpp by M.Benjamin/MIT             */
/*                                                               */
/* This IvP Behavior simply accepts desired speed/heading values */
/* by registering for the variables RLA_SPEED and RLA_HEADING    */
/* and returns an appropriate ivp function. The Reinforcement    */
/* Learning Agent (an independent process defined elsewhere)     */
/* controls the vehicle by posting these messages.               */
/*****************************************************************/

#include <cstdlib>
#include <math.h>
#include "BHV_RLAgent.h"
#include "MBUtils.h"
#include "AngleUtils.h"
#include "BuildUtils.h"
#include "ZAIC_PEAK.h"
#include "OF_Coupler.h"

using namespace std;

//-----------------------------------------------------------
// Procedure: Constructor

BHV_RLAgent::BHV_RLAgent(IvPDomain gdomain) : 
  IvPBehavior(gdomain)
{
  IvPBehavior::setParam("name", "rlagent");
  m_domain = subDomain(m_domain, "course,speed");

  // All distances are in meters, all speed in meters per second
  // Default values for behavior state variables
  m_desired_speed   = 0;
  m_desired_heading = 0;
  m_osx  = 0;
  m_osy  = 0;

  addInfoVars("NAV_X, NAV_Y");
  addInfoVars("RLA_SPEED, RLA_HEADING", "no_warning");
}

//---------------------------------------------------------------
// Procedure: setParam - handle behavior configuration parameters

bool BHV_RLAgent::setParam(string param, string val) 
{
  // Convert the parameter to lower case for more general matching
  param = tolower(param);

  double double_val = atof(val.c_str());
  if((param == "speed") && (double_val > 0) && (isNumber(val))) {
    m_desired_speed = double_val;
    return(true);
  }
  else if((param == "heading") && (double_val > 0) && (isNumber(val))) {
    m_desired_heading = double_val;
    return(true);
  }
  return(false);
}

//-----------------------------------------------------------
// Procedure: onIdleState

void BHV_RLAgent::onIdleState() 
{
}

//-----------------------------------------------------------
// Procedure: onRunState

IvPFunction *BHV_RLAgent::onRunState() 
{
  // Get vehicle position from InfoBuffer and post a 
  // warning if problem is encountered
  bool ok1, ok2;
  m_osx = getBufferDoubleVal("NAV_X", ok1);
  m_osy = getBufferDoubleVal("NAV_Y", ok2);
  if(!ok1 || !ok2) {
    postWMessage("No ownship X/Y info in info_buffer.");
    return(0);
  }

  // Update speed/heading if we get new values
  double speed = getBufferDoubleVal("RLA_SPEED", ok1);
  double heading = getBufferDoubleVal("RLA_HEADING", ok2);
  if (ok1 && ok2) {
    m_desired_speed = speed;
    m_desired_heading = heading;
  }
  
  // Build the IvP function with the ZAIC tool 
  IvPFunction *ipf = 0;
  ipf = buildFunctionWithZAIC();
  if(ipf == 0) 
    postWMessage("Problem Creating the IvP Function");

  if(ipf)
    ipf->setPWT(m_priority_wt);
  
  return(ipf);
}

//-----------------------------------------------------------
// Procedure: buildFunctionWithZAIC

IvPFunction *BHV_RLAgent::buildFunctionWithZAIC() 
{
  ZAIC_PEAK spd_zaic(m_domain, "speed");
  spd_zaic.setSummit(m_desired_speed);
  spd_zaic.setPeakWidth(0.5);
  spd_zaic.setBaseWidth(1.0);
  spd_zaic.setSummitDelta(0.8);  
  if(spd_zaic.stateOK() == false) {
    string warnings = "Speed ZAIC problems " + spd_zaic.getWarnings();
    postWMessage(warnings);
    return(0);
  }
  
  ZAIC_PEAK crs_zaic(m_domain, "course");
  crs_zaic.setSummit(m_desired_heading);
  crs_zaic.setPeakWidth(0);
  crs_zaic.setBaseWidth(180.0);
  crs_zaic.setSummitDelta(0);  
  crs_zaic.setValueWrap(true);
  if(crs_zaic.stateOK() == false) {
    string warnings = "Course ZAIC problems " + crs_zaic.getWarnings();
    postWMessage(warnings);
    return(0);
  }

  IvPFunction *spd_ipf = spd_zaic.extractIvPFunction();
  IvPFunction *crs_ipf = crs_zaic.extractIvPFunction();

  OF_Coupler coupler;
  IvPFunction *ivp_function = coupler.couple(crs_ipf, spd_ipf, 50, 50);

  return(ivp_function);
}
