/************************************************************/
/*    NAME: Mike McCarrick                                  */
/*    ORGN: NRL Code 5554                                   */
/*    FILE: main.cpp                                        */
/*    DATE: 23 Feb 2021                                     */
/************************************************************/

#include <string>
#include "MBUtils.h"
#include "ColorParse.h"
#include "RLMonitor.h"
#include "RLMonitor_Info.h"

using namespace std;

int main(int argc, char *argv[])
{
  string mission_file;
  string run_command = argv[0];

  for(int i=1; i<argc; i++) {
    string argi = argv[i];
    if((argi=="-v") || (argi=="--version") || (argi=="-version"))
      showReleaseInfoAndExit();
    else if((argi=="-e") || (argi=="--example") || (argi=="-example"))
      showExampleConfigAndExit();
    else if((argi == "-h") || (argi == "--help") || (argi=="-help"))
      showHelpAndExit();
    else if((argi == "-i") || (argi == "--interface"))
      showInterfaceAndExit();
    else if(strEnds(argi, ".moos") || strEnds(argi, ".moos++"))
      mission_file = argv[i];
    else if(strBegins(argi, "--alias="))
      run_command = argi.substr(8);
    else if(i==2)
      run_command = argi;
  }
  
  if(mission_file == "")
    showHelpAndExit();

  cout << termColor("green");
  cout << "pRLMonitor launching as " << run_command << endl;
  cout << termColor() << endl;

  RLMonitor RLMonitor;

  RLMonitor.Run(run_command.c_str(), mission_file.c_str());
  
  return(0);
}

