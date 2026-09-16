#ifndef IOINTERFACE_H
#define IOINTERFACE_H

#include "message/LowlevelCmd.h"
#include "message/LowlevelState.h"
#include "interface/CmdPanel.h"
#include <string>

class IOInterface{
public:
IOInterface(){}
virtual ~IOInterface(){delete cmdPanel;}
virtual void receive(LowlevelState *state) = 0;
virtual void send(const LowlevelCmd *cmd) = 0;
virtual bool watchdogFault() const { return false; }
virtual void resetWatchdog() {}
virtual void sendRecv(const LowlevelCmd *cmd, LowlevelState *state) = 0;
void zeroCmdPanel(){cmdPanel->setZero();}
void setPassive(){cmdPanel->setPassive();}

protected:
CmdPanel *cmdPanel = nullptr;
};

#endif  //IOINTERFACE_H