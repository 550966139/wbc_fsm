#ifndef CTRLCOMPONENTS_H
#define CTRLCOMPONENTS_H

#include "message/LowlevelCmd.h"
#include "message/LowlevelState.h"
#include "interface/IOInterface.h"
#include "interface/CmdPanel.h"
#include "common/safety_supervisor.h"
#include <array>
#include <string>
#include <iostream>


struct CtrlComponents{
public:
    CtrlComponents(IOInterface *ioInter):ioInter(ioInter){
        lowCmd = new LowlevelCmd();
        lowState = new LowlevelState();
        exitFlag = false;
    }
    ~CtrlComponents(){
        delete lowCmd;
        delete lowState;
        delete ioInter;
    }
    LowlevelCmd *lowCmd;
    LowlevelState *lowState;
    IOInterface *ioInter;

    double dt;
    bool *running;
    bool exitFlag;
    CtrlPlatform ctrlPlatform;
    control::SafetySupervisor safety;

    bool safetyCheck() {
        const auto now = control::SafetyClock::now();
        const auto gravity = getGravity();
        return safety.checkState(lowState->received, lowState->receivedAt, gravity, now);
    }
    std::array<float, 3> getGravity() const {
        Vec3 g = lowState->getRotMat().transpose() * Vec3(0.f, 0.f, -1.f);
        return {g[0], g[1], g[2]};
    }
    void setDampingCommand() {
        for (auto &motor : lowCmd->motorCmd) {
            motor.q = motor.dq = motor.tau = motor.Kp = 0.f;
            motor.Kd = 3.f;
        }
    }

    void sendRecv(){ ioInter->sendRecv(lowCmd, lowState); }



};

#endif  // CTRLCOMPONENTS_H
