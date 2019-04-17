// -*- coding: utf-8 -*-
// Copyright (C) 2006-2011 Rosen Diankov <rosen.diankov@gmail.com>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
#ifndef RAVE_CONTROLLERS_H
#define RAVE_CONTROLLERS_H

#include <boost/bind.hpp>

class TestController : public ControllerBase
{
public:
    TestController(EnvironmentBasePtr penv) : ControllerBase(penv), cmdid(0), _bPause(false), _bIsDone(true), _bCheckCollision(false), _bThrowExceptions(false)
    {
        __description = ":Interface Author: Akihiko Yamaguchi\n\nTest controller used for planning and non-physics simulations. Forces exact robot positions.\n\n\
If \ref ControllerBase::SetPath is called and the trajectory finishes, then the controller will continue to set the trajectory's final joint values and transformation until one of three things happens:\n\n\
1. ControllerBase::SetPath is called.\n\n\
2. ControllerBase::SetDesired is called.\n\n\
3. ControllerBase::Reset is called resetting everything\n\n\
If SetDesired is called, only joint values will be set at every timestep leaving the transformation alone.\n";
        RegisterCommand("Pause",boost::bind(&TestController::_Pause,this,_1,_2),
                        "pauses the controller from reacting to commands ");
        RegisterCommand("SetCheckCollisions",boost::bind(&TestController::_SetCheckCollisions,this,_1,_2),
                        "If set, will check if the robot gets into a collision during movement");
        RegisterCommand("SetThrowExceptions",boost::bind(&TestController::_SetThrowExceptions,this,_1,_2),
                        "If set, will throw exceptions instead of print warnings");
        fTime = 0;
        _fSpeed = 1;
        _nControlTransformation = 0;
    }
    virtual ~TestController() {
    }

    virtual bool Init(RobotBasePtr robot, const std::vector<int>& dofindices, int nControlTransformation)
    {
        _probot = robot;
        if( flog.is_open() ) {
            flog.close();
        }
        if( !!_probot ) {
            string filename = _probot->GetName() + string(".traj");
            flog.open(filename.c_str());
            if( !flog ) {
                RAVELOG_WARN(str(boost::format("failed to open %s\n")%filename));
            }
            flog << "# " << GetXMLId() << " " << _probot->GetName() << endl << endl;
            _dofindices = dofindices;
            _nControlTransformation = nControlTransformation;
            _dofchecklimits.resize(0);
            FOREACH(it,_dofindices) {
                KinBody::JointPtr pjoint = _probot->GetJointFromDOFIndex(*it);
                _dofchecklimits.push_back(!pjoint->IsCircular(*it-pjoint->GetDOFIndex()));
            }
            _cblimits = _probot->RegisterChangeCallback(KinBody::Prop_JointLimits,boost::bind(&TestController::_SetJointLimits,boost::bind(&sptr_from<TestController>, weak_controller())));
            _SetJointLimits();
        }
        _bPause = false;
        return true;
    }

    virtual void Reset(int options)
    {
        _ptraj.reset();
        _vecdesired.resize(0);
        if( flog.is_open() ) {
            flog.close();
        }
        _bIsDone = true;
    }

    virtual const std::vector<int>& GetControlDOFIndices() const {
        return _dofindices;
    }
    virtual int IsControlTransformation() const {
        return _nControlTransformation;
    }

    virtual bool SetDesired(const std::vector<dReal>& values, TransformConstPtr trans)
    {
        if( values.size() != _dofindices.size() ) {
            throw openrave_exception(str(boost::format("wrong desired dimensions %d!=%d")%values.size()%_dofindices.size()),ORE_InvalidArguments);
        }
        fTime = 0;
        _ptraj.reset();
        // do not set done to true here! let it be picked up by the simulation thread.
        // this will also let it have consistent mechanics as SetPath
        // (there's a race condition we're avoiding where a user calls SetDesired and then state savers revert the robot)
        if( !_bPause ) {
            EnvironmentMutex::scoped_lock lockenv(_probot->GetEnv()->GetMutex());
            _vecdesired = values;
            if( _nControlTransformation ) {
                if( !!trans ) {
                    _tdesired = *trans;
                }
                else {
                    _tdesired = _probot->GetTransform();
                }
                _SetDOFValues(_vecdesired,_tdesired);
            }
            else {
                _SetDOFValues(_vecdesired);
            }
            _bIsDone = false;     // set after _vecdesired has changed
        }
        return true;
    }

    virtual bool SetPath(TrajectoryBaseConstPtr ptraj)
    {
        if( _bPause ) {
            RAVELOG_DEBUG("TestController cannot player trajectories when paused\n");
            _ptraj.reset();
            _bIsDone = true;
            return false;
        }
        if( !!ptraj &&( ptraj->GetDOF() != (int)_dofindices.size()) ) {
            throw openrave_exception(str(boost::format("wrong path dimensions %d!=%d")%ptraj->GetDOF()%_dofindices.size()),ORE_InvalidArguments);
        }
        _ptraj = ptraj;
        fTime = 0;
        _bIsDone = !_ptraj;
        _vecdesired.resize(0);

        if( !!_ptraj && !!flog ) {
            flog << endl << "trajectory: " << ++cmdid << endl;
            _ptraj->Write(flog, Trajectory::TO_IncludeTimestamps|Trajectory::TO_IncludeBaseTransformation);
        }

        return true;
    }

    virtual void SimulationStep(dReal fTimeElapsed)
    {
        if( _bPause ) {
            return;
        }
        if( !!_ptraj ) {
            Trajectory::TPOINT tp;
            if( !_ptraj->SampleTrajectory(fTime, tp) ) {
                return;
            }
            if( tp.q.size() > 0 ) {
                if( _nControlTransformation ) {
                    _SetDOFValues(tp.q,tp.trans);
                }
                else {
                    _SetDOFValues(tp.q);
                }
            }
            else if( _nControlTransformation ) {
                _probot->SetTransform(tp.trans);
            }

            if( fTime > _ptraj->GetTotalDuration() ) {
                fTime = _ptraj->GetTotalDuration();
                _bIsDone = true;
            }

            fTime += _fSpeed * fTimeElapsed;
        }

        if( _vecdesired.size() > 0 ) {
            if( _nControlTransformation ) {
                _SetDOFValues(_vecdesired,_tdesired);
            }
            else {
                _SetDOFValues(_vecdesired);
            }
            _bIsDone = true;
        }
    }

    virtual bool IsDone() {
        return _bIsDone;
    }
    virtual dReal GetTime() const {
        return fTime;
    }
    virtual RobotBasePtr GetRobot() const {
        return _probot;
    }

private:
    virtual bool _Pause(std::ostream& os, std::istream& is)
    {
        is >> _bPause;
        return !!is;
    }
    virtual bool _SetCheckCollisions(std::ostream& os, std::istream& is)
    {
        is >> _bCheckCollision;
        if( _bCheckCollision ) {
            _report.reset(new CollisionReport());
        }
        return !!is;
    }
    virtual bool _SetThrowExceptions(std::ostream& os, std::istream& is)
    {
        is >> _bThrowExceptions;
        return !!is;
    }

    inline boost::shared_ptr<TestController> shared_controller() {
        return boost::static_pointer_cast<TestController>(shared_from_this());
    }
    inline boost::shared_ptr<TestController const> shared_controller_const() const {
        return boost::static_pointer_cast<TestController const>(shared_from_this());
    }
    inline boost::weak_ptr<TestController> weak_controller() {
        return shared_controller();
    }

    virtual void _SetJointLimits()
    {
        if( !!_probot ) {
            _probot->GetDOFLimits(_vlower,_vupper);
        }
    }

    virtual void _SetDOFValues(const std::vector<dReal>& values)
    {
        vector<dReal> curvalues, curvel;
        _probot->GetDOFValues(curvalues);
        _probot->GetDOFVelocities(curvel);
        Vector linearvel, angularvel;
        _probot->GetLinks().at(0)->GetVelocity(linearvel,angularvel);
        int i = 0;
        FOREACH(it,_dofindices) {
            curvalues.at(*it) = values.at(i++);
            curvel.at(*it) = 0;
        }
        _CheckLimits(curvalues);
        _probot->SetDOFValues(curvalues,true);
        _probot->SetDOFVelocities(curvel,linearvel,angularvel);
        _CheckConfiguration();
    }
    virtual void _SetDOFValues(const std::vector<dReal>& values, const Transform& t)
    {
        BOOST_ASSERT(_nControlTransformation);
        vector<dReal> curvalues, curvel;
        _probot->GetDOFValues(curvalues);
        _probot->GetDOFVelocities(curvel);
        int i = 0;
        FOREACH(it,_dofindices) {
            curvalues.at(*it) = values.at(i++);
            curvel.at(*it) = 0;
        }
        _CheckLimits(curvalues);
        _probot->SetDOFValues(curvalues,t, true);
        _probot->SetDOFVelocities(curvel,Vector(),Vector());
        _CheckConfiguration();
    }

    void _CheckLimits(std::vector<dReal>& curvalues)
    {
        for(size_t i = 0; i < _vlower.size(); ++i) {
            if( _dofchecklimits[i] ) {
                if( curvalues.at(i) < _vlower[i]-5e-5f ) {
                    _ReportError(str(boost::format("robot %s dof %d is violating lower limit %s < %s")%_probot->GetName()%i%_vlower[i]%curvalues[i]));
                }
                if( curvalues.at(i) > _vupper[i]+5e-5f ) {
                    _ReportError(str(boost::format("robot %s dof %d is violating upper limit %s > %s")%_probot->GetName()%i%_vupper[i]%curvalues[i]));
                }
            }
        }
    }

    void _CheckConfiguration()
    {
        if( _bCheckCollision ) {
            if( GetEnv()->CheckCollision(KinBodyConstPtr(_probot),_report) ) {
                _ReportError(str(boost::format("collsion in trajectory: %s\n")%_report->__str__()));
            }
            if( _probot->CheckSelfCollision(_report) ) {
                _ReportError(str(boost::format("self collsion in trajectory: %s\n")%_report->__str__()));
            }
        }
    }

    void _ReportError(const std::string& s)
    {
        if( _bThrowExceptions ) {
            throw openrave_exception(s,ORE_Assert);
        }
        else {
            RAVELOG_WARN(s);
        }
    }

    RobotBasePtr _probot;               ///< controlled body
    dReal _fSpeed;                    ///< how fast the robot should go
    TrajectoryBaseConstPtr _ptraj;         ///< computed trajectory robot needs to follow in chunks of _pbody->GetDOF()

    dReal fTime;

    std::vector<dReal> _vecdesired;         ///< desired values of the joints
    Transform _tdesired;

    std::vector<int> _dofindices;
    std::vector<uint8_t> _dofchecklimits;
    std::vector<dReal> _vlower, _vupper;
    int _nControlTransformation;
    ofstream flog;
    int cmdid;
    bool _bPause, _bIsDone, _bCheckCollision, _bThrowExceptions;
    CollisionReportPtr _report;
    boost::shared_ptr<void> _cblimits;
};

#endif
