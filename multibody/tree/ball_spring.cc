#include "drake/multibody/tree/ball_spring.h"

#include <utility>

#include "drake/multibody/tree/multibody_tree.h"

namespace drake {
namespace multibody {

using namespace drake::multibody;

template <typename T>
BallSpring<T>::BallSpring(const BallRpyJoint<T>& joint, const Vector3<double>& nominal_angles, 
                    const Vector3<double>& stiffness)
    : BallSpring(joint.model_instance(), joint.index(), nominal_angle,
                     stiffness) {}

template <typename T>
BallSpring<T>::BallSpring(ModelInstanceIndex model_instance,
                                  JointIndex joint_index, const Vector3<double>& nominal_angles,
                                  const Vector3<double>& stiffness)
    : ForceElement<T>(model_instance),
      joint_index_(joint_index),
      nominal_angles_(nominal_angles),
      stiffness_(stiffness) {
  DRAKE_THROW_UNLESS(stiffness >= 0);
}

template <typename T>
BallSpring<T>::~BallSpring() = default;

template <typename T>
const BallRpyJoint<T>& BallSpring<T>::joint() const {
  const BallRpyJoint<T>* joint = dynamic_cast<const BallRpyJoint<T>*>(
      &this->get_parent_tree().get_joint(joint_index_));
  DRAKE_DEMAND(joint != nullptr);
  return *joint;
}

template <typename T>
void BallSpring<T>::DoCalcAndAddForceContribution(
    const systems::Context<T>& context,
    const internal::PositionKinematicsCache<T>&,
    const internal::VelocityKinematicsCache<T>&,
    MultibodyForces<T>* forces) const {
  const T delta = nominal_angles_ - joint().get_angle(context);
  const T torque = stiffness_ * delta;
  joint().AddInTorque(context, torque, forces);
}

template <typename T>
T BallSpring<T>::CalcPotentialEnergy(
    const systems::Context<T>& context,
    const internal::PositionKinematicsCache<T>&) const {
  const T delta = nominal_angle_ - joint().get_angle(context);

  return 0.5 * stiffness_ * delta * delta;
}

template <typename T>
T BallSpring<T>::CalcConservativePower(
    const systems::Context<T>& context,
    const internal::PositionKinematicsCache<T>&,
    const internal::VelocityKinematicsCache<T>&) const {
  // Since the potential energy is:
  //   V = 1/2⋅k⋅(θ₀-θ)²
  // The conservative power is defined as:
  //  Pc = -d(V)/dt = -[k⋅(θ₀-θ)⋅-dθ/dt] = k⋅(θ₀-θ)⋅dθ/dt
  // being positive when the potential energy decreases.
  const T delta = nominal_angle_ - joint().get_angle(context);
  const T theta_dot = joint().get_angular_rate(context);
  return stiffness_ * delta * theta_dot;
}

template <typename T>
T BallSpring<T>::CalcNonConservativePower(
    const systems::Context<T>&, const internal::PositionKinematicsCache<T>&,
    const internal::VelocityKinematicsCache<T>&) const {
  // Purely conservative spring
  return 0;
}

template <typename T>
template <typename ToScalar>
std::unique_ptr<ForceElement<ToScalar>>
BallSpring<T>::TemplatedDoCloneToScalar(
    const internal::MultibodyTree<ToScalar>&) const {
  // N.B. We can't use std::make_unique here since this constructor is private
  // to std::make_unique.
  // N.B. We use the private constructor since it doesn't rely on a valid joint
  // reference, which might not be available during cloning.
  std::unique_ptr<BallSpring<ToScalar>> spring_clone(
      new BallSpring<ToScalar>(this->model_instance(), joint_index_,
                                   nominal_angle(), stiffness()));
  return spring_clone;
}

template <typename T>
std::unique_ptr<ForceElement<double>> BallSpring<T>::DoCloneToScalar(
    const internal::MultibodyTree<double>& tree_clone) const {
  return TemplatedDoCloneToScalar(tree_clone);
}

template <typename T>
std::unique_ptr<ForceElement<AutoDiffXd>> BallSpring<T>::DoCloneToScalar(
    const internal::MultibodyTree<AutoDiffXd>& tree_clone) const {
  return TemplatedDoCloneToScalar(tree_clone);
}

template <typename T>
std::unique_ptr<ForceElement<symbolic::Expression>>
BallSpring<T>::DoCloneToScalar(
    const internal::MultibodyTree<symbolic::Expression>& tree_clone) const {
  return TemplatedDoCloneToScalar(tree_clone);
}

template <typename T>
std::unique_ptr<ForceElement<T>> BallSpring<T>::DoShallowClone() const {
  // N.B. We use the private constructor since joint() requires a MbT pointer.
  return std::unique_ptr<ForceElement<T>>(new BallSpring<T>(
      this->model_instance(), joint_index_, nominal_angles(), stiffness()));
}

}  // namespace multibody
}  // namespace drake

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_SCALARS(
    class ::drake:multibody::BallSpring);
