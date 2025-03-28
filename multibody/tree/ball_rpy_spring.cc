#include "drake/multibody/tree/ball_rpy_spring.h"

#include <utility>

#include "drake/common/unused.h"
#include "drake/multibody/tree/multibody_tree.h"
namespace drake {
namespace multibody {

// // using namespace drake::multibody;

template <typename T>
BallRpySpring<T>::BallRpySpring(const BallRpyJoint<T>& joint,
                                const Vector3<double>& nominal_angles,
                                const Vector3<double>& stiffness)
    : BallRpySpring(joint.model_instance(), joint.index(), nominal_angles,
                    stiffness) {}

template <typename T>
BallRpySpring<T>::BallRpySpring(ModelInstanceIndex model_instance,
                                JointIndex joint_index,
                                const Vector3<double>& nominal_angles,
                                const Vector3<double>& stiffness)
    : ForceElement<T>(model_instance),
      joint_index_(joint_index),
      nominal_angles_(nominal_angles),
      stiffness_(stiffness) {
  DRAKE_THROW_UNLESS((stiffness.array() >= 0).all());
}

template <typename T>
BallRpySpring<T>::~BallRpySpring() = default;

template <typename T>
const BallRpyJoint<T>& BallRpySpring<T>::joint() const {
  const BallRpyJoint<T>* joint = dynamic_cast<const BallRpyJoint<T>*>(
      &this->get_parent_tree().get_joint(joint_index_));
  DRAKE_DEMAND(joint != nullptr);
  return *joint;
}

template <typename T>
void BallRpySpring<T>::DoCalcAndAddForceContribution(
    const systems::Context<T>& context,
    const internal::PositionKinematicsCache<T>&,
    const internal::VelocityKinematicsCache<T>&,
    MultibodyForces<T>* forces) const {
  const Vector3<T> delta = nominal_angles_ - joint().get_angles(context);
  const Vector3<T> torque = stiffness_.array() * delta.array();
  joint().AddInTorque(context, torque, forces);
}

template <typename T>
T BallRpySpring<T>::CalcPotentialEnergy(
    const systems::Context<T>& context,
    const internal::PositionKinematicsCache<T>&) const {
  const Vector3<T> delta = nominal_angles_ - joint().get_angles(context);
  const Vector3<T> delta2 = delta.array() * delta.array();
  return 0.5 * stiffness_.dot(delta2);
}

template <typename T>
T BallRpySpring<T>::CalcConservativePower(
    const systems::Context<T>& context,
    const internal::PositionKinematicsCache<T>&,
    const internal::VelocityKinematicsCache<T>&) const {
  const Vector3<T> delta = nominal_angles_ - joint().get_angles(context);
  const Vector3<T> theta_dot = joint().get_angular_velocity(context);
  const Vector3<T> f = stiffness_.array() * delta.array();
  return f.dot(theta_dot);
}

template <typename T>
T BallRpySpring<T>::CalcNonConservativePower(
    const systems::Context<T>&, const internal::PositionKinematicsCache<T>&,
    const internal::VelocityKinematicsCache<T>&) const {
  // Purely conservative spring
  return 0;
}

template <typename T>
template <typename ToScalar>
std::unique_ptr<ForceElement<ToScalar>>
BallRpySpring<T>::TemplatedDoCloneToScalar(
    const internal::MultibodyTree<ToScalar>&) const {
  // N.B. We can't use std::make_unique here since this constructor is private
  // to std::make_unique.
  // N.B. We use the private constructor since it doesn't rely on a valid joint
  // reference, which might not be available during cloning.
  std::unique_ptr<BallRpySpring<ToScalar>> spring_clone(
      new BallRpySpring<ToScalar>(this->model_instance(), joint_index_,
                                  nominal_angles(), stiffness()));
  return spring_clone;
}

template <typename T>
std::unique_ptr<ForceElement<double>> BallRpySpring<T>::DoCloneToScalar(
    const internal::MultibodyTree<double>& tree_clone) const {
  return TemplatedDoCloneToScalar(tree_clone);
}

template <typename T>
std::unique_ptr<ForceElement<AutoDiffXd>> BallRpySpring<T>::DoCloneToScalar(
    const internal::MultibodyTree<AutoDiffXd>& tree_clone) const {
  return TemplatedDoCloneToScalar(tree_clone);
}

template <typename T>
std::unique_ptr<ForceElement<symbolic::Expression>>
BallRpySpring<T>::DoCloneToScalar(
    const internal::MultibodyTree<symbolic::Expression>& tree_clone) const {
  return TemplatedDoCloneToScalar(tree_clone);
}

template <typename T>
std::unique_ptr<ForceElement<T>> BallRpySpring<T>::DoShallowClone() const {
  // N.B. We use the private constructor since joint() requires a MbT pointer.
  return std::unique_ptr<ForceElement<T>>(new BallRpySpring<T>(
      this->model_instance(), joint_index_, nominal_angles(), stiffness()));
}

}  // namespace multibody
}  // namespace drake

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_SCALARS(
    class ::drake::multibody::BallRpySpring);