#pragma once

#include <memory>
#include <vector>

#include "drake/common/default_scalars.h"
#include "drake/common/drake_copyable.h"
#include "drake/multibody/tree/force_element.h"
#include "drake/multibody/tree/ball_rpy_joint.h"

namespace drake {
namespace multibody {

template <typename Scalar>
using Vector3 = Eigen::Matrix<Scalar, 3, 1>;

template <typename T>
class BallSpring final : public ForceElement<T> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(BallSpring);

  BallSpring(const BallRpyJoint<T>& joint, const Vector3<double>& nominal_angles,
                 const Vector3<double>& stiffness);

  ~BallSpring() override;

  const BallRpyJoint<T>& joint() const;

  const Vector3<double>& nominal_angles() const { return nominal_angles_; }

  const Vector3<double>& stiffness() const { return stiffness_; }

  T CalcPotentialEnergy(
      const systems::Context<T>& context,
      const internal::PositionKinematicsCache<T>& pc) const override;

  T CalcConservativePower(
      const systems::Context<T>& context,
      const internal::PositionKinematicsCache<T>& pc,
      const internal::VelocityKinematicsCache<T>& vc) const override;

  T CalcNonConservativePower(
      const systems::Context<T>& context,
      const internal::PositionKinematicsCache<T>& pc,
      const internal::VelocityKinematicsCache<T>& vc) const override;

 protected:
  void DoCalcAndAddForceContribution(
      const systems::Context<T>& context,
      const internal::PositionKinematicsCache<T>& pc,
      const internal::VelocityKinematicsCache<T>& vc,
      MultibodyForces<T>* forces) const override;

  std::unique_ptr<ForceElement<double>> DoCloneToScalar(
      const internal::MultibodyTree<double>& tree_clone) const override;

  std::unique_ptr<ForceElement<AutoDiffXd>> DoCloneToScalar(
      const internal::MultibodyTree<AutoDiffXd>& tree_clone) const override;

  std::unique_ptr<ForceElement<symbolic::Expression>> DoCloneToScalar(
      const internal::MultibodyTree<symbolic::Expression>&) const override;

  std::unique_ptr<ForceElement<T>> DoShallowClone() const override;

 private:
  // Allow different specializations to access each other's private data for
  // scalar conversion.
  template <typename U>
  friend class BallSpring;

  BallSpring(ModelInstanceIndex model_instance, JointIndex joint_index,
                 const Vector3<double>& nominal_angles, const Vector3<double>& stiffness);

  // Helper method to make a clone templated on ToScalar().
  template <typename ToScalar>
  std::unique_ptr<ForceElement<ToScalar>> TemplatedDoCloneToScalar(
      const internal::MultibodyTree<ToScalar>& tree_clone) const;

  const JointIndex joint_index_;
  const Vector3<double>& nominal_angles_;
  const Vector3<double>& stiffness_;
};

}  // namespace multibody
}  // namespace drake

DRAKE_DECLARE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_SCALARS(
    class ::drake::multibody::BallSpring);
