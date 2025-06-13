#pragma once

#include <limits>
#include <memory>
#include <string>
#include <utility>

#include "drake/common/default_scalars.h"
#include "drake/common/drake_copyable.h"
#include "drake/multibody/tree/joint.h"
#include "drake/multibody/tree/multibody_forces.h"
#include "drake/multibody/tree/ball_quaternion_mobilizer.h"


namespace drake {
namespace multibody {
template <typename T>
class BallQuaternionJoint final : public Joint<T> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(BallQuaternionJoint);

  template <typename Scalar>
  using Context = systems::Context<Scalar>;

  static const char kTypeName[];

  BallQuaternionJoint(const std::string& name, const Frame<T>& frame_on_parent,
               const Frame<T>& frame_on_child, double damping = 0)
      : Joint<T>(name, frame_on_parent, frame_on_child,
                 VectorX<double>::Constant(3, damping),
                 VectorX<double>::Constant(
                     3, -std::numeric_limits<double>::infinity()),
                 VectorX<double>::Constant(
                     3, std::numeric_limits<double>::infinity()),
                 VectorX<double>::Constant(
                     3, -std::numeric_limits<double>::infinity()),
                 VectorX<double>::Constant(
                     3, std::numeric_limits<double>::infinity()),
                 VectorX<double>::Constant(
                     3, -std::numeric_limits<double>::infinity()),
                 VectorX<double>::Constant(
                     3, std::numeric_limits<double>::infinity())) {
    DRAKE_THROW_UNLESS(damping >= 0);
  }

  ~BallQuaternionJoint() final;

  const std::string& type_name() const final;

  double default_damping() const {
    return this->default_damping_vector()[0];
  }

  Vector3<T> get_angles(const Context<T>& context) const {
    return get_mobilizer().get_angles(context);
  }

  const BallQuaternionJoint<T>& set_angles(Context<T>* context,
                                    const Vector3<T>& angles) const {
    get_mobilizer().SetAngles(context, angles);
    return *this;
  }

 protected:
  void DoAddInOneForce(const systems::Context<T>&, int, const T&,
                       MultibodyForces<T>*) const final {
    throw std::logic_error(
        "Ball Quaternion joints do not allow applying forces to individual degrees of "
        "freedom.");
  }

  void DoAddInDamping(const systems::Context<T>& context,
                      MultibodyForces<T>* forces) const final {
    Eigen::Ref<VectorX<T>> t_BMo_F =
        get_mobilizer().get_mutable_generalized_forces_from_array(
            &forces->mutable_generalized_forces());
    const Vector3<T>& w_FM = get_angular_velocity(context);
    t_BMo_F = -this->GetDampingVector(context)[0] * w_FM;
  }

 private:
  // Joint<T> overrides:
  std::unique_ptr<internal::Mobilizer<T>> MakeMobilizerForJoint(
      const internal::SpanningForest::Mobod& mobod,
      internal::MultibodyTree<T>* tree) const final;

  std::unique_ptr<Joint<double>> DoCloneToScalar(
      const internal::MultibodyTree<double>& tree_clone) const final;

  std::unique_ptr<Joint<AutoDiffXd>> DoCloneToScalar(
      const internal::MultibodyTree<AutoDiffXd>& tree_clone) const final;

  std::unique_ptr<Joint<symbolic::Expression>> DoCloneToScalar(
      const internal::MultibodyTree<symbolic::Expression>&) const final;

  std::unique_ptr<Joint<T>> DoShallowClone() const final;

  template <typename>
  friend class BallQuaternionJoint;

  const internal::BallQuaternionMobilizer<T>& get_mobilizer() const {
    return this->template get_mobilizer_downcast<internal::BallQuaternionMobilizer>();
  }

  internal::BallQuaternionMobilizer<T>& get_mutable_mobilizer() {
    return this
        ->template get_mutable_mobilizer_downcast<internal::BallQuaternionMobilizer>();
  }

  template <typename ToScalar>
  std::unique_ptr<Joint<ToScalar>> TemplatedDoCloneToScalar(
      const internal::MultibodyTree<ToScalar>& tree_clone) const;
};

template <typename T>
const char BallQuaternionJoint<T>::kTypeName[] = "ball_quaternion";
}
}

DRAKE_DECLARE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_SCALARS(
    class ::drake::multibody::BallQuaternionJoint);