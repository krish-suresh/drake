#pragma once

#include <limits>
#include <memory>
#include <string>
#include <utility>

#include "drake/common/default_scalars.h"
#include "drake/common/drake_copyable.h"
#include "drake/multibody/tree/joint.h"
#include "drake/multibody/tree/multibody_forces.h"
#include "drake/multibody/tree/quaternion_ball_mobilizer.h"

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
                     4, -std::numeric_limits<double>::infinity()),
                 VectorX<double>::Constant(
                     4, std::numeric_limits<double>::infinity()),
                 VectorX<double>::Constant(
                     3, -std::numeric_limits<double>::infinity()),
                 VectorX<double>::Constant(
                     3, std::numeric_limits<double>::infinity()),
                 VectorX<double>::Constant(
                     3, -std::numeric_limits<double>::infinity()),
                 VectorX<double>::Constant(
                     3, std::numeric_limits<double>::infinity())) {
    DRAKE_THROW_UNLESS(damping >= 0);

    this->set_default_quaternion(Quaternion<double>::Identity());
  }

  ~BallQuaternionJoint() final;

  const std::string& type_name() const final;

  double default_damping() const { return this->default_damping_vector()[0]; }

  Quaternion<T> get_quaternion(const systems::Context<T>& context) const {
    return get_mobilizer().get_quaternion(context);
  }

  const BallQuaternionJoint<T>& SetQuaternion(systems::Context<T>* context,
                                              const Quaternion<T>& q_FM) const {
    get_mobilizer().SetQuaternion(context, q_FM);
    return *this;
  }

  const BallQuaternionJoint<T>& SetOrientation(
      systems::Context<T>* context, const math::RotationMatrix<T>& R_FM) const {
    get_mobilizer().SetOrientation(context, R_FM);
    return *this;
  }

  Vector3<T> get_angular_velocity(const systems::Context<T>& context) const {
    return get_mobilizer().get_angular_velocity(context);
  }

  const BallQuaternionJoint<T>& set_angular_velocity(
      systems::Context<T>* context, const Vector3<T>& w_FM) const {
    get_mobilizer().SetAngularVelocity(context, w_FM);
    return *this;
  }

  void set_default_quaternion(const Quaternion<double>& q_FM) {
    VectorX<double> default_positions = this->default_positions();
    // @note we store the quaternion components consistently with
    // `QuaternionFloatingMobilizer<T>::get_quaternion()`
    default_positions[0] = q_FM.w();
    default_positions[1] = q_FM.x();
    default_positions[2] = q_FM.y();
    default_positions[3] = q_FM.z();
    this->set_default_positions(default_positions);
  }

  Quaternion<double> get_default_quaternion() const {
    const Vector4<double>& q_FM = this->default_positions().template head<4>();
    return Quaternion<double>(q_FM[0], q_FM[1], q_FM[2], q_FM[3]);
  }

 protected:
  void DoAddInOneForce(const systems::Context<T>&, int, const T&,
                       MultibodyForces<T>*) const final {
    throw std::logic_error(
        "Ball Quaternion joints do not allow applying forces to individual "
        "degrees of "
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
  int do_get_velocity_start() const final {
    return get_mobilizer().velocity_start_in_v();
  }

  int do_get_num_velocities() const final { return 3; }

  int do_get_position_start() const final {
    return get_mobilizer().position_start_in_q();
  }

  int do_get_num_positions() const final { return 4; }

  std::string do_get_position_suffix(int index) const final {
    return get_mobilizer().position_suffix(index);
  }

  std::string do_get_velocity_suffix(int index) const final {
    return get_mobilizer().velocity_suffix(index);
  }

  void do_set_default_positions(
      const VectorX<double>& default_positions) final {
    if (this->has_mobilizer()) {
      get_mutable_mobilizer().set_default_position(default_positions);
    }
  }

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

  const internal::QuaternionBallMobilizer<T>& get_mobilizer() const {
    return this
        ->template get_mobilizer_downcast<internal::QuaternionBallMobilizer>();
  }

  internal::QuaternionBallMobilizer<T>& get_mutable_mobilizer() {
    return this->template get_mutable_mobilizer_downcast<
        internal::QuaternionBallMobilizer>();
  }

  template <typename ToScalar>
  std::unique_ptr<Joint<ToScalar>> TemplatedDoCloneToScalar(
      const internal::MultibodyTree<ToScalar>& tree_clone) const;
};

template <typename T>
const char BallQuaternionJoint<T>::kTypeName[] = "ball_quaternion";
}  // namespace multibody
}  // namespace drake

DRAKE_DECLARE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_SCALARS(
    class ::drake::multibody::BallQuaternionJoint);