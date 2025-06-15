#include "drake/multibody/tree/quaternion_ball_mobilizer.h"

#include <memory>
#include <stdexcept>
#include <string>

#include "drake/common/eigen_types.h"
#include "drake/math/roll_pitch_yaw.h"
#include "drake/math/rotation_matrix.h"
#include "drake/multibody/tree/body_node_impl.h"
#include "drake/multibody/tree/multibody_tree.h"
#include "drake/multibody/tree/rigid_body.h"

namespace drake {
namespace multibody {
namespace internal {

template <typename T>
QuaternionBallMobilizer<T>::~QuaternionBallMobilizer() = default;

template <typename T>
std::unique_ptr<BodyNode<T>> QuaternionBallMobilizer<T>::CreateBodyNode(
    const BodyNode<T>* parent_node, const RigidBody<T>* body,
    const Mobilizer<T>* mobilizer) const {
  return std::make_unique<BodyNodeImpl<T, QuaternionBallMobilizer>>(
      parent_node, body, mobilizer);
}

template <typename T>
std::string QuaternionBallMobilizer<T>::position_suffix(
    int position_index_in_mobilizer) const {
  switch (position_index_in_mobilizer) {
    case 0:
      return "qw";
    case 1:
      return "qx";
    case 2:
      return "qy";
    case 3:
      return "qz";
  }
  throw std::runtime_error("QuaternionBallMobilizer has only 3 positions.");
}

template <typename T>
std::string QuaternionBallMobilizer<T>::velocity_suffix(
    int velocity_index_in_mobilizer) const {
  switch (velocity_index_in_mobilizer) {
    case 0:
      return "wx";
    case 1:
      return "wy";
    case 2:
      return "wz";
  }
  throw std::runtime_error("QuaternionBallMobilizer has only 3 velocities.");
}

template <typename T>
Quaternion<T> QuaternionBallMobilizer<T>::get_quaternion(
    const systems::Context<T>& context) const {
  const auto q = this->get_positions(context);
  DRAKE_ASSERT(q.size() == kNq);
  return Quaternion<T>(q[0], q[1], q[2], q[3]);
}

template <typename T>
const QuaternionBallMobilizer<T>& QuaternionBallMobilizer<T>::SetQuaternion(
    systems::Context<T>* context, const Quaternion<T>& q_FM) const {
  DRAKE_DEMAND(context != nullptr);
  SetQuaternion(*context, q_FM, &context->get_mutable_state());
  return *this;
}

template <typename T>
const QuaternionBallMobilizer<T>& QuaternionBallMobilizer<T>::SetQuaternion(
    const systems::Context<T>&, const Quaternion<T>& q_FM,
    systems::State<T>* state) const {
  DRAKE_DEMAND(state != nullptr);
  auto q = this->get_mutable_positions(state);
  DRAKE_ASSERT(q.size() == kNq);
  // Note: The storage order documented in get_quaternion() is consistent with
  // the order below, q[0] is the "scalar" part and q[1:3] is the "vector" part.
  q[0] = q_FM.w();
  q.template segment<3>(1) = q_FM.vec();
  return *this;
}

template <typename T>
Vector3<T> QuaternionBallMobilizer<T>::get_angular_velocity(
    const systems::Context<T>& context) const {
  return this->get_velocities(context);
}

template <typename T>
const QuaternionBallMobilizer<T>&
QuaternionBallMobilizer<T>::SetAngularVelocity(systems::Context<T>* context,
                                               const Vector3<T>& w_FM) const {
  return SetAngularVelocity(*context, w_FM, &context->get_mutable_state());
}

template <typename T>
const QuaternionBallMobilizer<T>&
QuaternionBallMobilizer<T>::SetAngularVelocity(const systems::Context<T>&,
                                               const Vector3<T>& w_FM,
                                               systems::State<T>* state) const {
  auto v = this->get_mutable_velocities(state);
  DRAKE_ASSERT(v.size() == kNv);
  v = w_FM;
  return *this;
}

template <typename T>
math::RigidTransform<T>
QuaternionBallMobilizer<T>::CalcAcrossMobilizerTransform(
    const systems::Context<T>& context) const {
  const auto& q = this->get_positions(context);
  DRAKE_ASSERT(q.size() == kNq);
  return calc_X_FM(q.data());
}

template <typename T>
SpatialVelocity<T>
QuaternionBallMobilizer<T>::CalcAcrossMobilizerSpatialVelocity(
    const systems::Context<T>&, const Eigen::Ref<const VectorX<T>>& v) const {
  DRAKE_ASSERT(v.size() == kNv);
  return calc_V_FM(nullptr, v.data());
}

template <typename T>
SpatialAcceleration<T>
QuaternionBallMobilizer<T>::CalcAcrossMobilizerSpatialAcceleration(
    const systems::Context<T>&,
    const Eigen::Ref<const VectorX<T>>& vdot) const {
  DRAKE_ASSERT(vdot.size() == kNv);
  return calc_A_FM(nullptr, nullptr, vdot.data());
}

template <typename T>
void QuaternionBallMobilizer<T>::ProjectSpatialForce(
    const systems::Context<T>&, const SpatialForce<T>& F_BMo_F,
    Eigen::Ref<VectorX<T>> tau) const {
  DRAKE_ASSERT(tau.size() == kNv);
  calc_tau(nullptr, F_BMo_F, tau.data());
}

template <typename T>
Eigen::Matrix<T, 4, 3> QuaternionBallMobilizer<T>::CalcLMatrix(
    const Quaternion<T>& q_FM) {
  const T qs = q_FM.w();             // The scalar component.
  const Vector3<T> qv = q_FM.vec();  // The vector component.
  const Vector3<T> mqv = -qv;        // minus qv.

  return (Eigen::Matrix<T, 4, 3>() << mqv.transpose(), qs, qv.z(), mqv.y(),
          mqv.z(), qs, qv.x(), qv.y(), mqv.x(), qs)
      .finished();
}

template <typename T>
Eigen::Matrix<T, 4, 3>
QuaternionBallMobilizer<T>::AngularVelocityToQuaternionRateMatrix(
    const Quaternion<T>& q_FM) {
  return CalcLMatrix(
      {q_FM.w() / 2.0, q_FM.x() / 2.0, q_FM.y() / 2.0, q_FM.z() / 2.0});
}

template <typename T>
Eigen::Matrix<T, 3, 4>
QuaternionBallMobilizer<T>::QuaternionRateToAngularVelocityMatrix(
    const Quaternion<T>& q_FM) {
  const T q_norm = q_FM.norm();
  const Vector4<T> q_FM_tilde =
      Vector4<T>(q_FM.w(), q_FM.x(), q_FM.y(), q_FM.z()) / q_norm;

  const Matrix4<T> dqnorm_dq =
      (Matrix4<T>::Identity() - q_FM_tilde * q_FM_tilde.transpose()) / q_norm;

  return CalcLMatrix({2.0 * q_FM_tilde[0], 2.0 * q_FM_tilde[1],
                      2.0 * q_FM_tilde[2], 2.0 * q_FM_tilde[3]})
             .transpose() *
         dqnorm_dq;
}

template <typename T>
auto QuaternionBallMobilizer<T>::get_zero_position() const -> QVector<double> {
  QVector<double> q = QVector<double>::Zero();
  const Quaternion<double> quaternion = Quaternion<double>::Identity();
  q[0] = quaternion.w();
  q.template segment<3>(1) = quaternion.vec();
  return q;
}

template <typename T>
void QuaternionBallMobilizer<T>::DoCalcNMatrix(
    const systems::Context<T>& context, EigenPtr<MatrixX<T>> N) const {
  *N = AngularVelocityToQuaternionRateMatrix(get_quaternion(context));
}

template <typename T>
void QuaternionBallMobilizer<T>::DoCalcNplusMatrix(
    const systems::Context<T>& context, EigenPtr<MatrixX<T>> Nplus) const {
  *Nplus = QuaternionRateToAngularVelocityMatrix(get_quaternion(context));
}

template <typename T>
void QuaternionBallMobilizer<T>::DoMapVelocityToQDot(
    const systems::Context<T>& context, const Eigen::Ref<const VectorX<T>>& v,
    EigenPtr<VectorX<T>> qdot) const {
  const Quaternion<T> q_FM = get_quaternion(context);
  *qdot = AngularVelocityToQuaternionRateMatrix(q_FM) * v.template head<3>();
}

template <typename T>
void QuaternionBallMobilizer<T>::DoMapQDotToVelocity(
    const systems::Context<T>& context,
    const Eigen::Ref<const VectorX<T>>& qdot, EigenPtr<VectorX<T>> v) const {
  const Quaternion<T> q_FM = get_quaternion(context);
  *v = QuaternionRateToAngularVelocityMatrix(q_FM) * qdot.template head<4>();
}

template <typename T>
template <typename ToScalar>
std::unique_ptr<Mobilizer<ToScalar>>
QuaternionBallMobilizer<T>::TemplatedDoCloneToScalar(
    const MultibodyTree<ToScalar>& tree_clone) const {
  const Frame<ToScalar>& inboard_frame_clone =
      tree_clone.get_variant(this->inboard_frame());
  const Frame<ToScalar>& outboard_frame_clone =
      tree_clone.get_variant(this->outboard_frame());
  return std::make_unique<QuaternionBallMobilizer<ToScalar>>(
      tree_clone.get_mobod(this->mobod().index()), inboard_frame_clone,
      outboard_frame_clone);
}

template <typename T>
std::unique_ptr<Mobilizer<double>> QuaternionBallMobilizer<T>::DoCloneToScalar(
    const MultibodyTree<double>& tree_clone) const {
  return TemplatedDoCloneToScalar(tree_clone);
}

template <typename T>
std::unique_ptr<Mobilizer<AutoDiffXd>>
QuaternionBallMobilizer<T>::DoCloneToScalar(
    const MultibodyTree<AutoDiffXd>& tree_clone) const {
  return TemplatedDoCloneToScalar(tree_clone);
}

template <typename T>
std::unique_ptr<Mobilizer<symbolic::Expression>>
QuaternionBallMobilizer<T>::DoCloneToScalar(
    const MultibodyTree<symbolic::Expression>& tree_clone) const {
  return TemplatedDoCloneToScalar(tree_clone);
}

}  // namespace internal
}  // namespace multibody
}  // namespace drake

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_SCALARS(
    class ::drake::multibody::internal::QuaternionBallMobilizer);
