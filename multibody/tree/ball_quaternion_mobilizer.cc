#include "drake/multibody/tree/ball_quaternion_mobilizer.h"

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
BallQuaternionMobilizer<T>::~BallQuaternionMobilizer() = default;

template <typename T>
std::unique_ptr<BodyNode<T>> BallQuaternionMobilizer<T>::CreateBodyNode(
    const BodyNode<T>* parent_node, const RigidBody<T>* body,
    const Mobilizer<T>* mobilizer) const {
  return std::make_unique<BodyNodeImpl<T, BallQuaternionMobilizer>>(parent_node, body,
                                                             mobilizer);
}

template <typename T>
std::string BallQuaternionMobilizer<T>::position_suffix(
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
  throw std::runtime_error("BallQuaternionMobilizer has only 3 positions.");
}

template <typename T>
std::string BallQuaternionMobilizer<T>::velocity_suffix(
    int velocity_index_in_mobilizer) const {
  switch (velocity_index_in_mobilizer) {
    case 0:
      return "wx";
    case 1:
      return "wy";
    case 2:
      return "wz";
  }
  throw std::runtime_error("BallQuaternionMobilizer has only 3 velocities.");
}

template <typename T>
Quaternion<T> BallQuaternionMobilizer<T>::get_quaternion(
    const systems::Context<T>& context) const {
  const auto q = this->get_positions(context);
  DRAKE_ASSERT(q.size() == kNq);
  return Quaternion<T>(q[0], q[1], q[2], q[3]);
}

template <typename T>
const BallQuaternionMobilizer<T>&
BallQuaternionMobilizer<T>::SetQuaternion(systems::Context<T>* context,
                                              const Quaternion<T>& q_FM) const {
  DRAKE_DEMAND(context != nullptr);
  SetQuaternion(*context, q_FM, &context->get_mutable_state());
  return *this;
}

template <typename T>
const BallQuaternionMobilizer<T>&
BallQuaternionMobilizer<T>::SetQuaternion(const systems::Context<T>&,
                                              const Quaternion<T>& q_FM,
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
Vector3<T> BallQuaternionMobilizer<T>::get_angular_velocity(
    const systems::Context<T>& context) const {
  return this->get_velocities(context);
}

template <typename T>
const BallQuaternionMobilizer<T>& BallQuaternionMobilizer<T>::SetAngularVelocity(
    systems::Context<T>* context, const Vector3<T>& w_FM) const {
  return SetAngularVelocity(*context, w_FM, &context->get_mutable_state());
}

template <typename T>
const BallQuaternionMobilizer<T>& BallQuaternionMobilizer<T>::SetAngularVelocity(
    const systems::Context<T>&, const Vector3<T>& w_FM,
    systems::State<T>* state) const {
  auto v = this->get_mutable_velocities(state);
  DRAKE_ASSERT(v.size() == kNv);
  v = w_FM;
  return *this;
}

template <typename T>
math::RigidTransform<T> BallQuaternionMobilizer<T>::CalcAcrossMobilizerTransform(
    const systems::Context<T>& context) const {
  const auto& q = this->get_positions(context);
  DRAKE_ASSERT(q.size() == kNq);
  return calc_X_FM(q.data());
}

template <typename T>
SpatialVelocity<T> BallQuaternionMobilizer<T>::CalcAcrossMobilizerSpatialVelocity(
    const systems::Context<T>&, const Eigen::Ref<const VectorX<T>>& v) const {
  DRAKE_ASSERT(v.size() == kNv);
  return calc_V_FM(nullptr, v.data());
}

template <typename T>
SpatialAcceleration<T>
BallQuaternionMobilizer<T>::CalcAcrossMobilizerSpatialAcceleration(
    const systems::Context<T>&,
    const Eigen::Ref<const VectorX<T>>& vdot) const {
  DRAKE_ASSERT(vdot.size() == kNv);
  return calc_A_FM(nullptr, nullptr, vdot.data());
}

template <typename T>
void BallQuaternionMobilizer<T>::ProjectSpatialForce(
    const systems::Context<T>&, const SpatialForce<T>& F_BMo_F,
    Eigen::Ref<VectorX<T>> tau) const {
  DRAKE_ASSERT(tau.size() == kNv);
  calc_tau(nullptr, F_BMo_F, tau.data());
}

template <typename T>
Eigen::Matrix<T, 4, 3> BallQuaternionMobilizer<T>::CalcLMatrix(
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
BallQuaternionMobilizer<T>::AngularVelocityToQuaternionRateMatrix(
    const Quaternion<T>& q_FM) {
  return CalcLMatrix(
      {q_FM.w() / 2.0, q_FM.x() / 2.0, q_FM.y() / 2.0, q_FM.z() / 2.0});
}

template <typename T>
Eigen::Matrix<T, 3, 4>
BallQuaternionMobilizer<T>::QuaternionRateToAngularVelocityMatrix(
    const Quaternion<T>& q_FM) {
  const T q_norm = q_FM.norm();
  const Vector4<T> q_FM_tilde =
      Vector4<T>(q_FM.w(), q_FM.x(), q_FM.y(), q_FM.z()) / q_norm;

  // Gradient of the normalized quaternion with respect to the unnormalized
  // generalized coordinates:
  const Matrix4<T> dqnorm_dq =
      (Matrix4<T>::Identity() - q_FM_tilde * q_FM_tilde.transpose()) / q_norm;

  return CalcLMatrix({2.0 * q_FM_tilde[0], 2.0 * q_FM_tilde[1],
                      2.0 * q_FM_tilde[2], 2.0 * q_FM_tilde[3]})
             .transpose() *
         dqnorm_dq;
}



template <typename T>
void BallQuaternionMobilizer<T>::DoCalcNMatrix(const systems::Context<T>& context,
                                        EigenPtr<MatrixX<T>> N) const {
  // The matrix N(q) relates q̇ to v as q̇ = N(q) * v, where q̇ = [ṙ, ṗ, ẏ]ᵀ and
  // v = w_FM_F = [ω0, ω1, ω2]ᵀ is the mobilizer M frame's angular velocity in
  // the mobilizer F frame, expressed in the F frame.
  //
  // ⌈ ṙ ⌉   ⌈          cos(y) / cos(p),           sin(y) / cos(p),  0 ⌉ ⌈ ω0 ⌉
  // | ṗ | = |                  -sin(y),                    cos(y),  0 | | ω1 |
  // ⌊ ẏ ⌋   ⌊ sin(p) * cos(y) / cos(p),  sin(p) * sin(y) / cos(p),  1 ⌋ ⌊ ω2 ⌋
  //
  // Note: N(q) is singular for p = π/2 + kπ, for k = ±1, ±2, ...
  // See related code and comments in MapVelocityToQdot().
}

template <typename T>
void BallQuaternionMobilizer<T>::DoCalcNplusMatrix(const systems::Context<T>& context,
                                            EigenPtr<MatrixX<T>> Nplus) const {
  // The matrix N⁺(q) relates v to q̇ as v = N⁺(q) * q̇, where q̇ = [ṙ, ṗ, ẏ]ᵀ and
  // v = w_FM_F = [ω0, ω1, ω2]ᵀ is the mobilizer M frame's angular velocity in
  // the mobilizer F frame, expressed in the F frame (thus w_FM_F = N⁺(q) * q̇).
  //
  // ⌈ ω0 ⌉   ⌈ cos(y) * cos(p),  -sin(y),  0 ⌉ ⌈ ṙ ⌉
  // | ω1 | = | sin(y) * cos(p),   cos(y),  0 | | ṗ |
  // ⌊ ω2 ⌋   ⌊         -sin(p),        0,  1 ⌋ ⌊ ẏ ⌋
  //
}

template <typename T>
void BallQuaternionMobilizer<T>::DoCalcNDotMatrix(const systems::Context<T>& context,
                                           EigenPtr<MatrixX<T>> Ndot) const {
  // Computes the 3x3 matrix Ṅ(q,q̇) that helps relate q̈ = Ṅ(q,q̇)⋅v + N(q)⋅v̇,
  // where q = [r, p, y]ᵀ contains the roll (r), pitch (p) and yaw (y) angles
  // and v = [wx, wy, wz]ᵀ represents W_FM_F (the angular velocity of the
  // mobilizer's M frame measured in its F frame, expressed in the F frame).
  //
  // The 3x3 matrix N(q) relates q̇ to v as q̇ = N(q)⋅v, where
  //
  //        [          cos(y) / cos(p),           sin(y) / cos(p),  0]
  // N(q) = [                  -sin(y),                    cos(y),  0]
  //        [ sin(p) * cos(y) / cos(p),  sin(p) * sin(y) / cos(p),  1]
  //
  //          ⌈ -sy/cp ẏ + cy sp/cp² ṗ    cy/cp ẏ + sy sp/cp² ṗ,   0 ⌉
  // Ṅ(q,q̇) = |                  -cy ẏ,                   -sy ẏ,   0 |
  //          ⌊  cy/cp² ṗ - sp sy/cp ẏ,   sy/cp² ṗ + sp cy/cp ẏ,   0 ⌋
  //
  // where cp = cos(p), sp = sin(p), cy = cos(y), sy = sin(y).
  // Note: Although the elements of Ṅ(q,q̇) are simply the time-derivatives of
  // corresponding elements of N(q), result were simplified as follows.
  // Ṅ[2, 0] = cy ṗ + sp² cy/cp² ṗ - sp sy/cp ẏ
  //         =            cy/cp² ṗ - sp sy/cp ẏ.
  // Ṅ[2, 1] = sy ṗ + sp² sy/cp² ṗ + sp cy/cp ẏ
  //         =            sy/cp² ṗ + sp cy/cp ẏ.

}

template <typename T>
void BallQuaternionMobilizer<T>::DoCalcNplusDotMatrix(
    const systems::Context<T>& context, EigenPtr<MatrixX<T>> NplusDot) const {
  // Computes the matrix Ṅ⁺(q,q̇) that helps relate v̇ = Ṅ⁺(q,q̇)⋅q̇ + N⁺(q)⋅q̈,
  // where q = [r, p, y]ᵀ contains the roll (r), pitch (p) and yaw (y) angles
  // and v = [wx, wy, wz]ᵀ represents W_FM_F (the angular velocity of the
  // mobilizer's M frame measured in its F frame, expressed in the F frame).
  //
  // The 3x3 matrix N⁺(q) relates v to q̇ as v = N⁺(q)⋅q̇, where
  //
  //         [ cos(y) * cos(p),  -sin(y),  0]
  // N⁺(q) = [ sin(y) * cos(p),   cos(y),  0]
  //         [         -sin(p),        0,  1]
  //
  //           ⌈ -sy cp ẏ - cy sp ṗ,   -cy ẏ,   0 ⌉
  // Ṅ⁺(q,q̇) = |  cy cp ẏ - sy sp ṗ    -sy ẏ,   0 |
  //           ⌊              -cp ṗ,       0,   0 ⌋
  //
  // where cp = cos(p), sp = sin(p), cy = cos(y), sy = sin(y).
}

template <typename T>
void BallQuaternionMobilizer<T>::DoMapVelocityToQDot(
    const systems::Context<T>& context, const Eigen::Ref<const VectorX<T>>& v,
    EigenPtr<VectorX<T>> qdot) const {
  // The matrix N(q) relates q̇ to v as q̇ = N(q) * v, where q̇ = [ṙ, ṗ, ẏ]ᵀ and
  // v = w_FM_F = [ω0, ω1, ω2]ᵀ is the mobilizer M frame's angular velocity in
  // the mobilizer F frame, expressed in the F frame.
  //
  // ⌈ ṙ ⌉   ⌈          cos(y) / cos(p),           sin(y) / cos(p),  0 ⌉ ⌈ ω0 ⌉
  // | ṗ | = |                  -sin(y),                    cos(y),  0 | | ω1 |
  // ⌊ ẏ ⌋   ⌊ sin(p) * cos(y) / cos(p),  sin(p) * sin(y) / cos(p),  1 ⌋ ⌊ ω2 ⌋
  //
  // Note: N(q) is singular for p = π/2 + kπ, for k = ±1, ±2, ...
  // See related code and comments in CalcNMatrix().
  // Note: The calculation below is more efficient than calculating N(q) * v.
  //
  // Developer note: N(q) is calculated by first forming w_FM by adding three
  // angular velocities, each related to an Euler angle rate (ṙ or ṗ or ẏ) in
  // various frames (frame F, two intermediate frames, and frame M). This is
  // discussed in [Diebel 2006, §5.2; Mitiguy (August 2019, §9.1].
  // Note: Diebel's eq. 67 rotation matrix is the transpose of our R_FM. Still
  // the expression for N(q) in [Diebel 2006], Eq. 76, is the same as herein.
  //
  // [Diebel 2006] Representing attitude: Euler angles, unit quaternions, and
  //               rotation vectors. Stanford University.
  // [Mitiguy August 2019] Mitiguy, P., 2019. Advanced Dynamics & Motion
  //                       Simulation.

}

template <typename T>
void BallQuaternionMobilizer<T>::DoMapQDotToVelocity(
    const systems::Context<T>& context,
    const Eigen::Ref<const VectorX<T>>& qdot, EigenPtr<VectorX<T>> v) const {
  // The matrix N⁺(q) relates v to q̇ as v = N⁺(q) * q̇, where q̇ = [ṙ, ṗ, ẏ]ᵀ and
  // v = w_FM_F = [ω0, ω1, ω2]ᵀ is the mobilizer M frame's angular velocity in
  // the mobilizer F frame, expressed in the F frame (thus w_FM_F = N⁺(q) * q̇).
  //
  // ⌈ ω0 ⌉   ⌈ cos(y) * cos(p),  -sin(y),  0 ⌉ ⌈ ṙ ⌉
  // | ω1 | = | sin(y) * cos(p),   cos(y),  0 | | ṗ |
  // ⌊ ω2 ⌋   ⌊         -sin(p),        0,  1 ⌋ ⌊ ẏ ⌋
  //
  // See related code and comments in DoCalcNplusMatrix().
  //
  // Developer note: N(q) is calculated by first forming w_FM by adding three
  // angular velocities, each related to an Euler angle rate (ṙ or ṗ or ẏ) in
  // various frames (frame F, two intermediate frames, and frame M). This is
  // discussed in [Diebel 2006, §5.2; Mitiguy (August 2019, §9.1].
  // Note: Diebel's eq. 67 rotation matrix is the transpose of our R_FM. Still
  // the expression for N(q) in [Diebel 2006], Eq. 76, is the same as herein.
  //
  // [Diebel 2006] Representing attitude: Euler angles, unit quaternions, and
  //               rotation vectors. Stanford University.
  // [Mitiguy August 2019] Mitiguy, P., 2019. Advanced Dynamics & Motion
  //                       Simulation.

}

template <typename T>
template <typename ToScalar>
std::unique_ptr<Mobilizer<ToScalar>>
BallQuaternionMobilizer<T>::TemplatedDoCloneToScalar(
    const MultibodyTree<ToScalar>& tree_clone) const {
  const Frame<ToScalar>& inboard_frame_clone =
      tree_clone.get_variant(this->inboard_frame());
  const Frame<ToScalar>& outboard_frame_clone =
      tree_clone.get_variant(this->outboard_frame());
  return std::make_unique<BallQuaternionMobilizer<ToScalar>>(
      tree_clone.get_mobod(this->mobod().index()), inboard_frame_clone,
      outboard_frame_clone);
}

template <typename T>
std::unique_ptr<Mobilizer<double>> BallQuaternionMobilizer<T>::DoCloneToScalar(
    const MultibodyTree<double>& tree_clone) const {
  return TemplatedDoCloneToScalar(tree_clone);
}

template <typename T>
std::unique_ptr<Mobilizer<AutoDiffXd>> BallQuaternionMobilizer<T>::DoCloneToScalar(
    const MultibodyTree<AutoDiffXd>& tree_clone) const {
  return TemplatedDoCloneToScalar(tree_clone);
}

template <typename T>
std::unique_ptr<Mobilizer<symbolic::Expression>>
BallQuaternionMobilizer<T>::DoCloneToScalar(
    const MultibodyTree<symbolic::Expression>& tree_clone) const {
  return TemplatedDoCloneToScalar(tree_clone);
}

}  // namespace internal
}  // namespace multibody
}  // namespace drake

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_SCALARS(
    class ::drake::multibody::internal::BallQuaternionMobilizer);
